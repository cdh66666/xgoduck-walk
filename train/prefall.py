"""11-28 Molly: for falls at speed, what comes first in the 0.5 s before a fall - knee voltage saturation /
tracking error, or foot slip? Compare fall windows with windows from envs that stay up. Sim only."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
from duck_real import BASE_TASK, OnnxPolicy, apply_real

PER_CASE, W, DT, VOLT = 96, 25, 0.02, 7.3
CASES = [("forward 1.00", 1.0), ("forward 1.20", 1.2)]
P = json.loads(Path("src/mjlab_microduck/robot/xgoduck/params/1910_m6.json").read_text())
KT, RES = P["kt"], P["R"]


def run(path):
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    n = PER_CASE * len(CASES)
    cfg = apply_real(load_env_cfg(BASE_TASK, play=True), vin=(7.3, 7.7), vin_min=6.9, drop=(.05, .2), train=True)
    cfg.events.pop("sudden_brake", None)
    cfg.scene.num_envs, cfg.episode_length_s, cfg.seed = n, 30, 11
    c = cfg.commands["twist"]
    c.resampling_time_range = (1000.0, 1000.0)
    c.ranges.lin_vel_x, c.ranges.lin_vel_y, c.ranges.ang_vel_z = (0, 0), (0, 0), (0, 0)
    for a in ("rel_standing_envs", "rel_forward_envs", "rel_turn_in_place_envs", "rel_backward_envs",
              "rel_heading_envs", "independent_command_fraction"):
        if hasattr(c, a): setattr(c, a, 0.0)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    policy = OnnxPolicy(path)
    obs, _ = env.reset()
    term, robot = env.command_manager.get_term("twist"), env.scene["robot"]
    names = list(robot.joint_names)
    knees = [i for i, x in enumerate(names) if "knee" in x and not x.startswith("passive")]
    act = list(env.action_manager._terms.values())[0]
    tnames = list(act.target_names)
    kt_idx = [tnames.index(names[i]) for i in knees]
    sites = list(robot.site_names); feet = [sites.index("left_foot"), sites.index("right_foot")]
    target = torch.tensor([x[1] for x in CASES], device=env.device).repeat_interleave(PER_CASE)
    hist = torch.zeros(W, n, 4, device=env.device)  # knee sat, knee |err| (rad), foot slip (m/s), tilt
    ptr, alive = 0, torch.ones(n, dtype=torch.bool, device=env.device)
    fallw, stablew, steps = [], [], int(9.0 / DT)
    with torch.no_grad():
        for t in range(steps):
            s = t * DT
            term.vel_command_b[:] = 0.0
            if s >= 1.0: term.vel_command_b[:, 0] = target
            obs, _, done, tout, _ = env.step(policy(obs["actor"]))
            qd = robot.data.joint_vel[:, knees]; tau = robot.data.qfrc_actuator[:, knees]
            avail = KT * (VOLT - KT * qd * torch.sign(tau)) / RES
            sat = (tau.abs() >= 0.9 * avail.clamp(min=1e-6)).float().max(1).values
            err = (act._processed_actions[:, kt_idx] - robot.data.joint_pos[:, knees]).abs().max(1).values
            sp, sv = robot.data.site_pos_w[:, feet], robot.data.site_lin_vel_w[:, feet]
            on = (sp[..., 2] < 0.012).float()
            slip = (sv[..., :2].norm(dim=-1) * on).max(1).values
            g = robot.data.projected_gravity_b; tilt = torch.atan2(g[:, :2].norm(dim=1), -g[:, 2])
            hist[ptr % W] = torch.stack([sat, err, slip, tilt], 1); ptr += 1
            fell = done.bool() & ~tout.bool() & alive
            if s > 2.5 and ptr >= W:
                order = [(ptr + k) % W for k in range(W)]
                for i in fell.nonzero().flatten().tolist():
                    fallw.append((int(i // PER_CASE), hist[order, i].cpu()))
                if t % 50 == 0:
                    for i in (alive & ~fell).nonzero().flatten().tolist():
                        stablew.append((int(i // PER_CASE), hist[order, i].cpu()))
            alive &= ~done.bool()
    env.close()
    return fallw, stablew


def summarize(label, ws):
    if not ws: print(f"RESULT {label}: none"); return
    x = torch.stack([w for _, w in ws])  # N, W, 4
    early, late = x[:, :12].mean((0, 1)), x[:, 12:20].mean((0, 1))
    print(f"RESULT {label:7s} n={len(ws):3d} | -0.5..-0.26 s: knee sat {early[0]*100:4.0f}%  knee err {early[1]:.3f} rad  slip {early[2]:.3f} m/s  tilt {early[3]*57.3:4.1f} deg"
          f" | -0.26..-0.1 s: sat {late[0]*100:4.0f}%  err {late[1]:.3f}  slip {late[2]:.3f}  tilt {late[3]*57.3:4.1f}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        f, s = run(path)
        print(f"RESULT model {label}")
        summarize("fall", f); summarize("stable", s)

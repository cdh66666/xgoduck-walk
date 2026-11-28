"""Do the servos hit their speed or voltage limit during fast walking and turning? No training, sim only.
Usage (from ~/duck/xgoduck_rl): .venv/bin/python saturation_check.py name=model.onnx ..."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from duck_real import BASE_TASK, OnnxPolicy, apply_real

PER_CASE = 48
CASES = [("forward 0.50", .50, 0, 0), ("forward 0.70", .70, 0, 0), ("forward 1.00", 1.0, 0, 0), ("forward 1.20", 1.2, 0, 0),
         ("back -1.00", -1.0, 0, 0)]  # 11-28 Molly: why do I/J fall at speed - servo limit, slip, or policy
STAND_S, HOLD_S, DT, VOLT = 1.0, 8.0, 0.02, 7.3
P = json.loads(Path("src/mjlab_microduck/robot/xgoduck/params/1910_m6.json").read_text())
KT, RES = P["kt"], P["R"]


def joint_torque(env, robot):
    tau = robot.data.qfrc_actuator  # joint space, same order as joint_vel
    assert tau.shape == robot.data.joint_vel.shape, (tau.shape, robot.data.joint_vel.shape)
    return tau


def run(path):
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    n = PER_CASE * len(CASES)
    cfg = apply_real(load_env_cfg(BASE_TASK, play=True), vin=(7.3, 7.7), vin_min=6.9, drop=(.05, .2), train=True)
    cfg.events.pop("sudden_brake", None)
    cfg.scene.num_envs, cfg.episode_length_s, cfg.seed = n, 30, 7
    c = cfg.commands["twist"]
    c.resampling_time_range = (1000.0, 1000.0)
    c.ranges.lin_vel_x, c.ranges.lin_vel_y, c.ranges.ang_vel_z = (0, 0), (0, 0), (0, 0)
    for a in ("rel_standing_envs", "rel_forward_envs", "rel_turn_in_place_envs", "rel_backward_envs",
              "rel_heading_envs", "independent_command_fraction"):
        if hasattr(c, a):
            setattr(c, a, 0.0)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    policy = OnnxPolicy(path)
    obs, _ = env.reset()
    term, robot = env.command_manager.get_term("twist"), env.scene["robot"]
    names = list(robot.joint_names)
    legs = [i for i, x in enumerate(names) if any(k in x for k in ("hip", "knee", "ankle"))]
    target = torch.tensor([x[1:] for x in CASES], device=env.device, dtype=torch.float32).repeat_interleave(PER_CASE, 0)
    fell = torch.zeros(n, dtype=torch.bool, device=env.device)
    vel, volt, vx = [[] for _ in CASES], [[] for _ in CASES], [[] for _ in CASES]
    no_load = VOLT / KT
    with torch.no_grad():
        for t in range(int((STAND_S + HOLD_S) / DT)):
            s = t * DT
            term.vel_command_b[:] = target if s >= STAND_S else 0.0
            obs, _, done, _, _ = env.step(policy(obs["actor"]))
            fell |= done.bool()
            if s < STAND_S + 1.5:
                continue
            qd, tau = robot.data.joint_vel[:, legs], joint_torque(env, robot)[:, legs]
            # Torque the motor can still produce in the direction it is pushing, at this speed and voltage.
            avail = KT * (VOLT - KT * qd * torch.sign(tau)) / RES
            for i in range(len(CASES)):
                k = torch.arange(i * PER_CASE, (i + 1) * PER_CASE, device=env.device)
                k = k[~fell[k]]
                if len(k):
                    vel[i].append((qd[k].abs() / no_load).cpu())
                    volt[i].append((tau[k].abs() >= 0.9 * avail[k].clamp(min=1e-6)).float().cpu())
                    vx[i].append(robot.data.root_link_lin_vel_b[k, 0].cpu())
    env.close()
    rows = []
    for i, case in enumerate(CASES):
        v, u = torch.cat(vel[i]), torch.cat(volt[i])
        per_joint = u.mean(0) * 100
        j = per_joint.argmax().item()
        fr = fell[i * PER_CASE:(i + 1) * PER_CASE].float().mean().item() * 100
        rows.append((case[0], torch.quantile(v.flatten()[::7], .99).item() * 100, (v > .8).float().mean().item() * 100,
                     u.mean().item() * 100, names[legs[j]], per_joint[j].item(), fr, torch.cat(vx[i]).mean().item()))
    return rows


if __name__ == "__main__":
    print(f"RESULT no-load speed at {VOLT} V = {VOLT / KT:.1f} rad/s", flush=True)
    print("RESULT model case | p99 speed % no-load | time >80% speed | time at voltage limit % | worst joint, % | fell % | vx", flush=True)
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        for r in run(path):
            print(f"RESULT {label:9s} {r[0]:16s} {r[1]:6.1f} {r[2]:6.1f} {r[3]:6.1f}  {r[4]} {r[5]:.1f}  fell {r[6]:.0f}  vx {r[7]:.2f}", flush=True)

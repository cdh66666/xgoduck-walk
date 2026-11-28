import os
"""0.5 m/s acceptance: step to a command, hold, brake to zero in one step, stand.
Runs with the measured heavier head, charger and zero errors. Usage: accept_speed.py name=model.onnx ..."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from duck_real import BASE_TASK, OnnxPolicy, apply_real

PER_CASE = 48
CASES = [
    ("forward 0.30", .30, 0, 0), ("forward 0.50", .50, 0, 0), ("back -0.30", -.30, 0, 0), ("back -0.50", -.50, 0, 0),
    ("fwd 0.50 turn L", .50, 0, 1.5), ("fwd 0.50 turn R", .50, 0, -1.5), ("turn L 1.5", 0, 0, 1.5), ("turn R 1.5", 0, 0, -1.5),
    ("forward 0.70", .70, 0, 0), ("back -0.70", -.70, 0, 0),
    ("forward 1.00", 1.0, 0, 0), ("back -1.00", -1.0, 0, 0), ("turn L 3.0", 0, 0, 3.0), ("fwd 1.00 turn L", 1.0, 0, 1.5),
    ("forward 1.20", 1.2, 0, 0), ("back -1.20", -1.2, 0, 0),
]
STAND_S, HOLD_S, AFTER_S, DT = 1.0, 4.0, 3.0, 0.02


def run(path):
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    n = PER_CASE * len(CASES)
    cfg = apply_real(load_env_cfg(BASE_TASK, play=True), vin=tuple(float(x) for x in os.environ.get("ACC_VIN", "7.3,7.7").split(",")),
                     vin_min=6.9, drop=tuple(float(x) for x in os.environ.get("ACC_DROP", ".05,.2").split(",")),
                     max_lag=int(os.environ.get("ACC_LAG", "8")), train=True)
    if os.environ.get("SIM_ITERS"):   # 11-27 speed: check solver-iteration change does not change results
        cfg.sim.mujoco.iterations = int(os.environ["SIM_ITERS"]); cfg.sim.mujoco.ls_iterations = int(os.environ.get("SIM_LS", 10))
    if os.environ.get("ACC_FRIC"):   # 11-28: falls vs floor friction (training randomizes 0.2-0.8)
        cfg.events["foot_friction"].params["ranges"] = tuple(float(x) for x in os.environ["ACC_FRIC"].split(","))
    if os.environ.get("ACC_CARPET"):   # 11-28 user: duck walks on carpet -> soft foot contact (head_carpet_sweep.py model)
        rc = cfg.scene.entities["robot"]; orig = rc.spec_fn
        def spec_fn():
            spec = orig()
            for g in spec.geoms:
                if "foot" in g.name.lower():
                    g.priority, g.solref, g.solimp = 5, (0.05, 1.0), (0.6, 0.9, 0.006, 0.5, 2.0)
            return spec
        rc.spec_fn = spec_fn
    cfg.events.pop("sudden_brake", None)  # this test applies the brake itself, at a known time
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
    target = torch.tensor([x[1:] for x in CASES], device=env.device, dtype=torch.float32).repeat_interleave(PER_CASE, 0)
    moving, braking = torch.zeros(n, dtype=torch.bool, device=env.device), torch.zeros(n, dtype=torch.bool, device=env.device)
    vx, wz = torch.zeros(n, device=env.device), torch.zeros(n, device=env.device)
    count = 0
    jn = list(robot.joint_names); hip = jn.index(next(x for x in jn if "left" in x and "hip_pitch" in x))
    hist = []
    with torch.no_grad():
        for t in range(int((STAND_S + HOLD_S + AFTER_S) / DT)):
            s = t * DT
            on = STAND_S <= s < STAND_S + HOLD_S
            term.vel_command_b[:] = target if on else 0.0
            obs, _, done, _, _ = env.step(policy(obs["actor"]))
            d = done.bool() & ~moving & ~braking
            if s < STAND_S + HOLD_S:
                moving |= d
            else:
                braking |= d
            if STAND_S + 1.5 <= s < STAND_S + HOLD_S:
                vx += robot.data.root_link_lin_vel_b[:, 0]
                wz += robot.data.root_link_ang_vel_b[:, 2]
                hist.append(robot.data.joint_pos[:, hip].clone())
                count += 1
    env.close()
    H = torch.stack(hist)                       # T x n
    H = H - H.mean(0, keepdim=True)
    sgn = torch.sign(H); cross = ((sgn[1:] * sgn[:-1]) < 0).sum(0).float()
    span = len(hist) * DT
    freq = cross / 2 / span                     # left-leg gait cycles per second
    rows = []
    for i, case in enumerate(CASES):
        k = slice(i * PER_CASE, (i + 1) * PER_CASE)
        ok = ~moving[k] & ~braking[k]
        alive = ~moving[k]
        rows.append((case[0], moving[k].float().mean().item() * 100,
                     braking[k].sum().item() / max(alive.sum().item(), 1) * 100,
                     vx[k][ok].mean().item() / count if ok.any() else float("nan"),
                     wz[k][ok].mean().item() / count if ok.any() else float("nan"),
                     freq[k][ok].mean().item() if ok.any() else float("nan")))
    return rows


if __name__ == "__main__":
    print("RESULT model case | fell moving % | fell after brake % | actual vx | actual wz | gait Hz | stride m (vx/Hz)", flush=True)
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        for r in run(path):
            st = abs(r[3]) / r[5] if r[5] and r[5] == r[5] and r[5] > 0 else float("nan")
            print(f"RESULT {label:9s} {r[0]:16s} {r[1]:6.1f} {r[2]:6.1f} {r[3]:7.3f} {r[4]:7.3f} {r[5]:6.2f} {st:6.3f}", flush=True)

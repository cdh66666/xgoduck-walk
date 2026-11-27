"""From-scratch run (11-26, user: "not working in the old frame -> train a new model").
Fine-tuning fast2/gait capped at ~0.5 m/s (gait0.85 no gain). New policy from random init with:
  - wide linear tracking kernel (std 0.5, weight 6) so a 1.0 command still gives gradient (fast3_1ms stood still at std 0.22)
  - command range +-1.0 from the start, plus turn-in-place and standing envs
  - long-stride posture (hip_pitch std x1.8, air_time 0.15-0.5) from gait0.7, which was the only step that lifted the cap
  - lower foot lift target (2.0 -> 1.2 cm) and a penalty on leg joint speed above 80% of servo no-load speed (Jean)
  - curricula kept as in duck_real (start at step 0), since this is not a resume."""
import json, os, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from duck_real import BASE_TASK, apply_real

TASK = "Mjlab-Velocity-Flat-XgoDuck-Scratch"
VOLT_MIN = 7.0
_P = json.loads(Path("src/mjlab_microduck/robot/xgoduck/params/1910_m6.json").read_text())
V_LIM = 0.8 * VOLT_MIN / _P["kt"]
if os.environ.get("SCR_STAGE") in ("F", "G", "H"):   # 11-27 user: push 1 m/s -> relax servo margin to 90% no-load at 7.4 V (2S nominal)
    V_LIM = 0.9 * 7.4 / _P["kt"]


def leg_overspeed(env):
    import torch
    robot = env.scene["robot"]
    ids = [i for i, n in enumerate(robot.joint_names) if any(k in n for k in ("hip", "knee", "ankle"))]
    qd = robot.data.joint_vel[:, ids].abs()
    return torch.sum(torch.square(torch.clamp(qd - V_LIM, min=0.0)), dim=1)


STAGE = os.environ.get("SCR_STAGE", "A")


def lin_progress(env):
    """Jean 11-26: linear progress = projection of body xy velocity on the command / |command|^2, clipped 0..1.
    Standing still scores 0, so a wide tracking kernel can no longer pay for standing."""
    import torch
    cmd = env.command_manager.get_command("twist")[:, :2]
    v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    n2 = torch.sum(cmd * cmd, dim=1)
    prog = torch.clamp(torch.sum(v * cmd, dim=1) / n2.clamp(min=1e-4), 0.0, 1.0)
    return torch.where(n2 > 0.01, prog, torch.zeros_like(prog))


def yaw_progress(env):
    """11-26 stage C: turning progress = body yaw rate projected on the yaw command / |command|, clipped 0..1.
    Stage B never learned to turn: with +-2.5 rad/s commands and std 0.9 the yaw kernel gave ~0 reward at 0 turn."""
    import torch
    cmd = env.command_manager.get_command("twist")[:, 2]
    w = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    prog = torch.clamp(w * torch.sign(cmd) / cmd.abs().clamp(min=1e-3), 0.0, 1.0)
    return torch.where(cmd.abs() > 0.2, prog, torch.zeros_like(prog))


def cadence(env, sensor_name="feet_ground_contact", command_name="twist"):
    """11-27 stage G (user: force higher cadence). F kept 2.3 steps/s and 0.25 m stride -> ~0.6 m/s cap.
    At each touchdown, reward the finished swing time for being near a speed-dependent target:
    T(v) = 0.30 - 0.12|vx| (0.5 m/s -> 0.24 s, 1.0 m/s -> 0.18 s), Gaussian width 0.04 s."""
    import torch
    sensor = env.scene[sensor_name]
    first = sensor.compute_first_contact(env.step_dt).float()
    last = sensor.data.last_air_time
    cmd = env.command_manager.get_command(command_name)
    vx = cmd[:, 0].abs()
    tgt = torch.clamp(0.30 - 0.12 * vx, 0.16, 0.30).unsqueeze(1)
    r = torch.sum(first * torch.exp(-torch.square((last - tgt) / 0.04)), dim=1)
    steps = torch.sum(first, dim=1)
    env.extras["log"]["Metrics/touchdowns_per_step"] = steps.mean()
    return torch.where(torch.norm(cmd[:, :2], dim=1) > 0.2, r, torch.zeros_like(r))


def apply_scratch(cfg):
    """Stage A (learn to walk): duck_real rewards that learned walking from scratch (tight std 0.22),
    command +-0.6, plus overspeed penalty and slightly longer hip stride. First try (09-54-46) with std 0.5,
    +-1.0, low foot lift stood still for 600 iters: standing earned ~37% of the speed reward.
    Stage B (resume from A): +-1.0, std 0.35, swing 0.15-0.5 s, hip stride x1.8 (gait0.7 recipe)."""
    from mjlab.managers import RewardTermCfg
    cfg = apply_real(cfg)
    r = cfg.rewards
    r["leg_overspeed"] = RewardTermCfg(func=leg_overspeed, weight=-0.5)
    hip = 1.3 if STAGE == "A" else (2.3 if STAGE in ("F", "G", "H") else 1.8)
    seen = set()
    for key in ("std_walking", "std_running"):
        st = r["pose"].params.get(key)
        if st is not None and id(st) not in seen:
            seen.add(id(st))
            for j in list(st):
                if "hip_pitch" in j: st[j] *= hip
    c = cfg.commands["twist"]
    r["lin_progress"] = RewardTermCfg(func=lin_progress, weight=1.0 if STAGE == "A" else 3.0)
    if STAGE in ("B", "C", "D", "E", "F", "G", "H"):
        r["track_linear_velocity"].weight = 6.0
        r["track_linear_velocity"].params["std"] = 0.35
        r["track_angular_velocity"].weight = 5.0
        r["track_angular_velocity"].params["std"] = 0.9
        r["air_time"].params["threshold_min"] = 0.15
        r["air_time"].params["threshold_max"] = 0.50
        for term in (cfg.curriculum.values() if STAGE != "H" else []):   # resumed run restarts the step counter (H = from scratch: keep curricula)
            for name in ("weight_stages", "range_stages"):
                stages = (term.params or {}).get(name)
                if stages:
                    last = dict(stages[-1]); last["step"] = 0; term.params[name] = [last]
    if STAGE == "C":
        r["yaw_progress"] = RewardTermCfg(func=yaw_progress, weight=4.0)
        r["track_angular_velocity"].params["std"] = 1.2
    if STAGE == "E":   # 11-26: D (w1, std .5) over-corrected: in-place turn collapsed to ~0.1. Middle ground, resume from C.
        r["yaw_progress"] = RewardTermCfg(func=yaw_progress, weight=2.5)
        r["track_angular_velocity"].weight = 6.0
        r["track_angular_velocity"].params["std"] = 0.8
    if STAGE == "F":   # 11-27 user picked "push 1 m/s": E saturates ~0.55. Keep E turning terms, longer stride (hip x2.3),
        # relaxed overspeed penalty, stronger forward progress, longer swing allowed, forward command up to 1.2.
        r["yaw_progress"] = RewardTermCfg(func=yaw_progress, weight=2.5)
        r["track_angular_velocity"].weight = 6.0
        r["track_angular_velocity"].params["std"] = 0.8
        r["leg_overspeed"].weight = -0.2
        r["lin_progress"].weight = 4.0
        r["track_linear_velocity"].params["std"] = 0.4
        r["air_time"].params["threshold_max"] = 0.6
    if STAGE == "H":   # 11-27 G (resume E) only moved cadence 2.2 -> 2.45 Hz, 0.54 -> 0.60 m/s; user: else train from scratch.
        # From scratch with G terms, stronger cadence push and a shorter swing window, curricula kept.
        r["yaw_progress"] = RewardTermCfg(func=yaw_progress, weight=2.5)
        r["track_angular_velocity"].weight = 6.0
        r["track_angular_velocity"].params["std"] = 0.8
        r["leg_overspeed"].weight = -0.2
        r["lin_progress"].weight = 4.0
        r["track_linear_velocity"].params["std"] = 0.4
        r["air_time"].params["threshold_min"] = 0.08
        r["air_time"].params["threshold_max"] = 0.22
        r["cadence"] = RewardTermCfg(func=cadence, weight=15.0)
    if STAGE == "G":   # 11-27 user picked "change gait": F's air_time window (to 0.6 s) paid for slow long swings.
        # Resume from E; short swing window + touchdown cadence reward + F's stride/progress/overspeed relaxations.
        r["yaw_progress"] = RewardTermCfg(func=yaw_progress, weight=2.5)
        r["track_angular_velocity"].weight = 6.0
        r["track_angular_velocity"].params["std"] = 0.8
        r["leg_overspeed"].weight = -0.2
        r["lin_progress"].weight = 4.0
        r["track_linear_velocity"].params["std"] = 0.4
        r["air_time"].params["threshold_min"] = 0.10
        r["air_time"].params["threshold_max"] = 0.28
        r["cadence"] = RewardTermCfg(func=cadence, weight=8.0)
    if STAGE == "D":   # 11-26: C learned to turn but overshoots ~50% (1.5 -> 2.2) and drifts when walking straight
        r["yaw_progress"] = RewardTermCfg(func=yaw_progress, weight=1.0)
        r["track_angular_velocity"].weight = 7.0
        r["track_angular_velocity"].params["std"] = 0.5
    if c.ranges.lin_vel_x[1] > 0.1:   # training only
        v = 0.6 if STAGE == "A" else 1.0
        c.ranges.lin_vel_x = (-v, v)
        if STAGE == "B":
            c.ranges.ang_vel_z = (-2.5, 2.5)
            c.rel_turn_in_place_envs = 0.1
        if STAGE in ("F", "G", "H"):
            c.ranges.lin_vel_x = (-1.0, 1.2)
        if STAGE in ("C", "D", "E", "F", "G", "H"):
            c.ranges.ang_vel_z = (-1.5, 1.5)
            c.rel_turn_in_place_envs = 0.25
    return cfg


def main(argv):
    import mjlab, mjlab.tasks  # noqa: F401
    import tyro
    from mjlab.scripts.train import TrainConfig, launch_training
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls, register_mjlab_task
    rl = replace(load_rl_cfg(BASE_TASK), experiment_name="xgoduck_real", run_name="scratch")
    register_mjlab_task(TASK, apply_scratch(load_env_cfg(BASE_TASK)),
                        apply_scratch(load_env_cfg(BASE_TASK, play=True)), rl, load_runner_cls(BASE_TASK))
    args = tyro.cli(TrainConfig, args=argv, default=TrainConfig.from_task(TASK), config=mjlab.TYRO_FLAGS)
    launch_training(task_id=TASK, args=args)


if __name__ == "__main__":
    main(sys.argv[1:])

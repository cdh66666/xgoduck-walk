"""XGO Duck training/evaluation under conditions measured on the real robot.

Differences from the official Velocity task, all taken from the 2026-11-22 walk logs and runtime 0.3.5:
  - action EMA as on the board (legs 0.35, head 0.45)
  - about 5% of 50 Hz control ticks are missed
  - battery sag down to about 6.7 V
  - command range centred on what the duck is actually driven with

Usage (from ~/duck/xgoduck_rl, with its .venv python):
  duck_real.py train --env.scene.num-envs 2048 --agent.max-iterations 2500
  duck_real.py eval name=model.onnx [name2=model2.onnx ...]
"""
import json, math, sys
from dataclasses import dataclass, replace

import numpy as np
import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction

LEG_ALPHA, HEAD_ALPHA = 0.35, 0.45
ALPHA_JITTER = 0.05
MISSED_TICK_PROB = 0.05
TRAIN_TASK = "Mjlab-Velocity-Flat-XgoDuck-Real"
BASE_TASK = "Mjlab-Velocity-Flat-XgoDuck"
ZERO_ERROR_RAD = math.radians(4.0)
HEAD_BODY = "yaw_roll_motion"
HEAD_PAYLOAD_KG, HEAD_FORWARD_M = 0.069, 0.040  # measured 890 g total minus model minus charger
HEAD_FORWARD_AXIS = np.array([0.0, 1.0, 0.0])  # trunk-forward in the head frame at the home pose
HEAD_MASS_SCALE = (0.80, 1.30)
CHARGER_KG = 0.030
CHARGER_OFFSET_M = np.array([-0.013, 0.0, -0.040])  # estimated from photos, relative to trunk CoM
BRAKE_INTERVAL_S = (1.5, 4.0)
BRAKE_PROB = 0.0  # off: every run trained with this event learned to stand still (accept_controls.log)
BRAKE_HOLD_S = (0.8, 1.6)  # stand this long after braking, then get a new moving command


def sudden_brake(env, env_ids, command_name="twist", probability=BRAKE_PROB):
    """Drop the velocity command to zero in one step, like releasing the stick at speed."""
    term = env.command_manager.get_term(command_name)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    picked = ids[torch.rand(len(ids), device=env.device) < probability]
    term.vel_command_b[picked] = 0.0
    # Without this a braked env stays at zero until its 3-8 s resample, so training becomes mostly standing.
    term.time_left[picked] = torch.empty(len(picked), device=env.device).uniform_(*BRAKE_HOLD_S)


@dataclass(kw_only=True)
class RealDeployActionCfg(JointPositionActionCfg):
    alpha_jitter: float = ALPHA_JITTER
    missed_tick_prob: float = MISSED_TICK_PROB

    def build(self, env):
        return RealDeployAction(self, env)


class RealDeployAction(JointPositionAction):
    """Matches policy.py: filtered = alpha * filtered + (1 - alpha) * action."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        base = [LEG_ALPHA if any(k in n for k in ("hip", "knee", "ankle")) else HEAD_ALPHA for n in self.target_names]
        self._base_alpha = torch.tensor(base, device=self.device).repeat(self.num_envs, 1)
        self._alpha = self._base_alpha.clone()
        self._filtered = torch.zeros_like(self._raw_actions)
        self.reset(None)

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        update = torch.rand(self.num_envs, device=self.device) >= self.cfg.missed_tick_prob
        mixed = self._alpha * self._filtered + (1 - self._alpha) * actions
        self._filtered[update] = mixed[update]
        self._processed_actions = self._filtered * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(self._processed_actions, self._clip[:, :, 0], self._clip[:, :, 1])

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._filtered[ids] = 0.0
        noise = (torch.rand_like(self._alpha) * 2 - 1) * self.cfg.alpha_jitter
        self._alpha[ids] = (self._base_alpha + noise)[ids]


def apply_real(cfg, *, vin=(6.9, 8.1), vin_min=6.6, drop=(0.0, 0.3), max_lag=8, train=True):
    old = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = RealDeployActionCfg(
        entity_name=old.entity_name, actuator_names=old.actuator_names, scale=old.scale, offset=old.offset,
        clip=old.clip, preserve_order=old.preserve_order, use_default_offset=old.use_default_offset,
        alpha_jitter=ALPHA_JITTER if train else 0.0,
    )
    art = cfg.scene.entities["robot"].articulation
    art.actuators = tuple(
        replace(a, vin_range=vin, vin_min=vin_min, vin_drop_gain_range=drop, delay_min_lag=1, delay_max_lag=max_lag)
        for a in art.actuators
    )
    if train:
        r = cfg.commands["twist"].ranges
        r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (-0.55, 0.55), (-0.25, 0.25), (-1.5, 1.5)
        add_robustness(cfg)
    return cfg


def add_robustness(cfg):
    """Hand calibration error (official: +-0.86 deg) and the heavier, forward AV head.
    The head payload is an estimate until the real duck is weighed, so its mass is randomized."""
    from mjlab.envs.mdp import dr
    from mjlab.managers import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    if "encoder_bias" in cfg.events:
        cfg.events["encoder_bias"].params["bias_range"] = (-ZERO_ERROR_RAD, ZERO_ERROR_RAD)
    robot = cfg.scene.entities["robot"]
    original = robot.spec_fn

    def spec_with_head():
        spec = original()
        b = spec.body(HEAD_BODY)
        m0, m1 = b.mass, HEAD_PAYLOAD_KG
        b.ipos = np.array(b.ipos) + HEAD_FORWARD_AXIS * HEAD_FORWARD_M * m1 / (m0 + m1)
        k = (m0 + m1) / m0
        b.mass = m0 + m1
        if not np.isnan(b.fullinertia[0]):
            b.fullinertia = b.fullinertia * k
        else:
            b.inertia = b.inertia * k
        t = spec.body("trunk_base")
        kt = (t.mass + CHARGER_KG) / t.mass
        t.ipos = np.array(t.ipos) + CHARGER_OFFSET_M * CHARGER_KG / (t.mass + CHARGER_KG)
        t.mass += CHARGER_KG
        if not np.isnan(t.fullinertia[0]):
            t.fullinertia = t.fullinertia * kt
        else:
            t.inertia = t.inertia * kt
        return spec

    robot.spec_fn = spec_with_head
    if BRAKE_PROB > 0:
        cfg.events["sudden_brake"] = EventTermCfg(
            func=sudden_brake, mode="interval", interval_range_s=BRAKE_INTERVAL_S, params={"command_name": "twist"},
        )
    cfg.events["randomize_head_mass"] = EventTermCfg(
        func=dr.pseudo_inertia, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=(HEAD_BODY,)),
                "alpha_range": (math.log(HEAD_MASS_SCALE[0]) / 2, math.log(HEAD_MASS_SCALE[1]) / 2)},
    )


def register():
    import mjlab.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls, register_mjlab_task
    rl = replace(load_rl_cfg(BASE_TASK), experiment_name="xgoduck_real", run_name="real")
    register_mjlab_task(TRAIN_TASK, apply_real(load_env_cfg(BASE_TASK)),
                        apply_real(load_env_cfg(BASE_TASK, play=True)), rl, load_runner_cls(BASE_TASK))


def train(argv):
    register()
    from mjlab.scripts.train import TrainConfig, launch_training
    import mjlab, tyro
    args = tyro.cli(TrainConfig, args=argv, default=TrainConfig.from_task(TRAIN_TASK), config=mjlab.TYRO_FLAGS)
    launch_training(task_id=TRAIN_TASK, args=args)


class OnnxPolicy:
    def __init__(self, path):
        import onnxruntime as ort
        self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.name = self.s.get_inputs()[0].name
        self.batch = self.s.get_inputs()[0].shape[0]

    def __call__(self, obs):
        x = obs.detach().cpu().numpy().astype("float32")
        if isinstance(self.batch, int):
            import numpy as np
            out = np.concatenate([self.s.run(None, {self.name: x[i:i + 1]})[0] for i in range(len(x))])
        else:
            out = self.s.run(None, {self.name: x})[0]
        return torch.as_tensor(out, device=obs.device)


def evaluate(models, envs=128, seconds=12.0):
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    cases = [("stand", 0, 0, 0), ("fwd.15", .15, 0, 0), ("fwd.20", .20, 0, 0), ("fwd.30", .30, 0, 0),
             ("back.15", -.15, 0, 0), ("side.15", 0, .15, 0), ("turn.7", 0, 0, .7)]
    results = {}
    for label, path in models.items():
        policy, results[label] = OnnxPolicy(path), {}
        for case, vx, vy, wz in cases:
            cfg = apply_real(load_env_cfg(BASE_TASK, play=True), vin=(7.0, 7.6), vin_min=6.7, drop=(.1, .25), train=False)
            cfg.scene.num_envs = envs
            cfg.episode_length_s = seconds + 5
            cfg.seed = 7
            c = cfg.commands["twist"]
            c.ranges.lin_vel_x, c.ranges.lin_vel_y, c.ranges.ang_vel_z = (vx, vx), (vy, vy), (wz, wz)
            for a in ("rel_standing_envs", "rel_forward_envs", "rel_turn_in_place_envs", "rel_backward_envs",
                      "rel_heading_envs", "independent_command_fraction"):
                if hasattr(c, a):
                    setattr(c, a, 0.0)
            env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
            obs, _ = env.reset()
            robot = env.scene["robot"]
            fell = torch.zeros(envs, dtype=torch.bool, device=env.device)
            speed, yaw, tilt = [], [], []
            steps = int(seconds * 50)
            with torch.no_grad():
                for t in range(steps):
                    obs, _, term, _, _ = env.step(policy(obs["actor"]))
                    fell |= term.bool()
                    if t > 100:
                        alive = ~fell
                        if alive.any():
                            speed.append(robot.data.root_link_lin_vel_b[alive, :2].mean(0).cpu())
                            yaw.append(robot.data.root_link_ang_vel_b[alive, 2].mean().item())
                            g = robot.data.projected_gravity_b[alive]
                            tilt.append(torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1))).mean().item())
            v = torch.stack(speed).mean(0) if speed else torch.zeros(2)
            results[label][case] = {
                "fall_pct": round(fell.float().mean().item() * 100, 1), "vx": round(v[0].item(), 3),
                "vy": round(v[1].item(), 3), "yaw": round(sum(yaw) / max(1, len(yaw)), 2),
                "tilt": round(sum(tilt) / max(1, len(tilt)), 2),
            }
            env.close()
            print(label, case, results[label][case], flush=True)
    print("RESULT_JSON " + json.dumps(results))


if __name__ == "__main__":
    command, rest = sys.argv[1], sys.argv[2:]
    if command == "train":
        train(rest)
    elif command == "eval":
        evaluate(dict(a.split("=", 1) for a in rest))
    else:
        sys.exit(__doc__)

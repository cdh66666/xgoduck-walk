"""Sim only: render a small 2x2 GIF and a key-frame strip of one model. Usage: render_motion.py model.onnx out_dir"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from duck_real import BASE_TASK, OnnxPolicy, apply_real

CASES = [("Forward 0.5", .5, 0, 0), ("Back 0.5", -.5, 0, 0), ("Walk+turn", .5, 0, 1.5), ("Turn in place", 0, 0, 1.5)]
STAND_S, HOLD_S, AFTER_S, DT = 0.5, 4.0, 1.5, 0.02
W, H, EVERY = 220, 180, 3  # 50 Hz / 3 = about 17 fps


def simulate(path):
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    cfg = apply_real(load_env_cfg(BASE_TASK, play=True), vin=(7.3, 7.7), vin_min=6.9, drop=(.05, .2), train=True)
    cfg.events.pop("sudden_brake", None)
    cfg.scene.num_envs, cfg.episode_length_s, cfg.seed = len(CASES), 30, 7
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
    term = env.command_manager.get_term("twist")
    target = torch.tensor([x[1:] for x in CASES], device=env.device, dtype=torch.float32)
    origins = env.scene.env_origins.cpu().numpy()
    frames, falls = [], np.zeros(len(CASES), dtype=bool)
    with torch.no_grad():
        for t in range(int((STAND_S + HOLD_S + AFTER_S) / DT)):
            s = t * DT
            term.vel_command_b[:] = target if STAND_S <= s < STAND_S + HOLD_S else 0.0
            obs, _, done, _, _ = env.step(policy(obs["actor"]))
            falls |= done.cpu().numpy().astype(bool)
            if t % EVERY == 0:
                q = env.sim.data.qpos.cpu().numpy().copy()
                q[:, :2] -= origins[:, :2]
                frames.append(q)
    model = env.sim.mj_model
    return model, np.stack(frames), falls


def render(model, frames):
    import mujoco
    model.vis.global_.offwidth, model.vis.global_.offheight = max(W, 640), max(H, 480)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.75, 135, -18
    out = np.zeros((len(frames), len(CASES), H, W, 3), dtype=np.uint8)
    for f, q in enumerate(frames):
        for i in range(len(CASES)):
            data.qpos[:] = q[i]
            mujoco.mj_forward(model, data)
            cam.lookat[:] = [q[i, 0], q[i, 1], 0.12]
            renderer.update_scene(data, cam)
            out[f, i] = renderer.render()
    return out


def main(path, out_dir):
    from PIL import Image, ImageDraw
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, frames, falls = simulate(path)
    video = render(model, frames)

    def label(img, i):
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, W, 14), fill=(255, 255, 255))
        d.text((3, 2), CASES[i][0] + ("  FELL" if falls[i] else ""), fill=(0, 0, 0))
        return img

    gif = []
    for f in range(len(video)):
        sheet = Image.new("RGB", (W * 2, H * 2))
        for i in range(len(CASES)):
            sheet.paste(label(Image.fromarray(video[f, i]), i), ((i % 2) * W, (i // 2) * H))
        gif.append(sheet.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    gif[0].save(out_dir / "duck_motion.gif", save_all=True, append_images=gif[1:], duration=EVERY * 20, loop=0, optimize=True)

    picks = np.linspace(int(1.5 / (DT * EVERY)), int(3.0 / (DT * EVERY)), 6).astype(int)
    strip = Image.new("RGB", (W * len(picks), H * len(CASES)), "white")
    for i in range(len(CASES)):
        for j, f in enumerate(picks):
            strip.paste(label(Image.fromarray(video[f, i]), i), (j * W, i * H))
    strip.save(out_dir / "duck_sequence.jpg", quality=80)
    print("RESULT falls", dict(zip([c[0] for c in CASES], falls.tolist())), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

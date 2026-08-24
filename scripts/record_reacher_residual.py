"""Videos of the DAgger clone and its residual, against the expert and vanilla.

Early stopping is OFF, so every clip runs the full 50 steps. That is the point:
journey 12 found every controller arrives at roughly half its final error and then
backs off, and a clip that stops at first contact cannot show it. The readout
carries BOTH the live distance and the best so far, so "arrive and leave" is
visible as the two numbers separating.

Scenarios are chosen by what they demonstrate rather than by index -- a case the
residual rescues from the clone, one where all four succeed, and the widest
best->final gap -- because a random scenario mostly shows four arms doing the same
thing.

    uv run python scripts/record_reacher_residual.py
"""
from __future__ import annotations

import argparse
import os

import gymnasium as gym
import numpy as np

import reacher  # noqa: F401  registers the Gym ID
from core.video_encoding import encode_video
from reacher.clone_data import build_bank, build_select_controller
from reacher.eval import ClonePolicy, ControllerPolicy, run_episode
from reacher.model import load_model
from rl.clone import load_clone
from rl.sb3 import load_policy


def _font(size):
    """DejaVu Sans Mono, bundled with matplotlib so no system font is assumed.

    An earlier version of this HUD hand-rolled a 5x7 bitmap font that only
    covered DIGITS, and rendered every letter as a grey block -- so the readout
    showed numbers with unlabelled smudges beside them. Use a real font.
    """
    from pathlib import Path

    import matplotlib
    from PIL import ImageFont

    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono-Bold.ttf"
    try:
        return ImageFont.truetype(str(ttf), size)
    except OSError:
        return ImageFont.load_default()


def annotate(frame, label, step, dist, best, tol, need):
    """Burn a labelled readout into the frame.

    Two lines, both left-aligned and laid out by measured text width so nothing
    overlaps or clips -- an earlier version right-aligned a second column at a
    fixed offset and the longer controller names ran straight through it.

    `now` is the live tip-to-goal distance, `best` the closest approach so far.
    A controller that arrives and leaves shows those two separating, which is why
    these clips run the full horizon instead of stopping at first contact.
    """
    from PIL import Image, ImageDraw

    h, w = frame.shape[:2]
    # h264 + yuv420p requires EVEN dimensions in both axes; an odd total height
    # makes ffmpeg close the pipe mid-write with a bare "broken pipe". Round the
    # banner up so `h + bar` stays even.
    bar = int(0.155 * h)
    bar += (h + bar) % 2
    img = Image.new("RGB", (w, h + bar), (250, 250, 248))
    img.paste(Image.fromarray(frame), (0, bar))
    d = ImageDraw.Draw(img)

    # 58 chars of mono at advance ~0.6*size must fit w - 2*pad, which caps
    # the small font at ~0.026*h on a 720 px frame.
    big, small = _font(int(0.042 * h)), _font(int(0.026 * h))
    hit = dist < tol
    accent = (12, 163, 12) if hit else ((235, 104, 52) if best < tol else (150, 150, 148))

    pad = int(0.022 * w)
    d.text((pad, int(0.015 * h)), label, font=big, fill=(20, 20, 20))

    # Second line as measured segments, so `now` can be coloured without a
    # fixed-offset column that long labels would collide with.
    segs = [(f"step {step:02d}/50", (110, 110, 108)),
            ("   start ", (150, 150, 148)), (f"{need * 1e3:.0f}mm", (60, 60, 58)),
            ("   now ", (150, 150, 148)), (f"{dist * 1e3:.1f}mm", accent),
            ("   best ", (150, 150, 148)), (f"{best * 1e3:.1f}mm", (60, 60, 58)),
            ("   tol ", (150, 150, 148)), (f"{tol * 1e3:.0f}mm", (150, 150, 148))]
    x, y = pad, int(0.082 * h)
    for text, colour in segs:
        d.text((x, y), text, font=small, fill=colour)
        x += int(d.textlength(text, font=small))
    d.rectangle([0, bar - 6, w, bar - 1], fill=accent)
    return np.asarray(img)


def record(env, policy, qpos, goal, label, tol, residual=False, model=None):
    """One episode, returning (frames, summary). Mirrors `run_episode`'s metrics."""
    obs, info = env.reset(seed=0, options={"qpos": qpos, "goal": goal})
    base = env.unwrapped
    need = float(info["dist"])
    best = need
    frames = [annotate(env.render(), label, 0, need, best, tol, need)]
    for t in range(base.max_steps):
        if residual:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = policy(env, info)
        obs, _r, _term, trunc, info = env.step(action)
        best = min(best, float(info["dist"]))
        frames.append(annotate(env.render(), label, t + 1,
                               float(info["dist"]), best, tol, need))
        if trunc:
            break
    return frames, {"need": need, "best": best, "final": float(info["dist"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", default="data/reacher_scenarios_v1.npz")
    p.add_argument("--clone", default="data/dagger_clone_r3.pt")
    p.add_argument("--residual", default="data/reacher_residual_dagger_200k.zip")
    p.add_argument("--vanilla", default="data/reacher_vanilla_200k.zip")
    p.add_argument("--scan", type=int, default=40)
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--size", type=int, default=720)
    p.add_argument("--out-dir", default="videos/reacher_residual")
    args = p.parse_args()

    with np.load(args.scenarios) as z:
        eps = [(z["qpos"][i], z["goal"][i]) for i in range(args.scan)]

    mj, dat = load_model()
    bank, _ = build_bank(mj, dat, np.random.default_rng(0))
    predictor = load_clone(args.clone, device="cpu")
    res_model = load_policy(args.residual, algo="sac", device="cpu")
    van_model = load_policy(args.vanilla, algo="sac", device="cpu")

    # --- scan headless to pick scenarios worth watching ---------------------
    from reacher.residual_env import ResidualSelectEnv
    scan = gym.make("ReacherGoal-v0")
    clone_r = [run_episode(scan, ClonePolicy(predictor), q, g) for q, g in eps]
    scan.close()
    rese = ResidualSelectEnv(clone_path=args.clone)
    res_r = []
    for q, g in eps:
        o, i = rese.reset(seed=0, options={"qpos": q, "goal": g})
        need = float(i["dist"])
        best = need
        for _ in range(rese.base.max_steps):
            a, _ = res_model.predict(o, deterministic=True)
            o, _rw, _t, tr, i = rese.step(a)
            best = min(best, float(i["dist"]))
            if tr:
                break
        res_r.append({"best": best, "final": float(i["dist"]),
                      "reached": best < args.tol})
    rese.close()

    rescue = [i for i in range(len(eps))
              if res_r[i]["reached"] and not clone_r[i]["reached"]]
    both = [i for i in range(len(eps))
            if res_r[i]["reached"] and clone_r[i]["reached"]]
    drift = max(range(len(eps)),
                key=lambda i: clone_r[i]["final"] - clone_r[i]["best"])
    picks = []
    if rescue:
        picks.append((rescue[0], "rescue"))
    if both:
        picks.append((both[0], "both_succeed"))
    picks.append((drift, "widest_drift"))
    print(f"scanned {len(eps)}: {len(rescue)} residual rescues, {len(both)} both, "
          f"widest clone drift at #{drift}")

    os.makedirs(args.out_dir, exist_ok=True)
    env = gym.make("ReacherGoal-v0", render_mode="rgb_array", render_size=args.size)
    rese = ResidualSelectEnv(clone_path=args.clone)
    rese.env = gym.make("ReacherGoal-v0", render_mode="rgb_array",
                        render_size=args.size)
    rese.base = rese.env.unwrapped
    for idx, why in picks:
        q, g = eps[idx]
        rows = []
        for label, pol, kw in (
            ("Select-DPC  (expert)", ControllerPolicy(
                build_select_controller(bank, carry_prediction=False)), {}),
            ("DAgger clone", ClonePolicy(predictor), {}),
            ("DAgger clone + residual", None, {"residual": True, "model": res_model}),
            ("vanilla RL", None, {}),
        ):
            if label == "vanilla RL":
                def pol(e, _i, m=van_model):
                    a, _ = m.predict(e.unwrapped.build_obs(), deterministic=True)
                    return a
            target = rese if kw.get("residual") else env
            frames, summ = record(target, pol, q, g, label, args.tol, **kw)
            path = os.path.join(args.out_dir,
                                f"{why}_{label.lower().replace('  ', '_').replace(' ', '_')}.mp4".replace("(", "").replace(")", ""))
            encode_video(frames, path, args.fps)
            rows.append((label, summ, path))
        print(f"\nscenario #{idx} ({why}), need {rows[0][1]['need']*1e3:.0f} mm")
        for label, s, path in rows:
            print(f"  {label:<22} best {s['best']*1e3:5.1f}mm  "
                  f"final {s['final']*1e3:5.1f}mm  -> {path}")
    env.close()
    rese.close()


if __name__ == "__main__":
    main()

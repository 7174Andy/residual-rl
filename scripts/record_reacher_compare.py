"""Paired videos: fixed anchors vs Select-DPC (n_max=1) on the same scenarios.

Recorded with early stopping OFF, so every clip runs the full horizon. That is
the point: the 120-scenario evaluation found both controllers arrive at roughly
half their final error and then back off by the same factor (fixed 4.3 -> 8.9 mm,
Select 3.0 -> 6.6 mm). Clips that stop at first contact cannot show that, and
every earlier video in this project did stop.

The readout therefore carries BOTH the current distance and the best so far, so a
controller that arrives and leaves is visible as the two numbers separating.

Scenarios are scanned first and one example of each interesting case is recorded:

    rescue     fixed misses, Select-DPC reaches   -- where selection earns its keep
    drift      largest best-vs-final gap          -- the failure reach rate hides
    both       both reach                         -- the speed/precision difference
    neither    both miss                          -- what is left after selection

    uv run python scripts/record_reacher_compare.py --scan 30
"""
from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

from core.video_encoding import encode_video
from reacher.deepc_setup import (
    TIP_WEIGHT, anchor_grid, collect_anchor, make_controller, y_ref_for,
)
from reacher.model import (
    NQ_ARM, fingertip, frame_skip, model_path, sample_config, sample_goal,
    set_state, step_torque,
)
from reacher.selectdpc import SelectDPC, trajectory_bank

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record_reacher_video import _ring  # noqa: E402


def annotate(frame, label, step, dist, best, tol):
    """Burn label / step / current / best into the frame.

    `best` is what the reach metric scores; `dist` is where the tip actually is.
    Showing only one of them is how the drift stayed invisible for so long.
    """
    from PIL import Image, ImageDraw

    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 68], fill=(252, 252, 251))
    dr.text((12, 6), f"{label}    step {step:>2}", fill=(11, 11, 11))
    dr.text((12, 26), f"now   {dist * 1e3:6.1f} mm",
            fill=(20, 160, 20) if dist < tol else (200, 60, 60))
    dr.text((12, 46), f"best  {best * 1e3:6.1f} mm"
                      f"    (tolerance {tol * 1e3:.0f} mm)",
            fill=(20, 160, 20) if best < tol else (120, 120, 120))
    return np.asarray(im)


def episode(model, data, q0, goal, args, ctrl, renderer=None, cam=None, label=""):
    fs = frame_skip(model)
    set_state(model, data, q0, goal)
    t0 = fingertip(data)
    need = float(np.linalg.norm(goal - t0))
    ctrl.reset(np.concatenate([q0, t0]), u_initial=np.zeros(NQ_ARM))
    yref = y_ref_for(goal)
    best, dist, frames = need, need, []
    for t in range(args.steps):
        if renderer is not None:
            renderer.update_scene(data, camera=cam)
            _ring(renderer.scene, goal, args.tol, [0.2, 0.9, 0.2, 0.30])
            frames.append(annotate(renderer.render(), label, t, dist, best, args.tol))
        y = np.concatenate([np.asarray(data.qpos[:NQ_ARM]), fingertip(data)])
        try:
            u = ctrl.act(y, yref)
        except RuntimeError:
            break
        step_torque(model, data, u, fs)
        dist = float(np.linalg.norm(fingertip(data) - goal))
        best = min(best, dist)
    return {"need": need, "best": best, "final": dist,
            "reached": best < args.tol}, frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, nargs=2, default=[6, 5])
    p.add_argument("--T", type=int, default=1200)
    p.add_argument("--scan", type=int, default=30)
    p.add_argument("--n-cols", type=int, default=300)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--size", type=int, default=720)
    p.add_argument("--T-ini", type=int, default=5)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--lambda-g", type=float, default=5e-3)
    p.add_argument("--out-dir", default="videos/reacher_compare")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    xml = open(model_path()).read().replace(
        "<worldbody>",
        f"<visual><global offwidth='{args.size}' offheight='{args.size}'/></visual>"
        "\n<worldbody>", 1)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = 0.5
    model.vis.headlight.diffuse[:] = 0.7
    renderer = mujoco.Renderer(model, height=args.size, width=args.size)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.0]
    cam.distance, cam.elevation, cam.azimuth = 0.62, -90.0, 90.0

    rng = np.random.default_rng(args.seed)
    anchors = anchor_grid(model, *args.grid)
    print(f"collecting {len(anchors)} libraries ...")
    payload = {"anchors": anchors}
    for i, a in enumerate(anchors):
        rec = collect_anchor(model, data, a, args.T, rng)
        payload[f"u_{i}"], payload[f"q_{i}"], payload[f"tip_{i}"] = (
            rec["u"], rec["q"], rec["tip"])
    bank = trajectory_bank(payload, args.T_ini, args.N, stride=args.stride)
    fixed, _ = make_controller(payload, T_ini=args.T_ini, N=args.N,
                               lambda_g=args.lambda_g)
    select = SelectDPC(
        bank, anchor_headings=np.zeros(1),
        Q=np.diag([0.0] * NQ_ARM + [TIP_WEIGHT] * 2), R=1e-3 * np.eye(NQ_ARM),
        T_ini=args.T_ini, N=args.N, lambda_g=args.lambda_g, lambda_y=7.5e3,
        u_bounds=(-np.ones(NQ_ARM), np.ones(NQ_ARM)), solver="SCS",
        n_cols=args.n_cols, n_max=1)

    rng = np.random.default_rng(args.seed + 99)
    eps = [(sample_config(model, data, rng)[0], sample_goal(rng))
           for _ in range(args.scan)]
    print(f"scanning {len(eps)} scenarios (early stopping OFF) ...")
    scored = []
    for i, (q0, goal) in enumerate(eps):
        rf, _ = episode(model, data, q0, goal, args, fixed)
        rs, _ = episode(model, data, q0, goal, args, select)
        scored.append((i, rf, rs))
    print(f"  fixed  {sum(r[1]['reached'] for r in scored)}/{len(scored)}   "
          f"Select {sum(r[2]['reached'] for r in scored)}/{len(scored)}")

    picks = {}
    resc = [s for s in scored if s[2]["reached"] and not s[1]["reached"]]
    if resc:
        picks["rescue"] = max(resc, key=lambda s: s[1]["final"])[0]
    picks["drift"] = max(scored, key=lambda s: s[2]["final"] - s[2]["best"])[0]
    both = [s for s in scored if s[1]["reached"] and s[2]["reached"]]
    if both:
        picks["both"] = min(both, key=lambda s: s[2]["best"])[0]
    nei = [s for s in scored if not s[1]["reached"] and not s[2]["reached"]]
    if nei:
        picks["neither"] = max(nei, key=lambda s: s[2]["final"])[0]

    os.makedirs(args.out_dir, exist_ok=True)
    print()
    for name, i in picks.items():
        q0, goal = eps[i]
        for tag, ctrl, lab in (("fixed", fixed, "fixed anchors"),
                               ("select", select, "Select-DPC (n_max=1)")):
            r, frames = episode(model, data, q0, goal, args, ctrl, renderer, cam, lab)
            path = os.path.join(args.out_dir, f"{name}_{tag}.mp4")
            encode_video(frames, path, args.fps)
            print(f"  {name:>8} {tag:>6} (ep{i:>2}): need {r['need'] * 1e3:5.0f}  "
                  f"best {r['best'] * 1e3:5.1f}  final {r['final'] * 1e3:5.1f} mm"
                  f"  {'REACH' if r['reached'] else 'miss'} -> {path}")


if __name__ == "__main__":
    main()

"""Record Reacher-v5 episodes: DeePC vs a random-torque control, same episodes.

`reacher.xml` ships no camera and no lights, so both are supplied here -- a
top-down free camera framing the 0.21 m workspace, plus the built-in headlight
bumped so frames are not black. Both are `vis`/camera-level settings and cannot
affect a rollout.

The target is already a body in the model (the red sphere), so unlike
`panda/rendering.py` there is no scratch geom to inject and no `update_scene`
ordering trap. What IS drawn on top is the tolerance ring: the reach threshold is
1 cm against a 21 cm workspace, so without it a "miss" and a "reach" look
identical.

    uv run python scripts/record_reacher_video.py --episodes 4
    uv run python scripts/record_reacher_video.py --out-dir videos/reacher --fps 25
"""
from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

from core.video_encoding import encode_video
from reacher.deepc_setup import (
    anchor_grid, collect_anchor, make_controller, y_ref_for,
)
from reacher.model import (
    NQ_ARM, fingertip, frame_skip, is_reachable, reachable_annulus,
    sample_config, sample_goal, set_state, step_torque,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _ring(scene, centre, radius, rgba):
    """Draw the tolerance ball as a translucent sphere in the scratch scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0, 0], dtype=np.float64),
        np.array([centre[0], centre[1], 0.01], dtype=np.float64),
        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _disc(scene, radius, rgba):
    """Flat disc marking the inner radius the arm cannot fold past."""
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([radius, radius, 0.0005], dtype=np.float64),
        np.array([0.0, 0.0, 0.004], dtype=np.float64),
        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def classify(model, goal, final, tol):
    """Which failure mode is this? The four are genuinely different problems."""
    if not is_reachable(model, goal):
        return "unreachable"          # the safe box excludes it; not a control failure
    if final < tol:
        return "reached"
    if final < 3 * tol:
        return "near-miss"            # converged, stalled just outside tolerance
    return "failure"                  # never converged


def _annotate(frame, step, dist, tol, label, reached, mode=None):
    """Burn step / distance / verdict into the frame.

    Without this the viewer cannot tell a 6 mm reach from a 12 mm near-miss --
    both are a few pixels at this scale, which is exactly the confusion the first
    version of these videos caused.
    """
    from PIL import Image, ImageDraw

    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im)
    col = (20, 160, 20) if reached else (200, 60, 60)
    dr.rectangle([0, 0, im.width, 54], fill=(252, 252, 251))
    tag = f"   [{mode}]" if mode else ""
    dr.text((12, 8), f"{label}   step {step:>2}{tag}", fill=(11, 11, 11))
    dr.text((12, 30), f"fingertip-target  {dist * 1e3:6.1f} mm"
                      f"   (tolerance {tol * 1e3:.0f} mm)", fill=col)
    if reached:
        dr.text((im.width - 90, 30), "REACHED", fill=col)
    return np.asarray(im)


def record(model, data, renderer, cam, q0, goal, args, ctrl=None, rand_amp=None,
           mode=None):
    """Run one episode, returning (frames, reached, final_distance)."""
    fs = frame_skip(model)
    set_state(model, data, q0, goal)
    t0 = fingertip(data)
    need = float(np.linalg.norm(goal - t0))
    if ctrl is not None:
        ctrl.reset(np.concatenate([q0, t0]), u_initial=np.zeros(NQ_ARM))
        yref = y_ref_for(goal)
    rw = np.random.default_rng(int(abs(q0[0]) * 1e6) % 2**31)
    frames, best = [], need
    label = "DeePC" if ctrl is not None else "random torque"
    for _t in range(args.steps):
        renderer.update_scene(data, camera=cam)
        # AFTER update_scene (which zeroes ngeom) and BEFORE render.
        _ring(renderer.scene, goal, args.tol, [0.2, 0.9, 0.2, 0.30])
        if mode == "unreachable":
            # Show WHY: the goal sits inside the radius the arm cannot fold past.
            _disc(renderer.scene, reachable_annulus(model)[0], [0.82, 0.24, 0.24, 0.22])
        cur = float(np.linalg.norm(fingertip(data) - goal))
        frames.append(_annotate(renderer.render(), _t, cur, args.tol, label,
                                cur < args.tol, mode))
        if ctrl is not None:
            y = np.concatenate([np.asarray(data.qpos[:NQ_ARM]), fingertip(data)])
            try:
                u = ctrl.act(y, yref)
            except RuntimeError:
                break
        else:
            u = rw.uniform(-rand_amp, rand_amp, NQ_ARM)
        step_torque(model, data, u, fs)
        best = min(best, float(np.linalg.norm(fingertip(data) - goal)))
        if best < args.tol:
            for _ in range(args.hold):      # hold on the reached frame
                renderer.update_scene(data, camera=cam)
                _ring(renderer.scene, goal, args.tol, [0.2, 0.9, 0.2, 0.55])
                frames.append(_annotate(renderer.render(), _t + 1, best,
                                        args.tol, label, True, mode))
            break
    return frames, best < args.tol, best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, nargs=2, default=[6, 5])
    p.add_argument("--T", type=int, default=1200)
    p.add_argument("--scan", type=int, default=20,
                   help="episodes to classify before picking examples")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--hold", type=int, default=10, help="frames held after a reach")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--size", type=int, nargs=2, default=[720, 720])
    p.add_argument("--out-dir", default="videos/reacher")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # reacher.xml caps the offscreen framebuffer at 640; raise it before compiling
    # so the tolerance ball is legible. At the wide camera a 10 mm tolerance on a
    # 210 mm workspace is ~9 px at 480 -- a reach and a near-miss look identical.
    from reacher.model import model_path
    xml = open(model_path()).read().replace(
        "<worldbody>",
        f"<visual><global offwidth='{max(args.size)}' "
        f"offheight='{max(args.size)}'/></visual>\n<worldbody>", 1)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = 0.5      # reacher.xml ships no lights
    model.vis.headlight.diffuse[:] = 0.7
    renderer = mujoco.Renderer(model, height=args.size[0], width=args.size[1])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.0]
    cam.distance = 0.62                        # frames the 0.21 m reach envelope
    cam.elevation = -90.0                      # straight down: the arm is planar
    cam.azimuth = 90.0

    rng = np.random.default_rng(args.seed)
    anchors = anchor_grid(model, *args.grid)
    print(f"collecting {len(anchors)} libraries ...")
    payload = {"anchors": anchors}
    for i, a in enumerate(anchors):
        rec = collect_anchor(model, data, a, args.T, rng)
        payload[f"u_{i}"], payload[f"q_{i}"], payload[f"tip_{i}"] = (
            rec["u"], rec["q"], rec["tip"])
    deepc, _ = make_controller(payload, T_ini=5, N=12, lambda_g=5e-3)

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed + 99)
    eps = [(sample_config(model, data, rng)[0], sample_goal(rng))
           for _ in range(args.scan)]
    r_min, r_max = reachable_annulus(model)
    print(f"scanning {len(eps)} episodes; reachable annulus "
          f"[{r_min:.3f}, {r_max:.3f}] m")

    # Classify every episode first, then record ONE example of each mode. Recording
    # all of them would bury the informative cases: the interesting videos are the
    # three FAILURE modes, and they are a minority of episodes.
    scored = []
    for i, (q0, goal) in enumerate(eps):
        _, hit, final = record(model, data, renderer, cam, q0, goal, args,
                               ctrl=deepc)
        mode = classify(model, goal, final, args.tol)
        scored.append((i, q0, goal, final, mode))
        print(f"  ep{i:>2}  goal r={np.linalg.norm(goal):.3f}  "
              f"final {final * 1e3:6.1f} mm  -> {mode}")
    counts = {m: sum(1 for r in scored if r[4] == m) for m in
              ("reached", "near-miss", "failure", "unreachable")}
    print(f"\n  {counts}")

    print("\nrecording one example per mode (+ the random control on each):")
    for mode in ("reached", "near-miss", "failure", "unreachable"):
        picks = [r for r in scored if r[4] == mode]
        if not picks:
            print(f"  {mode:>12}: none in this scan")
            continue
        # Worst example of each failure mode, best of 'reached' -- the clearest case.
        i, q0, goal, final, _ = (min(picks, key=lambda r: r[3]) if mode == "reached"
                                 else max(picks, key=lambda r: r[3]))
        for label, kw in (("deepc", dict(ctrl=deepc)),
                          ("random", dict(rand_amp=1.0))):
            fr, hit, fin = record(model, data, renderer, cam, q0, goal, args,
                                  mode=mode, **kw)
            path = os.path.join(args.out_dir, f"{mode}_{label}.mp4")
            encode_video(fr, path, args.fps)
            print(f"  {mode:>12} {label:>6} (ep{i}): final {fin * 1e3:6.1f} mm  "
                  f"{len(fr)} frames -> {path}")


if __name__ == "__main__":
    main()

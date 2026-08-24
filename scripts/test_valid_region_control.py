"""Does DeePC work INSIDE the radius where its libraries are valid?

Every closed-loop episode run so far started at a random configuration 2-4 rad
from the nearest anchor -- the regime `scripts/verify_libraries.py` measures at
skill -9.9 and cos -0.03, i.e. the predictor points the wrong way half the time.
The controller was only ever tested outside its own domain of validity.

This starts the arm a controlled distance from an anchor and gives it a goal a
short hop away, so the whole episode stays inside (or outside) the measured
usable radius on purpose. It is the experiment that separates two very different
diagnoses:

  * reach rate high near the anchor, collapsing with distance
        -> the method WORKS and the open problem is purely COVERAGE
           (more anchors, or per-step data selection a la Select-DPC)
  * reach rate low even at the anchor
        -> something structural is still wrong and denser anchors are wasted work

A random-walk control of the same command magnitude runs on the identical
episodes, because a DeePC number without that reference has been misread once
already today: at the settings used earlier, `u = q + noise` beat the controller.

    uv run python scripts/test_valid_region_control.py
    uv run python scripts/test_valid_region_control.py --starts 0 0.5 --eps 4
"""
from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_anchors import _goal_at_joint_distance  # noqa: E402

from panda.model import (  # noqa: E402
    MIN_TIP_Z, frame_skip, load_model, safe_box, tip_id,
)
from panda.qdes import make_controller, step_qdes, y_ref_for  # noqa: E402


def _valid_at(model, data, base, dist, rng, lo, hi, tip, tries=60):
    """A valid configuration `dist` rad from `base` (dist=0 returns `base`)."""
    if dist == 0.0:
        return np.clip(base, lo, hi)
    for _ in range(tries):
        v = rng.standard_normal(model.nq)
        q = np.clip(base + dist * v / np.linalg.norm(v), lo, hi)
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if data.ncon == 0 and data.site_xpos[tip][2] > MIN_TIP_Z:
            return q
    return None


def episode(model, data, q0, goal, args, deepc=None, info=None, rand_amp=None):
    """One episode under DeePC or under a random walk. Returns metrics."""
    lo, hi = safe_box(model)
    fs, tip = frame_skip(model), tip_id(model)
    data.qpos[:] = q0
    data.qvel[:] = 0.0
    data.ctrl[:] = q0
    mujoco.mj_forward(model, data)
    tip0 = data.site_xpos[tip].copy()
    need = float(np.linalg.norm(goal - tip0))
    if deepc is not None:
        deepc.reset(np.concatenate([q0, tip0]), u_initial=q0)
        y_ref = y_ref_for(goal, info["nq"])
    rw = np.random.default_rng(int(abs(q0[0]) * 1e6) % 2**31)

    best, path, prev, steps = need, 0.0, tip0.copy(), args.steps
    for t in range(args.steps):
        q = np.asarray(data.qpos).copy()
        if deepc is not None:
            y = np.concatenate([q, np.asarray(data.site_xpos[tip])])
            try:
                u = deepc.act(y, y_ref)
            except RuntimeError:
                break
        else:
            u = q + rw.uniform(-rand_amp, rand_amp, model.nq)
        step_qdes(model, data, u, lo, hi, fs)
        path += float(np.linalg.norm(data.site_xpos[tip] - prev))
        prev = data.site_xpos[tip].copy()
        d = float(np.linalg.norm(data.site_xpos[tip] - goal))
        if d < best:
            best = d
        if d < args.tol:
            steps = t + 1
            break
    net = need - best
    return {"reached": best < args.tol, "need": need, "final": best,
            "closed": 100.0 * net / need, "steps": steps,
            # Path per unit of progress. A controller that STEERS runs ~1-2x;
            # measured 5-15x earlier, which is what "vigorous but undirected"
            # looks like numerically.
            "eff": path / max(net, 1e-6)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--libs", default="data/panda_anchors_k4_libs.npz")
    p.add_argument("--starts", type=float, nargs="+", default=[0.0, 0.5, 2.0],
                   help="joint distance from the anchor to start at, rad")
    p.add_argument("--goal-dist", type=float, default=0.4,
                   help="joint distance from start to goal, rad")
    p.add_argument("--eps", type=int, default=3, help="episodes per start distance")
    p.add_argument("--du-max", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--tol", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    model, data = load_model()
    lo, hi = safe_box(model)
    tip = tip_id(model)
    rng = np.random.default_rng(args.seed)
    with np.load(args.libs) as z:
        payload = {k: z[k] for k in z.files}
    anchors = payload["anchors"]
    deepc, info = make_controller(payload, model, du_max=args.du_max)

    print(f"du_max={args.du_max} rad/step ({args.du_max * 50:.1f} rad/s), "
          f"goal {args.goal_dist} rad from start, max {args.steps} steps, "
          f"tol={args.tol * 1e3:.0f} mm")
    print(f"max travel available: {args.steps * args.du_max:.2f} rad\n")
    print(f"  {'start dist':>11}{'':>3}{'reached':>9}{'median closed':>15}"
          f"{'median path/net':>17}")
    for sd in args.starts:
        eps = []
        while len(eps) < args.eps:
            a = anchors[rng.integers(len(anchors))]
            q0 = _valid_at(model, data, a, sd, rng, lo, hi, tip)
            if q0 is None:
                continue
            _, goal = _goal_at_joint_distance(model, data, q0, args.goal_dist,
                                              rng, lo, hi, tip)
            if goal is not None:
                eps.append((q0, goal))
        for label, kw in [("DeePC", dict(deepc=deepc, info=info)),
                          ("random", dict(rand_amp=args.du_max))]:
            res = [episode(model, data, q0, goal, args, **kw) for q0, goal in eps]
            hits = sum(r["reached"] for r in res)
            print(f"  {sd:>11.2f}{label:>9}{hits:>4}/{len(res):<4}"
                  f"{np.median([r['closed'] for r in res]):>13.1f}%"
                  f"{np.median([r['eff'] for r in res]):>17.1f}", flush=True)
        print()

    print("Read: if DeePC reaches near the anchor and collapses with start distance,")
    print("the method works and the open problem is COVERAGE. If it fails even at")
    print("distance 0, denser anchors will not help.")


if __name__ == "__main__":
    main()

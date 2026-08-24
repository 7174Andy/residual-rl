"""Are the anchor clusters actually LTI? Test it rather than assume it.

DeePC's precondition (Willems' lemma) is that the data comes from ONE linear
time-invariant system. For this plant that claim decomposes into three separate
questions, and they fail in different ways:

  [T] Time-invariance -- FREE. MuJoCo's dynamics have no explicit time
      dependence: `mj_step` from the same (qpos, qvel, ctrl) gives the same result
      at any point in a run. So this half of "LTI" holds by construction and is
      asserted here rather than measured on a sweep.

  [L] Linearity AT A POINT -- superposition and homogeneity of the input->output
      map, in deviation coordinates about the free response. Holds in the limit of
      small commands by differentiability, so the only meaningful question is how
      large the violation is at the amplitude actually used (sigma = 0.25 rad).

  [S] Spatial invariance ACROSS THE CLUSTER -- whether the SAME linear map holds
      everywhere in the region. This is the one the anchor plan really needs: a
      map that changes with `q` is LPV, not LTI, and a single Hankel library
      cannot represent it. Measured against each cluster's own measured radius.

Reported in deviation coordinates from the free response, because the plant is
affine, not linear: holding `u = q` still produces motion (gravity sag), so raw
superposition would fail trivially on the offset alone.

    uv run python scripts/test_cluster_lti.py
    uv run python scripts/test_cluster_lti.py --amps 0.05 0.25 --radii 0 1 2 4
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np

from panda.model import frame_skip, load_model, safe_box, tip_id

N_HORIZON = 12


def rollout(model, data, q_base, deltas, lo, hi, fs, tip):
    """Response to `u_t = q_base + delta_t`. Returns y = [q; tip] per step."""
    data.qpos[:] = q_base
    data.qvel[:] = 0.0
    data.ctrl[:] = q_base
    mujoco.mj_forward(model, data)
    out = np.empty((len(deltas), model.nq + 3))
    for t, dl in enumerate(deltas):
        data.ctrl[:] = np.clip(q_base + dl, lo, hi)
        mujoco.mj_step(model, data, nstep=fs)
        mujoco.mj_forward(model, data)
        out[t] = np.concatenate([data.qpos, data.site_xpos[tip]])
    return out


def deviation(model, data, q_base, deltas, free, lo, hi, fs, tip):
    return rollout(model, data, q_base, deltas, lo, hi, fs, tip) - free


def linearity(model, data, q_base, amp, rng, lo, hi, fs, tip, n=12):
    """Superposition and homogeneity error, relative, at the final horizon step."""
    free = rollout(model, data, q_base, np.zeros((N_HORIZON, model.nq)), lo, hi, fs, tip)
    sup, hom = [], []
    for _ in range(n):
        d1 = rng.uniform(-amp, amp, (N_HORIZON, model.nq))
        d2 = rng.uniform(-amp, amp, (N_HORIZON, model.nq))
        y1 = deviation(model, data, q_base, d1, free, lo, hi, fs, tip)
        y2 = deviation(model, data, q_base, d2, free, lo, hi, fs, tip)
        y12 = deviation(model, data, q_base, d1 + d2, free, lo, hi, fs, tip)
        # Tip block only: it is the channel with a physical scale, and the one the
        # controller tracks. Mixing rad and m into one norm would hide which fails.
        a, b = y12[-1, model.nq:], (y1 + y2)[-1, model.nq:]
        sup.append(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))
        y2x = deviation(model, data, q_base, 2 * d1, free, lo, hi, fs, tip)
        c, e = y2x[-1, model.nq:], 2 * y1[-1, model.nq:]
        hom.append(np.linalg.norm(c - e) / max(np.linalg.norm(e), 1e-12))
    return float(np.median(sup)), float(np.median(hom))


def fit_map(model, data, q_base, amp, rng, lo, hi, fs, tip, n):
    """Least-squares linear map from the flattened command deviation to y."""
    free = rollout(model, data, q_base, np.zeros((N_HORIZON, model.nq)), lo, hi, fs, tip)
    D = rng.uniform(-amp, amp, (n, N_HORIZON, model.nq))
    Y = np.stack([deviation(model, data, q_base, d, free, lo, hi, fs, tip).ravel()
                  for d in D])
    G, *_ = np.linalg.lstsq(D.reshape(n, -1), Y, rcond=None)
    return G, D, Y, free


def spatial(model, data, anchor, radius, amp, rng, lo, hi, fs, tip, n):
    """Does the anchor's linear map still hold `radius` rad away? Relative error."""
    G, D, Y, _ = fit_map(model, data, anchor, amp, rng, lo, hi, fs, tip, n)
    if radius == 0.0:
        pred = D.reshape(n, -1) @ G
        d = (pred - Y).reshape(n, -1, model.nq + 3)[:, -1, model.nq:]
        t = Y.reshape(n, -1, model.nq + 3)[:, -1, model.nq:]
        return float(np.median(np.linalg.norm(d, axis=1) / np.linalg.norm(t, axis=1)))
    errs = []
    for _ in range(6):
        v = rng.standard_normal(model.nq)
        q_t = np.clip(anchor + radius * v / np.linalg.norm(v), lo, hi)
        free_t = rollout(model, data, q_t, np.zeros((N_HORIZON, model.nq)),
                         lo, hi, fs, tip)
        Yt = np.stack([deviation(model, data, q_t, d, free_t, lo, hi, fs, tip).ravel()
                       for d in D[:24]])
        pred = D[:24].reshape(24, -1) @ G
        d = (pred - Yt).reshape(24, -1, model.nq + 3)[:, -1, model.nq:]
        t = Yt.reshape(24, -1, model.nq + 3)[:, -1, model.nq:]
        errs.append(np.median(np.linalg.norm(d, axis=1) / np.linalg.norm(t, axis=1)))
    return float(np.median(errs))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anchors", default="data/panda_anchors_k4.npz")
    p.add_argument("--amps", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.25, 0.40])
    p.add_argument("--radii", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--fit-n", type=int, default=120, help="samples for the fitted map")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, data = load_model()
    lo, hi = safe_box(model)
    fs, tip = frame_skip(model), tip_id(model)
    with np.load(args.anchors) as z:
        anchors, radius = z["anchors"], z["radius"]
    rng = np.random.default_rng(args.seed)

    print("[T] Time-invariance: holds BY CONSTRUCTION -- MuJoCo's dynamics carry no")
    print("    explicit time dependence, so identical (qpos, qvel, ctrl) always give")
    print("    an identical step. Verifying:")
    q = anchors[0]
    d = rng.uniform(-0.2, 0.2, (N_HORIZON, model.nq))
    r1 = rollout(model, data, q, d, lo, hi, fs, tip)
    _ = rollout(model, data, anchors[1], d, lo, hi, fs, tip)   # perturb the sim between
    r2 = rollout(model, data, q, d, lo, hi, fs, tip)
    print(f"    max |repeat - original| = {np.abs(r1 - r2).max():.2e}   -> time-invariant\n")

    print("[L] Linearity at the anchor, median relative error of the TIP response")
    print("    (superposition: y(d1+d2) vs y(d1)+y(d2);  homogeneity: y(2d) vs 2y(d))")
    print(f"\n    {'amplitude':>10}" + "".join(f"{f'anchor {j}':>22}" for j in range(len(anchors))))
    print(f"    {'(rad)':>10}" + "".join(f"{'superpos   homog':>22}" for _ in anchors))
    for amp in args.amps:
        row = f"    {amp:>10.2f}"
        for a in anchors:
            s, h = linearity(model, data, a, amp, rng, lo, hi, fs, tip)
            row += f"{s * 100:>13.1f}%{h * 100:>8.1f}%"
        print(row)

    print("\n[S] Spatial invariance: does the ANCHOR's linear map hold at radius r?")
    print("    median relative tip error, fitted at amplitude 0.25 rad")
    print(f"\n    {'radius (rad)':>13}" + "".join(f"{f'anchor {j}':>12}" for j in range(len(anchors))))
    for r in args.radii:
        row = f"    {r:>13.2f}"
        for a in anchors:
            row += f"{spatial(model, data, a, r, 0.25, rng, lo, hi, fs, tip, args.fit_n) * 100:>11.0f}%"
        print(row)
    print(f"\n    measured cluster radii (from k-medoids): {np.round(radius, 2)} rad")
    print("    -> read the [S] row nearest each cluster's own radius; that is the")
    print("       regime one library is actually being asked to cover.")


if __name__ == "__main__":
    main()

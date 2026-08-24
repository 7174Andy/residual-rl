"""How far from an anchor does one local-linear model still predict?

That is the question anchor placement actually turns on. A DeePC library asserts
that over the prediction horizon the map `u -> y` is LTI; the library is valid
wherever that holds. So measure it directly:

  1. At an anchor configuration, drive the plant with many random admissible delta
     sequences of length N and least-squares fit the linear map `Y = U @ G + free`.
     This IS the content of the Hankel constraint, fit explicitly so it can be
     transported.
  2. At a configuration offset from the anchor, apply the SAME sequences and ask
     whether the anchor's `G` still predicts, given the test point's own free
     response (which `y_ini` supplies to DeePC, so handing it over is fair).
  3. Report the offset at which the horizon-N tip error crosses a threshold.

The threshold is half the 0.05 m `goal_tolerance`: a model that misplaces the tip
by more than 25 mm at the horizon cannot be steering to a 50 mm target on it.

Part 0 checks the premise of the CURRENT keying. Joint 1's axis is vertical and
gravity is along the same axis, so rotating q1 rotates the whole arm about z and
leaves the joint-space dynamics invariant. If that holds numerically, azimuth is a
symmetry direction, not a nonlinearity direction, and dividing anchors along it
buys nothing that a coordinate rotation would not.

    uv run python scripts/measure_linear_region.py
    uv run python scripts/measure_linear_region.py --joints 1 3 --n-samples 400
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np

from panda.model import frame_skip, load_model, safe_box, tip_id

N_HORIZON = 12          # matches build_canonical_panda_deepc's N
DELTA_PROBE = 0.1       # half of env DELTA_MAX: excites without riding the box edge
THRESH_M = 0.025        # half of goal_tolerance


def rollout(model, data, q0, U, fs, tip, lo, hi):
    """Tip trajectory over len(U) control steps from rest at `q0`. Returns (N,3).

    Raises if the safe-box clip fires -- a clipped step means the applied input is
    not `U`, which would silently corrupt both the fit and the transfer test.
    """
    data.qpos[:] = q0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    out = np.empty((len(U), 3))
    for t, u in enumerate(U):
        target = data.qpos + u
        if np.any(target < lo - 1e-12) or np.any(target > hi + 1e-12):
            raise ValueError(f"safe-box clip at step {t}; offset too large for probe")
        data.ctrl[:] = target
        mujoco.mj_step(model, data, nstep=fs)
        mujoco.mj_forward(model, data)      # site_xpos is stale otherwise (primer sec. 3)
        out[t] = data.site_xpos[tip]
    return out


def responses(model, data, q0, Us, fs, tip, lo, hi):
    """(n_samples, N*3) tip responses, and the (N*3,) free response at `q0`."""
    Y = np.stack([rollout(model, data, q0, U, fs, tip, lo, hi).ravel() for U in Us])
    free = rollout(model, data, q0, np.zeros_like(Us[0]), fs, tip, lo, hi).ravel()
    return Y, free


def fit_slope(Us, Y, free):
    """Least-squares G with the free response removed: Y - free ~ U_flat @ G."""
    A = Us.reshape(len(Us), -1)
    G, *_ = np.linalg.lstsq(A, Y - free, rcond=None)
    return G


def horizon_err(Us, Y_true, free_test, G_anchor):
    """RMS euclidean tip error at the final horizon step, in metres."""
    pred = Us.reshape(len(Us), -1) @ G_anchor + free_test
    d = (pred - Y_true).reshape(len(Us), -1, 3)[:, -1, :]
    return float(np.sqrt(np.mean(np.linalg.norm(d, axis=1) ** 2)))


def check_q1_equivariance(model, data, fs, tip, lo, hi, rng, q_base, n=8):
    """Same deltas at two q1 values agree on tip once de-rotated about z."""
    Us = rng.uniform(-DELTA_PROBE, DELTA_PROBE, (n, N_HORIZON, model.nu))
    qa, qb = q_base.copy(), q_base.copy()
    qa[0], qb[0] = -1.8, 0.6
    dq = qb[0] - qa[0]
    Rz = np.array([[np.cos(dq), -np.sin(dq), 0], [np.sin(dq), np.cos(dq), 0], [0, 0, 1]])
    worst = 0.0
    for U in Us:
        ta = rollout(model, data, qa, U, fs, tip, lo, hi)
        tb = rollout(model, data, qb, U, fs, tip, lo, hi)
        worst = max(worst, float(np.abs(ta @ Rz.T - tb).max()))
    return worst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--joints", type=int, nargs="+", default=list(range(7)),
                   help="joint indices to sweep the anchor offset along")
    p.add_argument("--n-samples", type=int, default=250,
                   help="probe sequences per point; must exceed N*nu=84 to fit")
    p.add_argument("--max-offset", type=float, default=1.2,
                   help="largest joint offset from the anchor, rad")
    p.add_argument("--steps", type=int, default=8, help="offsets per joint")
    p.add_argument("--probe", type=float, default=DELTA_PROBE,
                   help="delta amplitude of the probe sequences, rad")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    probe = args.probe

    model, data = load_model()
    fs, tip = frame_skip(model), tip_id(model)
    lo, hi = safe_box(model)
    rng = np.random.default_rng(args.seed)
    q_anchor = np.asarray(model.key_qpos[0], dtype=np.float64).copy()
    q_anchor[0] = -0.6          # a PANDA_ANCHOR_Q1 value, interior to the box

    if args.n_samples <= N_HORIZON * model.nu:
        raise SystemExit(f"--n-samples must exceed {N_HORIZON * model.nu}")

    print(f"anchor q = {np.round(q_anchor, 4)}")
    print(f"N={N_HORIZON}  delta_probe={probe}  n_samples={args.n_samples}  "
          f"threshold={THRESH_M * 1e3:.0f} mm\n")

    worst = check_q1_equivariance(model, data, fs, tip, lo, hi, rng, q_anchor)
    print(f"[0] q1 equivariance: max |Rz(dq)*tip(q1=-1.8) - tip(q1=0.6)| = {worst:.3e} m")
    print("    -> azimuth is a SYMMETRY of the plant, not a nonlinearity axis.\n")
    # This is the script's self-check as well as its first result. Joint 1's axis is
    # z and gravity is along z, so the joint-space dynamics cannot depend on q1 --
    # an exact geometric invariant, measured at 6.7e-16 m. If it ever fails, either
    # `rollout` is broken (stale kinematics, leaked state between calls) or the model
    # gained a q1-dependent term; both invalidate everything printed below.
    assert worst < 1e-6, (
        f"q1 equivariance broken ({worst:.3e} m). Either rollout() leaks state "
        "between calls or the model is no longer rotationally symmetric about z."
    )

    Us = rng.uniform(-probe, probe, (args.n_samples, N_HORIZON, model.nu))
    Y_a, free_a = responses(model, data, q_anchor, Us, fs, tip, lo, hi)
    G = fit_slope(Us, Y_a, free_a)
    rms_a = horizon_err(Us, Y_a, free_a, G)
    print(f"[1] anchor self-fit residual (the noise floor): {rms_a * 1e3:.3f} mm\n")

    print(f"[2] transfer error at horizon {N_HORIZON}, RMS mm "
          "-- '-' = probe left the safe box")
    offsets = np.linspace(0.0, args.max_offset, args.steps + 1)[1:]
    print("      offset:" + "".join(f"{o:>9.2f}" for o in offsets))
    for j in args.joints:
        cells, cross = [], None
        for o in offsets:
            q_t = q_anchor.copy()
            q_t[j] += o
            if np.any(q_t < lo) or np.any(q_t > hi):
                q_t[j] = q_anchor[j] - o          # try the other direction
                if np.any(q_t < lo) or np.any(q_t > hi):
                    cells.append("        -")
                    continue
            try:
                Y_t, free_t = responses(model, data, q_t, Us, fs, tip, lo, hi)
            except ValueError:
                cells.append("        -")
                continue
            rms = horizon_err(Us, Y_t, free_t, G)
            cells.append(f"{rms * 1e3:>9.1f}")
            if cross is None and rms > THRESH_M:
                cross = o
        tag = f"<{cross:.2f}" if cross is not None else f">{offsets[-1]:.2f}"
        print(f"  joint {j + 1}:" + "".join(cells) + f"   | valid radius {tag} rad")

    print(f"\nCell radius = the offset before crossing {THRESH_M * 1e3:.0f} mm. "
          "Anchor spacing along a joint should be ~2x its radius.")


if __name__ == "__main__":
    main()

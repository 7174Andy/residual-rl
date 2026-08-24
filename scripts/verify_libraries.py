"""Are the Hankel libraries useful to DeePC? Three tests, none needing a QP.

A library can fail DeePC in three escalating ways, and only the third explains a
controller that loses to a random walk:

  [1] SPAN -- Willems' lemma says any trajectory of the LTI system lies in the
      column span of a persistently-exciting Hankel. Project a fresh trajectory
      onto the span and measure the residual. A large residual means the library
      cannot even REPRESENT what the plant does, so every downstream constraint
      is being satisfied by slack rather than by data.

  [2] SKILL -- prediction error is meaningless without a reference. Compare the
      library's prediction against the trivial "the tip does not move" predictor.
      Skill = 1 - MSE_library / MSE_no_motion. Positive means the library beats
      assuming nothing happens; NEGATIVE means you would do better ignoring it.

  [3] DIRECTION -- the controller does not need an accurate magnitude, it needs
      the right WAY TO GO. Cosine between the predicted tip displacement and the
      true one, over the horizon. cos ~ 1 steers correctly; cos ~ 0 carries no
      directional information (a random walk ties it); cos < 0 steers backwards,
      which is strictly worse than random and is the only thing that explains
      `u = q + noise` beating this controller.

All three are reported against DISTANCE FROM THE ANCHOR, because that is the
variable `test_cluster_lti.py` shows the local model degrading along, and the
one anchor placement controls.

    uv run python scripts/verify_libraries.py
    uv run python scripts/verify_libraries.py --radii 0.1 0.25 0.5 1 2 4

Uses an l2 (Tikhonov) solve for `g`, while `core/deepc.py` uses l1. l1 is sparser
-- it leans on FEWER columns -- so it cannot extrapolate better than the l2 fit
reported here. Read these numbers as an upper bound on the library's usefulness.
"""
from __future__ import annotations

import argparse

import numpy as np

from panda.model import load_model, safe_box
from panda.qdes import build_libraries, collect_anchor, outputs


def probe(library, rec, T_ini, N, p_y, nq, lam):
    """Span residual, skill vs no-motion, and direction cosine for one trajectory."""
    Up, Uf, Yp, Yf = library
    u, y = rec["u"], outputs(rec["q"], rec["tip"])
    u_ini, y_ini = u[:T_ini].ravel(), y[:T_ini].ravel()
    u_f, y_f = u[T_ini:T_ini + N].ravel(), y[T_ini:T_ini + N]

    # [1] Can the library REPRESENT this trajectory at all?
    A_full = np.vstack([Up, Yp, Uf, Yf])
    b_full = np.concatenate([u_ini, y_ini, u_f, y_f.ravel()])
    g_span, *_ = np.linalg.lstsq(A_full, b_full, rcond=None)
    span = np.linalg.norm(A_full @ g_span - b_full) / np.linalg.norm(b_full)

    # [2]/[3] Predict the future from the past + the known future input.
    A = np.vstack([Up, Yp, Uf])
    b = np.concatenate([u_ini, y_ini, u_f])
    g = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
    pred = (Yf @ g).reshape(N, p_y)

    tip_true, tip_pred = y_f[:, nq:], pred[:, nq:]
    tip_start = y[T_ini - 1, nq:]
    no_motion = np.tile(tip_start, (N, 1))
    mse_lib = np.mean(np.sum((tip_pred - tip_true) ** 2, axis=1))
    mse_nil = np.mean(np.sum((no_motion - tip_true) ** 2, axis=1))
    skill = 1.0 - mse_lib / max(mse_nil, 1e-15)

    dp, dt = tip_pred[-1] - tip_start, tip_true[-1] - tip_start
    denom = np.linalg.norm(dp) * np.linalg.norm(dt)
    cos = float(dp @ dt / denom) if denom > 1e-12 else 0.0
    return float(span), float(skill), cos


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--libs", default="data/panda_anchors_k4_libs.npz")
    p.add_argument("--radii", type=float, nargs="+",
                   default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--n", type=int, default=10, help="test trajectories per cell")
    p.add_argument("--T-ini", type=int, default=5)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--lambda-g", type=float, default=5e-3)
    p.add_argument("--gravity-comp", action="store_true",
                   help="offset ctrl by qfrc_bias/kp; MUST match between "
                        "collection and verification")
    p.add_argument("--servo-scale", type=float, default=1.0,
                   help="multiply the PD servo gains; must MATCH between "
                        "collection and verification or the libraries "
                        "describe a different plant than the one probed")
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()

    model, data = load_model(servo_scale=args.servo_scale)
    lo, hi = safe_box(model)
    rng = np.random.default_rng(args.seed)
    with np.load(args.libs) as z:
        payload = {k: z[k] for k in z.files}
    anchors = payload["anchors"]
    libs = build_libraries(payload, args.T_ini, args.N)
    nq, p_y = model.nq, model.nq + 3

    print(f"{len(anchors)} libraries, {args.n} fresh trajectories per radius, "
          f"sigma={args.sigma}, T_ini={args.T_ini}, N={args.N}")
    print("\n  span   = relative residual projecting a real trajectory onto the "
          "Hankel span (0 = perfect)")
    print("  skill  = 1 - MSE_library/MSE_no-motion   (>0 useful, <0 worse than "
          "assuming nothing moves)")
    print("  cos    = direction of predicted vs true tip displacement "
          "(1 = right way, 0 = no info, <0 = backwards)")
    print(f"\n  {'radius':>8}{'span':>10}{'skill':>10}{'cos':>10}"
          f"{'cos>0.5':>10}{'cos<0':>8}")
    for r in args.radii:
        S, K, C = [], [], []
        for i, a in enumerate(anchors):
            for _ in range(args.n):
                if r == 0.0:
                    q0 = a.copy()
                else:
                    v = rng.standard_normal(nq)
                    q0 = np.clip(a + r * v / np.linalg.norm(v), lo, hi)
                rec = collect_anchor(model, data, q0, args.T_ini + args.N + 1, rng,
                                     sigma=args.sigma,
                                     gravity_comp=args.gravity_comp)
                s, k, c = probe(libs[i], rec, args.T_ini, args.N, p_y, nq,
                                args.lambda_g)
                S.append(s)
                K.append(k)
                C.append(c)
        S, K, C = map(np.array, (S, K, C))
        print(f"  {r:>8.2f}{np.median(S):>10.3f}{np.median(K):>10.2f}"
              f"{np.median(C):>10.2f}{100 * np.mean(C > 0.5):>9.0f}%"
              f"{100 * np.mean(C < 0):>7.0f}%")

    print("\n  A library is USEFUL to DeePC only where skill > 0 and cos is near 1.")
    print("  Where cos ~ 0 the predictor carries no directional information and a")
    print("  random walk of equal magnitude ties it; where cos < 0 it steers the")
    print("  wrong way and random BEATS it -- which is the measured outcome.")


if __name__ == "__main__":
    main()

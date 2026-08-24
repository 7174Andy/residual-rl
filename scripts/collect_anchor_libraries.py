"""Stage 2 of the anchor-selection plan: one local dataset per anchor.

Covers plan section 6 (offline steps 8-10). Applies bounded, persistently
exciting `q_des` commands in a neighbourhood of each anchor and records the
input/output trajectories a Hankel library is built from.

Stored as `u`, `q` and `tip`. With the output `y = [q; p_ee]` the libraries are
goal-independent -- the goal enters only through `y_ref` -- so one collection
serves every goal. (The plan's scalar `d_g` output would not have been: see
`panda/qdes.py`'s module docstring for the measurement that motivated the change.)

Gates on rank, the same early warning `panda/data_collection.py` applies: a
library that is not persistently exciting cannot support a DeePC solve, and
finding that out here costs seconds rather than a closed-loop run.

    uv run python scripts/collect_anchor_libraries.py --anchors data/panda_anchors_k4.npz
    uv run python scripts/collect_anchor_libraries.py --T 3000 --sigma 0.15
"""
from __future__ import annotations

import argparse

import numpy as np

from core.hankel import build_hankel
from panda.model import load_model
from panda.qdes import DEFAULT_T, OU_SIGMA, OU_THETA, collect_anchor, outputs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anchors", default="data/panda_anchors_k4.npz")
    p.add_argument("--out", default=None, help="default: <anchors>_libs.npz")
    p.add_argument("--T", type=int, default=DEFAULT_T, help="steps per library")
    p.add_argument("--sigma", type=float, default=OU_SIGMA,
                   help="rad, excitation std around the anchor")
    p.add_argument("--theta", type=float, default=OU_THETA, help="OU correlation")
    p.add_argument("--T-ini", type=int, default=5)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--gravity-comp", action="store_true",
                   help="offset ctrl by qfrc_bias/kp; MUST match between "
                        "collection and verification")
    p.add_argument("--servo-scale", type=float, default=1.0,
                   help="multiply the PD servo gains; must MATCH between "
                        "collection and verification or the libraries "
                        "describe a different plant than the one probed")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = args.out or args.anchors.replace(".npz", "_libs.npz")
    model, data = load_model(servo_scale=args.servo_scale)
    rng = np.random.default_rng(args.seed)
    with np.load(args.anchors) as z:
        anchors = z["anchors"]
        weights = z["weights"]

    nq = model.nq
    # Full row rank of the stacked Hankel needs m_u*(T_ini+N) + n_state rows
    # covered; n_state = 2*nv = 14 (verified against mjd_transitionFD).
    floor = nq * (args.T_ini + args.N) + 2 * model.nv
    print(f"collecting {len(anchors)} libraries, T={args.T}, sigma={args.sigma} rad")
    print(f"  rank floor m_u(T_ini+N) + n_state = {nq}*{args.T_ini + args.N} + 14 = {floor}")

    payload: dict = {"anchors": anchors, "T": np.asarray(args.T),
                     "excitation": np.array([args.theta, args.sigma]),
                     "weights": weights, "T_ini": np.asarray(args.T_ini),
                     "N": np.asarray(args.N)}
    print(f"\n  {'lib':>4}{'n_cols':>9}{'rank':>7}{'floor':>7}{'cols/rows':>12}"
          f"{'reach (rad)':>14}")
    ok = True
    for i, a in enumerate(anchors):
        rec = collect_anchor(model, data, a, args.T, rng,
                             gravity_comp=args.gravity_comp,
                             theta=args.theta, sigma=args.sigma)
        payload[f"u_{i}"] = rec["u"]
        payload[f"q_{i}"] = rec["q"]
        payload[f"tip_{i}"] = rec["tip"]
        y = outputs(rec["q"], rec["tip"])
        blocks = build_hankel(rec["u"], y, T_ini=args.T_ini, N=args.N)
        M = np.vstack(blocks)
        rank = int(np.linalg.matrix_rank(M))
        reach = float(np.abs(rec["q"] - a).max())
        good = rank >= floor
        ok &= good
        print(f"  {i:>4}{blocks[0].shape[1]:>9}{rank:>7}{floor:>7}"
              f"{blocks[0].shape[1] / M.shape[0]:>12.2f}{reach:>14.3f}"
              f"{'' if good else '   RANK DEFICIENT'}")

    np.savez(out, **payload)
    print(f"\nwrote {out}")
    if not ok:
        raise SystemExit("at least one library is rank deficient -- raise --T or --sigma")
    print(f"next: uv run python scripts/validate_anchors.py --libs {out}")


if __name__ == "__main__":
    main()

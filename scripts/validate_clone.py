# scripts/validate_clone.py
"""Run the layered fidelity gate on a trained clone and print the report.

Usage:
    uv run python scripts/validate_clone.py --clone data/clone.pt \
        --data data/clone_dataset.npz --n_seeds 78 --base_seed 4104626029
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from rl.clone import load_clone
from two_wheel_robot.rl.clone_eval import paired_outcomes, regression_by_regime
from two_wheel_robot.rl.deepc_setup import build_canonical_deepc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clone", default="data/clone.pt")
    p.add_argument("--data", default="data/clone_dataset.npz")
    p.add_argument("--libraries", default="data/libraries_v0.npz")
    p.add_argument("--n_seeds", type=int, default=78)
    p.add_argument("--base_seed", type=int, default=4104626029)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    predictor = load_clone(args.clone, device=args.device)
    deepc, info = build_canonical_deepc(libraries_path=args.libraries)

    # (1) regime-conditioned open-loop regression. Score on the held-out
    # validation split (stored in the checkpoint) so the numbers aren't biased
    # by training-on-test; fall back to the full set if the split can't be
    # matched to this dataset.
    ds = np.load(args.data, allow_pickle=True)
    feats, targs, regime = ds["features"], ds["targets"], ds["regime"]
    val_idx = predictor.val_idx
    if val_idx is not None and predictor.n_train_samples == feats.shape[0]:
        feats, targs, regime = feats[val_idx], targs[val_idx], regime[val_idx]
        scope = f"held-out, n={len(val_idx)}"
    else:
        scope = f"FULL dataset, n={feats.shape[0]} — no matching split; numbers optimistic"
    pred = predictor.predict(feats)
    print(f"=== (1) regression by regime [{scope}] ===")
    for r, m in regression_by_regime(pred, targs, regime).items():
        print(f"  {r:11s} n={m['n']:6d}  MAE v={m['mae_v']:.4f} w={m['mae_w']:.4f}  "
              f"RMSE v={m['rmse_v']:.4f} w={m['rmse_w']:.4f}")

    # (2)+(3) closed-loop fidelity + paired per-seed outcomes.
    seeds = [args.base_seed + i for i in range(args.n_seeds)]
    print(f"\n=== (2)/(3) closed-loop over {len(seeds)} seeds "
          f"(base {args.base_seed}) ===")
    rep = paired_outcomes(deepc, predictor, info, seeds, info["action_bounds"])
    print(f"  confusion        : {rep['confusion']}")
    print(f"  agreement rate   : {rep['agreement_rate']:.3f}")
    print(f"  McNemar p        : {rep['mcnemar_p']:.4f}  "
          f"(high = clone agrees seed-by-seed)")
    print(f"  DeePC reach rate : {rep['deepc_reach_rate']:.3f}  "
          f"CI {tuple(round(x, 3) for x in rep['deepc_reach_ci'])}")
    print(f"  clone reach rate : {rep['clone_reach_rate']:.3f}  "
          f"CI {tuple(round(x, 3) for x in rep['clone_reach_ci'])}")
    print(f"  traj pos dev     : median {rep['traj_pos_median']:.3f}  "
          f"median-of-p95 {rep['traj_pos_median_p95']:.3f}")
    print("\nGate: pass if trajectory deviation is bounded AND agreement is high "
          "(McNemar not significant). Marginal reach-rate parity alone is NOT enough.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

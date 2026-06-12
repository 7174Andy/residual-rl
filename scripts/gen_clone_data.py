# scripts/gen_clone_data.py
"""Generate the deep-lcc clone dataset and cache it to a .npz.

Usage:
    uv run python scripts/gen_clone_data.py --out data/clone_dataset.npz \
        --n_synthetic 20000 --p_degenerate 0.25 --n_onpolicy 100 --seed 0
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from two_wheel_robot.rl.clone_data import generate_clone_dataset, save_dataset
from two_wheel_robot.rl.deepc_setup import build_canonical_deepc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/clone_dataset.npz")
    p.add_argument("--libraries", default="data/libraries_v0.npz")
    p.add_argument("--n_synthetic", type=int, default=20000)
    p.add_argument("--p_degenerate", type=float, default=0.25)
    p.add_argument("--n_onpolicy", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    deepc, info = build_canonical_deepc(libraries_path=args.libraries)
    ds = generate_clone_dataset(
        deepc, info,
        n_synthetic=args.n_synthetic,
        p_degenerate=args.p_degenerate,
        n_onpolicy_episodes=args.n_onpolicy,
        seed=args.seed,
    )
    save_dataset(args.out, ds)
    n = ds["features"].shape[0]
    regs, counts = np.unique(ds["regime"], return_counts=True)
    print(f"wrote {args.out}: {n} samples, dim {ds['features'].shape[1]}")
    print("  regimes:", dict(zip(regs.tolist(), counts.tolist())))
    return 0


if __name__ == "__main__":
    sys.exit(main())

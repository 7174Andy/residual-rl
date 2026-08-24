"""Collect the Select-DPC clone dataset.

The expensive step: 76.7 ms per control step x 50 steps x N episodes, plus the
anchor collection. 200 episodes is roughly 15 minutes.

    uv run python scripts/gen_reacher_clone_data.py --episodes 200
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from reacher.clone_data import generate_clone_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--n-cols", type=int, default=300)
    p.add_argument("--n-max", type=int, default=1)
    p.add_argument("--grid", type=int, nargs=2, default=[6, 5])
    p.add_argument("--T", type=int, default=1200)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--out", default="data/reacher_clone_dataset.npz")
    p.add_argument("--base", default="select", choices=["select", "fixed"],
                   help="'fixed' is spec R1's fallback: worse controller, "
                        "smooth-in-time law, therefore clonable")
    p.add_argument("--memoryless", action="store_true",
                   help="clear Select-DPC's carried prediction each step, making "
                        "its action a FUNCTION of the clone's features")
    p.add_argument("--bank-seed", type=int, default=0,
                   help="seeds the anchor collection only; hold it FIXED across "
                        "chunks so every chunk shares one identical controller")
    p.add_argument("--episode-offset", type=int, default=0,
                   help="shifts episode seeds so chunks cover different starts")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.monotonic()
    ds = generate_clone_dataset(
        n_episodes=args.episodes, seed=args.seed, grid=tuple(args.grid),
        T=args.T, n_cols=args.n_cols, n_max=args.n_max, stride=args.stride,
        base=args.base, bank_seed=args.bank_seed,
        carry_prediction=not args.memoryless,
        episode_offset=args.episode_offset)
    elapsed = time.monotonic() - t0

    meta = ds["meta"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, features=ds["features"], actions=ds["actions"],
             **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})
    kept = meta["n_episodes"] - meta["n_dropped"]
    print(f"wrote {args.out}  ({elapsed / 60:.1f} min, base={args.base})")
    print(f"  {ds['features'].shape[0]} rows x {ds['features'].shape[1]} features")
    print(f"  bank {meta['bank_columns']} columns, "
          f"{meta['n_reached']}/{kept} episodes reached "
          f"({100 * meta['n_reached'] / max(kept, 1):.0f}%)")
    print(f"  |action| median {np.median(np.abs(ds['actions'])):.3f}, "
          f"saturated at +-1 on {100 * np.mean(np.abs(ds['actions']) > 0.999):.0f}% "
          f"of components")
    if meta["n_dropped"]:
        print(f"  WARNING: dropped {meta['n_dropped']} episodes on solver failure")


if __name__ == "__main__":
    main()

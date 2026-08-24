# scripts/train_clone.py
"""Train the deep-lcc clone on a generated dataset and save the checkpoint.

Usage:
    uv run python scripts/train_clone.py --data data/clone_dataset.npz \
        --out data/clone.pt --device auto
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from rl.clone import save_clone, train_clone


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/clone_dataset.npz")
    p.add_argument("--out", default="data/clone.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = np.load(args.data, allow_pickle=True)
    features = ds["features"]
    targets = ds["targets"]
    n_lib = int(ds["n_lib"])

    model, stats, history = train_clone(
        features, targets, n_lib=n_lib,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, seed=args.seed,
    )
    save_clone(args.out, model, stats)
    print(
        f"trained on {features.shape[0]} samples (dim {features.shape[1]}); "
        f"final val MSE {history['val_mse'][-1]:.5f} over "
        f"{len(history['val_mse'])} epochs -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

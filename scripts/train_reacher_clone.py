"""Train the behavioral clone of Select-DPC on Reacher.

`n_lib=0`: unlike the unicycle clone there is no library one-hot to protect from
standardization, because Select-DPC has no library index.

    uv run python scripts/train_reacher_clone.py
"""
from __future__ import annotations

import argparse

import numpy as np

from rl.clone import save_clone, train_clone


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/reacher_clone_dataset.npz")
    p.add_argument("--out", default="data/reacher_clone.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    with np.load(args.dataset) as z:
        features, actions = z["features"], z["actions"]
        # Read from the dataset, never a flag: these columns must be protected
        # from standardization and only the collector knows how many there are.
        n_lib = int(z["meta_n_lib"]) if "meta_n_lib" in z else 0
    print(f"{features.shape[0]} rows x {features.shape[1]} features, n_lib={n_lib}")

    model, stats, history = train_clone(
        features, actions, n_lib=n_lib, hidden=tuple(args.hidden),
        epochs=args.epochs, lr=args.lr, seed=args.seed, device=args.device)
    save_clone(args.out, model, stats)
    print(f"wrote {args.out}")
    print(f"  epochs run {len(history['val_mse'])}, "
          f"best val MSE {min(history['val_mse']):.5f}")
    # The clone's job is to reproduce the controller, so the number that matters
    # is error relative to the spread of what it is imitating -- an absolute MSE
    # is unreadable without knowing the action scale.
    print(f"  target std {actions.std(axis=0)}, "
          f"val RMSE / std = {np.sqrt(min(history['val_mse'])):.3f} "
          f"(standardized units)")


if __name__ == "__main__":
    main()

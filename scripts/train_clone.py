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
from two_wheel_robot.rl.deepc_setup import canonical_action_bounds
from two_wheel_robot.rl.features import egocentric


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/clone_dataset.npz")
    p.add_argument("--out", default="data/clone.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--squash", action="store_true",
                   help="train under SB3's SAC actor parameterization (rl/clone.py), on "
                        "targets rescaled to [-1, 1] over the --libraries action bounds: "
                        "the clone that initializes train_residual.py --warm")
    p.add_argument("--libraries", default="data/libraries_v0.npz")
    p.add_argument("--ego", action="store_true",
                   help="train on the robot-centred window (features.egocentric); pair "
                        "with train_residual.py --warm --ego-obs")
    args = p.parse_args()

    ds = np.load(args.data, allow_pickle=True)
    features = ds["features"]
    targets = ds["targets"]
    n_lib = int(ds["n_lib"])
    if args.ego:
        features = egocentric(features, int(ds["T_ini"]))
    if args.squash:
        b = canonical_action_bounds(args.libraries)
        targets = 2.0 * (targets - b[:, 0]) / (b[:, 1] - b[:, 0]) - 1.0

    model, stats, history = train_clone(
        features, targets, n_lib=n_lib,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, seed=args.seed, squash=args.squash,
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

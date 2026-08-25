# scripts/train_vanilla.py
"""Train the from-scratch (vanilla) TD3 baseline: no DeePC, no clone, same spaces."""
from __future__ import annotations

import argparse

from two_wheel_robot.rl.train_sb3 import train_vanilla


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--libraries", default="data/libraries_v0.npz",
                   help="only read for the canonical action bounds (spaces must match DeePC)")
    p.add_argument("--out", default="data/vanilla_td3.zip")
    p.add_argument("--algo", default="td3", choices=["td3", "sac"])
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--noise-sigma", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--monitor-out", default=None,
                   help="persist per-episode returns to <path>.monitor.csv "
                        "(feed to scripts/plot_training_return.py --monitor)")
    p.add_argument("--checkpoint-dir", default=None,
                   help="also snapshot the policy every --checkpoint-freq steps "
                        "(feed to scripts/sweep_checkpoints.py)")
    p.add_argument("--checkpoint-freq", type=int, default=25_000)
    p.add_argument("--wandb-project", default=None,
                   help="log this run to Weights & Biases (opt-in)")
    args = p.parse_args()

    train_vanilla(
        libraries_path=args.libraries, out_path=args.out, algo=args.algo,
        total_timesteps=args.timesteps, action_noise_sigma=args.noise_sigma,
        learning_rate=args.lr, device=args.device, seed=args.seed, verbose=1,
        monitor_path=args.monitor_out,
        checkpoint_dir=args.checkpoint_dir, checkpoint_freq=args.checkpoint_freq,
        wandb_project=args.wandb_project,
    )
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

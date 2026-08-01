# scripts/train_residual.py
"""Train the TD3 residual over the frozen DeePC clone (RL + MPC)."""
from __future__ import annotations

import argparse

from two_wheel_robot.rl.train_sb3 import train_residual


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clone", default="data/clone.pt")
    p.add_argument("--libraries", default="data/libraries_v0.npz")
    p.add_argument("--out", default="data/residual_td3.zip")
    p.add_argument("--algo", default="td3", choices=["td3", "sac"],
                   help="TD3 (default) or SAC (fallback for the hard collapse regime)")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--residual-frac", type=float, default=1.0)
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
    args = p.parse_args()

    train_residual(
        clone_path=args.clone, libraries_path=args.libraries, out_path=args.out,
        algo=args.algo, total_timesteps=args.timesteps, residual_frac=args.residual_frac,
        action_noise_sigma=args.noise_sigma, learning_rate=args.lr,
        device=args.device, seed=args.seed, verbose=1, monitor_path=args.monitor_out,
        checkpoint_dir=args.checkpoint_dir, checkpoint_freq=args.checkpoint_freq,
    )
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

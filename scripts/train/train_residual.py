# scripts/train/train_residual.py
"""Train the TD3 residual over the frozen DeePC clone (RL + MPC)."""
from __future__ import annotations

import argparse
import math

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
                        "(feed to scripts/plot/plot_training_return.py --monitor)")
    p.add_argument("--checkpoint-dir", default=None,
                   help="also snapshot the policy every --checkpoint-freq steps "
                        "(feed to scripts/sweep_checkpoints.py)")
    p.add_argument("--checkpoint-freq", type=int, default=25_000)
    p.add_argument("--wandb-project", default=None,
                   help="log this run to Weights & Biases (opt-in)")
    p.add_argument("--reward", default="deepc", choices=["deepc", "linear"],
                   help="env reward: DeePC-form stage cost (default) or linear distance "
                        "(TwoWheelGoal-v0 `reward=`)")
    p.add_argument("--warm", action="store_true",
                   help="Eq. 17 warm-start arm instead of the residual: SAC actor "
                        "initialized from --clone, which must be a squash clone "
                        "(train_clone.py --squash)")
    p.add_argument("--body-obs", action="store_true",
                   help="--warm: also observe the residual arm's normalized body-frame "
                        "obs; the actor's weights on it start at zero")
    p.add_argument("--std-obs", action="store_true",
                   help="--warm: features standardized by the clone's own mean/std")
    p.add_argument("--ego-obs", action="store_true",
                   help="--warm: robot-centred window (clone from train_clone.py --ego)")
    p.add_argument("--cold", action="store_true",
                   help="--warm's env and observation, but SB3's random actor init")
    p.add_argument("--init-log-std", type=float, default=math.log(0.1),
                   help="--warm: behaviour std at t=0 is exp(this)")
    p.add_argument("--warmup-steps", type=int, default=0, metavar="M",
                   help="DAWN data-anchored warmup (arXiv:2602.10539): M transitions of "
                        "the prior acting alone put in the replay buffer before learning. "
                        "Paper default M = 20000.")
    p.add_argument("--warmup-sigma", type=float, default=0.0)
    p.add_argument("--warmup-failures-only", action="store_true",
                   help="ablation: keep only the prior's FAILED warmup episodes (still "
                        "--warmup-steps transitions), so the buffer holds no successful finish")
    p.add_argument("--critic-layernorm", action="store_true",
                   help="DAWN critic normalization (rl/sb3.py::LayerNormCritic), SAC only")
    args = p.parse_args()
    if args.warm and args.algo != "sac":
        p.error("--warm needs --algo sac (init_actor_from_clone is SAC-only)")

    train_residual(
        clone_path=args.clone, libraries_path=args.libraries, out_path=args.out,
        algo=args.algo, total_timesteps=args.timesteps, residual_frac=args.residual_frac,
        action_noise_sigma=args.noise_sigma, learning_rate=args.lr,
        device=args.device, seed=args.seed, verbose=1, monitor_path=args.monitor_out,
        checkpoint_dir=args.checkpoint_dir, checkpoint_freq=args.checkpoint_freq,
        wandb_project=args.wandb_project, warm=args.warm, init_log_std=args.init_log_std,
        warmup_steps=args.warmup_steps, warmup_sigma=args.warmup_sigma,
        critic_layernorm=args.critic_layernorm, body_obs=args.body_obs, std_obs=args.std_obs,
        cold=args.cold,
        ego_obs=args.ego_obs, reward=args.reward,
        warmup_failures_only=args.warmup_failures_only,
    )
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

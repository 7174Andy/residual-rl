"""Train the zero-init RL residual over the frozen clone of the base controller.

SAC rather than TD3: journey 08 records TD3 needing the SAC fallback in the
hard-exploration regime, and this env's reward is dense, which suits SAC's
entropy exploration. The actor's mean head is zero-initialized so the policy
starts bit-for-bit identical to the clone -- the floor is "no worse than clone",
and `tests/test_reacher_residual_env.py` pins that invariant.

    uv run python scripts/train_reacher_residual.py --steps 200000
"""
from __future__ import annotations

import argparse
import os

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from reacher.residual_env import ResidualSelectEnv
from rl.sb3 import build_model, check_algo, ckpt_cb, zero_init_actor
from rl.wb import callbacks, finish, init_run, sb3_callback


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clone", default="data/dagger_clone_r3.pt")
    p.add_argument("--out", default="data/reacher_residual_dagger_200k.zip")
    p.add_argument("--algo", default="sac", choices=["sac", "td3"])
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--residual-frac", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--action-noise-sigma", type=float, default=0.1)
    p.add_argument("--monitor", default="data/reacher_residual")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--checkpoint-freq", type=int, default=25_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default=None,
                   help="log this run to Weights & Biases (opt-in)")
    args = p.parse_args()

    algo = check_algo(args.algo)
    run = init_run(args.wandb_project, name=os.path.basename(args.out),
                   config=vars(args), tags=["reacher", "residual", algo],
                   sync_tensorboard=True)

    def _factory():
        return Monitor(
            ResidualSelectEnv(clone_path=args.clone,
                              residual_frac=args.residual_frac, device=args.device),
            filename=args.monitor)

    venv = DummyVecEnv([_factory])
    model = build_model(algo, venv, args.lr, args.device, args.seed, 1,
                        args.action_noise_sigma,
                        tensorboard_log="data/tb" if run else None)
    zero_init_actor(model)
    try:
        model.learn(total_timesteps=args.steps, progress_bar=False,
                    callback=callbacks(
                        ckpt_cb(args.checkpoint_dir, args.checkpoint_freq),
                        sb3_callback(run, prefix="reacher")))
        model.save(args.out)
    finally:
        venv.close()
        finish(run)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

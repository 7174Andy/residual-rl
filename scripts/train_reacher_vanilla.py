"""Train the from-scratch RL control: same env, same budget, no clone anywhere.

Journey 09 measured vanilla TD3 at 78/78 against the residual's 74/78 on the
unicycle (McNemar p = 0.13) -- the residual's edge was NOT significant, and that
was only discovered because this control was run. Without it here, a good residual
result cannot be distinguished from "SAC solves 2-DoF reaching".

No zero-init: a from-scratch actor has no baseline to stay close to.

    uv run python scripts/train_reacher_vanilla.py --steps 200000
"""
from __future__ import annotations

import argparse

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import reacher  # noqa: F401  registers the Gym ID
from rl.sb3 import build_model, check_algo, ckpt_cb


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/reacher_vanilla_sac.zip")
    p.add_argument("--algo", default="sac", choices=["sac", "td3"])
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--action-noise-sigma", type=float, default=0.1)
    p.add_argument("--monitor", default="data/reacher_vanilla")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--checkpoint-freq", type=int, default=25_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    algo = check_algo(args.algo)
    venv = DummyVecEnv([lambda: Monitor(gym.make("ReacherGoal-v0"),
                                        filename=args.monitor)])
    model = build_model(algo, venv, args.lr, args.device, args.seed, 1,
                        args.action_noise_sigma)
    try:
        model.learn(total_timesteps=args.steps, progress_bar=False,
                    callback=ckpt_cb(args.checkpoint_dir, args.checkpoint_freq))
        model.save(args.out)
    finally:
        venv.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

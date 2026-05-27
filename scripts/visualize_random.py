"""Visualize TwoWheelGoal-v0 with a random policy.

Usage:
    uv run python scripts/visualize_random.py
    uv run python scripts/visualize_random.py --episodes 5 --seed 42
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

import two_wheel_robot.env  # noqa: F401  (side-effect: registers Gym ID)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make("TwoWheelGoal-v0", render_mode="human", action_bounds=((10.0, 20.0), (-np.pi / 6, np.pi / 6)))
    try:
        for ep in range(args.episodes):
            _, info = env.reset(seed=args.seed + ep)
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = env.action_space.sample()
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
            outcome = "REACHED" if terminated else "truncated"
            print(
                f"episode {ep}: {outcome:9s} after {steps:3d} steps "
                f"return={total_reward:+.1f} final_dist={info['distance']:.2f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()

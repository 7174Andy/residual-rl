"""Solvability check for QuadrupedGoal-v0: random vs holonomic P-control.

    .venv/bin/python scripts/check_quadruped_env.py [--seeds 20 --policy both]
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

import quadruped  # noqa: F401  (registers QuadrupedGoal-v0)


def pcontrol(obs, lo, hi):
    d, sb, cb = obs[:3]
    b = np.arctan2(sb, cb)
    return np.clip([d * cb, d * sb, 2.0 * b], lo, hi).astype(np.float32)


def run(policy: str, seeds: int) -> str:
    env = gym.make("QuadrupedGoal-v0")
    reached, steps, falls, trunc = 0, [], 0, 0
    for s in range(seeds):
        obs, _ = env.reset(seed=s)
        env.action_space.seed(s)
        while True:
            a = env.action_space.sample() if policy == "random" else pcontrol(obs, env.action_space.low, env.action_space.high)
            obs, _, term, tr, info = env.step(a)
            if term or tr:
                break
        if info["reached"]:
            reached += 1
            steps.append(info["step_idx"])
        falls += bool(info["fell"])
        trunc += bool(tr and not term)
    mean = f"{np.mean(steps):.1f} (min {min(steps)}, max {max(steps)})" if steps else "n/a"
    return (f"{policy}: reached {reached}/{seeds}, mean steps-to-reach {mean}, "
            f"falls {falls}, truncations {trunc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--policy", choices=["random", "pcontrol", "both"], default="both")
    a = ap.parse_args()
    for p in ("random", "pcontrol") if a.policy == "both" else (a.policy,):
        print(run(p, a.seeds))

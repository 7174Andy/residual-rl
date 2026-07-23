#!/usr/bin/env python
"""Generate the committed per-seed traces behind the seed-showcase figures.

Writes full per-step (x, y, heading, v, w) closed-loop traces for one or more
showcase seeds, for both clone and residual, to
`<outdir>/traj_<seed>_{clone,residual}.csv`, for `scripts/plot_seed_traces.py`.
Both closed loops are DeePC-free (clone/residual only -- no QP), so this runs
in well under a second per seed. Defaults to `--trace-model
data/residual_td3.zip` (the 200k checkpoint) to match the checkpoint used to
record that seed's embedded video in docs/journey/10-residual-rl.md.
"""
from __future__ import annotations

import argparse
import csv
from typing import cast

import gymnasium as gym
import numpy as np

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from two_wheel_robot.env.env import UnicycleGoalEnv
from two_wheel_robot.rl.clone import load_clone
from two_wheel_robot.rl.clone_eval import run_clone_closed_loop_with_actions
from two_wheel_robot.rl.deepc_setup import build_canonical_deepc
from two_wheel_robot.rl.residual_env import ResidualDeePCEnv
from two_wheel_robot.rl.residual_eval import run_residual_closed_loop_with_actions
from two_wheel_robot.rl.train_sb3 import load_residual


def _write_trace(path: str, traj: np.ndarray, actions: np.ndarray, goal: np.ndarray) -> None:
    """One row per step: (x, y, heading, v, w, goal_x, goal_y).

    Step 0 is the post-reset state, before any action -- `v, w = 0, 0` there,
    matching the env's own "last_action is zero after reset" convention.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "x", "y", "heading", "v", "w", "goal_x", "goal_y"])
        writer.writerow([0, traj[0, 0], traj[0, 1], traj[0, 2], 0.0, 0.0, goal[0], goal[1]])
        for t in range(len(actions)):
            writer.writerow([t + 1, traj[t + 1, 0], traj[t + 1, 1], traj[t + 1, 2],
                              actions[t, 0], actions[t, 1], goal[0], goal[1]])
    print(f"wrote {path}  ({len(actions)} steps)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clone", default="data/clone.pt")
    p.add_argument("--libraries", default="data/libraries_v0.npz")
    p.add_argument("--outdir", default="docs/journey/figures")
    p.add_argument("--device", default="cpu")

    p.add_argument("--trace-model", default="data/residual_td3.zip")
    p.add_argument("--trace-seeds", default="4104626029,4104626034")
    args = p.parse_args()

    _, info = build_canonical_deepc(libraries_path=args.libraries)
    predictor = load_clone(args.clone, device=args.device)
    action_bounds = info["action_bounds"]

    trace_model = load_residual(args.trace_model, algo="td3", device=args.device)
    trace_env = ResidualDeePCEnv(
        clone_path=args.clone, libraries_path=args.libraries, device=args.device,
    )
    env_c = gym.make("TwoWheelGoal-v0", action_bounds=action_bounds)
    try:
        for seed in (int(s) for s in args.trace_seeds.split(",")):
            _, traj_c, act_c = run_clone_closed_loop_with_actions(predictor, info, env_c, seed)
            goal_c = cast(UnicycleGoalEnv, env_c.unwrapped).goal
            _write_trace(f"{args.outdir}/traj_{seed}_clone.csv", traj_c, act_c, goal_c)

            _, traj_r, act_r = run_residual_closed_loop_with_actions(trace_model, trace_env, seed)
            goal_r = trace_env.base.goal
            _write_trace(f"{args.outdir}/traj_{seed}_residual.csv", traj_r, act_r, goal_r)
    finally:
        env_c.close()
        trace_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Per-seed closed-loop trace cache: trace_io's CSVs, regenerated on demand.

Used by scripts/render_dashboard_video.py so a new seed needs one command,
not a separate scripts/eval_seed_showcase.py pass first. Kept separate from
two_wheel_robot.rl.trace_io (pure CSV I/O, no gym/torch/sb3) so
scripts/plot_seed_traces.py can read traces without paying for those imports.
"""
from __future__ import annotations

import os
from typing import cast

import gymnasium as gym

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from two_wheel_robot.env.env import UnicycleGoalEnv
from two_wheel_robot.rl.clone import load_clone
from two_wheel_robot.rl.clone_eval import run_clone_closed_loop_with_actions
from two_wheel_robot.rl.deepc_setup import build_canonical_deepc
from two_wheel_robot.rl.residual_env import ResidualDeePCEnv
from two_wheel_robot.rl.residual_eval import run_residual_closed_loop_with_actions
from two_wheel_robot.rl.train_sb3 import load_residual
from two_wheel_robot.rl.trace_io import (
    clone_trace_path,
    read_trace,
    residual_trace_path,
    write_trace,
)


def generate_trace_pair(
    seed: int,
    figdir: str,
    clone_path: str = "data/clone.pt",
    residual_model_path: str = "data/residual_td3.zip",
    algo: str = "td3",
    libraries_path: str = "data/libraries_v0.npz",
    device: str = "cpu",
) -> tuple[dict, dict]:
    """Run both closed loops for `seed`, write their trace CSVs, and return the
    trace dicts. Always regenerates -- see `ensure_traces` for a
    cache-checking wrapper.
    """
    os.makedirs(figdir, exist_ok=True)
    clone_csv = clone_trace_path(figdir, seed)
    residual_csv = residual_trace_path(figdir, seed)

    _, info = build_canonical_deepc(libraries_path=libraries_path)
    predictor = load_clone(clone_path, device=device)
    action_bounds = info["action_bounds"]

    env_c = gym.make("TwoWheelGoal-v0", action_bounds=action_bounds)
    try:
        _, traj_c, act_c = run_clone_closed_loop_with_actions(predictor, info, env_c, seed)
        goal_c = cast(UnicycleGoalEnv, env_c.unwrapped).goal
    finally:
        env_c.close()
    write_trace(clone_csv, traj_c, act_c, goal_c)

    model = load_residual(residual_model_path, algo=algo, device=device)
    res_env = ResidualDeePCEnv(
        clone_path=clone_path, libraries_path=libraries_path, device=device,
    )
    try:
        _, traj_r, act_r = run_residual_closed_loop_with_actions(model, res_env, seed)
        goal_r = res_env.base.goal
    finally:
        res_env.close()
    write_trace(residual_csv, traj_r, act_r, goal_r)

    return read_trace(clone_csv), read_trace(residual_csv)


def ensure_traces(
    seed: int,
    figdir: str,
    clone_path: str = "data/clone.pt",
    residual_model_path: str = "data/residual_td3.zip",
    algo: str = "td3",
    libraries_path: str = "data/libraries_v0.npz",
    device: str = "cpu",
) -> tuple[dict, dict]:
    """Return `(clone_trace, residual_trace)`, generating + caching CSVs if missing.

    Cache hit (both CSVs already on disk) never touches `clone_path` /
    `residual_model_path` / `libraries_path` -- neither loaded nor validated
    to exist.
    """
    clone_csv = clone_trace_path(figdir, seed)
    residual_csv = residual_trace_path(figdir, seed)
    if os.path.exists(clone_csv) and os.path.exists(residual_csv):
        return read_trace(clone_csv), read_trace(residual_csv)
    return generate_trace_pair(
        seed, figdir, clone_path, residual_model_path, algo, libraries_path, device,
    )

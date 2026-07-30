"""Pure CSV read/write for closed-loop trace files -- no gym/torch/sb3 deps.

Schema: `traj_<seed>_{clone,residual}.csv`, columns `step, x, y, heading, v,
w, goal_x, goal_y`. Kept gym/torch-free so lightweight consumers
(`scripts/plot_seed_traces.py`) don't pay for `stable_baselines3`/`torch`
imports just to read a trace someone else already generated.
"""
from __future__ import annotations

import csv
import os

import numpy as np


def clone_trace_path(figdir: str, seed: int) -> str:
    return os.path.join(figdir, f"traj_{seed}_clone.csv")


def residual_trace_path(figdir: str, seed: int) -> str:
    return os.path.join(figdir, f"traj_{seed}_residual.csv")


def write_trace(path: str, traj: np.ndarray, actions: np.ndarray, goal: np.ndarray) -> None:
    """One row per step: (x, y, heading, v, w, goal_x, goal_y).

    Row 0 is the post-reset state, before any action -- v, w = 0 there,
    matching the env's own "last_action is zero after reset" convention.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "x", "y", "heading", "v", "w", "goal_x", "goal_y"])
        writer.writerow([0, traj[0, 0], traj[0, 1], traj[0, 2], 0.0, 0.0, goal[0], goal[1]])
        for t in range(len(actions)):
            writer.writerow([t + 1, traj[t + 1, 0], traj[t + 1, 1], traj[t + 1, 2],
                              actions[t, 0], actions[t, 1], goal[0], goal[1]])


def read_trace(path: str) -> dict:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {
        "step": np.array([int(r["step"]) for r in rows]),
        "x": np.array([float(r["x"]) for r in rows]),
        "y": np.array([float(r["y"]) for r in rows]),
        "heading": np.array([float(r["heading"]) for r in rows]),
        "v": np.array([float(r["v"]) for r in rows]),
        "w": np.array([float(r["w"]) for r in rows]),
        "goal": np.array([float(rows[0]["goal_x"]), float(rows[0]["goal_y"])]),
    }

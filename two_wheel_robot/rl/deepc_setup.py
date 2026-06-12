# two_wheel_robot/rl/deepc_setup.py
"""The canonical deep-lcc (DeePC) configuration the clone imitates.

This is the measured ~39%-reach baseline from `scripts/run_deepc.py` defaults:
4 orientation-keyed libraries, bearing heading reference, Q heading weight 2,
T_ini=5, N=12, lambda_g=2, lambda_y=3e6, SCS. Centralized here so the clone
dataset, the fidelity gate, and the later residual env can't drift from it.
"""
from __future__ import annotations

from typing import cast

import gymnasium as gym
import numpy as np

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from two_wheel_robot.controllers.data_collection import (
    PAPER_INIT_HEADINGS,
    PAPER_SAMPLE_BOUNDS,
)
from two_wheel_robot.controllers.deepc import DeePC
from two_wheel_robot.controllers.hankel import build_hankel
from two_wheel_robot.env.dynamics import wrap_to_pi
from two_wheel_robot.env.env import UnicycleGoalEnv

DEFAULT_LIBRARIES = "data/libraries_v0.npz"


def bearing_y_ref(state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Per-step DeePC reference `(g_x, g_y, bearing_to_goal)`.

    `bearing = atan2(g_y - y, g_x - x)` — the heading that points the robot at
    the goal from its current position. Matches `run_deepc.py`'s default.
    """
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    bearing = float(np.arctan2(goal[1] - state[1], goal[0] - state[0]))
    return np.array([goal[0], goal[1], bearing], dtype=np.float64)


def build_canonical_deepc(
    libraries_path: str = DEFAULT_LIBRARIES,
    T_ini: int = 5,
    N: int = 12,
    lambda_g: float = 2.0,
    lambda_y: float = 3e6,
) -> tuple[DeePC, dict]:
    """Construct the canonical deep-lcc controller + the info needed to run it.

    `libraries_path` is resolved relative to the current working directory
    (default assumes the repo root, as the CLIs are run from there).

    Returns `(deepc, info)` where info has `anchors`, `u_init_midpoint`,
    `action_bounds`, `Q`, `R`, `T_ini`, `N`, `lambda_g`, `lambda_y`.
    """
    # Read everything out of the npz inside the context manager so the file
    # descriptor is released promptly (NpzFile lazily loads on access).
    with np.load(libraries_path) as data:
        if "sample_bounds" in data.files:
            sample_bounds = np.asarray(data["sample_bounds"], dtype=np.float64)
        else:
            sample_bounds = PAPER_SAMPLE_BOUNDS
        uy = [(data[f"u_{i}"], data[f"y_{i}"]) for i in range(4)]

    env = gym.make("TwoWheelGoal-v0", action_bounds=sample_bounds)
    base = cast(UnicycleGoalEnv, env.unwrapped)
    # Pin the canonical Q heading weight to 2 explicitly. This currently equals
    # the env default (diag(1, 1, 2)); the explicit set keeps the canonical
    # config fixed even if that env default ever changes.
    Q = base.Q.copy()
    Q[2, 2] = 2.0
    R = base.R.copy()
    action_bounds = base.action_bounds.copy()
    env.close()

    u_bounds = (action_bounds[:, 0], action_bounds[:, 1])
    anchors = np.array([float(wrap_to_pi(h)) for h in PAPER_INIT_HEADINGS],
                       dtype=np.float64)
    libraries = [build_hankel(u, y, T_ini=T_ini, N=N) for (u, y) in uy]
    deepc = DeePC(
        libraries, anchor_headings=anchors, Q=Q, R=R, T_ini=T_ini, N=N,
        lambda_g=lambda_g, lambda_y=lambda_y, u_bounds=u_bounds,
    )
    info = {
        "anchors": anchors,
        "u_init_midpoint": 0.5 * (action_bounds[:, 0] + action_bounds[:, 1]),
        "action_bounds": action_bounds,
        "Q": Q,
        "R": R,
        "T_ini": T_ini,
        "N": N,
        "lambda_g": lambda_g,
        "lambda_y": lambda_y,
    }
    return deepc, info

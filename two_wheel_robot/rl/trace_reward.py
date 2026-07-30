"""Recompute per-step DeePC-form reward from a closed-loop CSV trace.

Given the `(x, y, heading, v, w)` columns `scripts/eval_seed_showcase.py`
already writes per seed, reward is exactly recoverable with no model rerun:
`v, w` are the applied (post-clip) actions and `x, y, heading` are the
post-step state -- exactly what `env.py::step()` uses to compute reward. Row
0 is the post-reset state, before any action was taken, so it carries no
reward (excluded from the cumulative sum).
"""
from __future__ import annotations

import numpy as np

# two_wheel_robot/env/env.py's defaults, duplicated (not imported) so this
# module stays gym-free and numpy-only.
DEFAULT_Q = np.diag([1.0, 1.0, 2.0])
DEFAULT_R = 1.3e-3 * np.eye(2)
DEFAULT_REACH_BONUS = 100.0
DEFAULT_GOAL_TOLERANCE = 0.5


def recompute_reward(
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    goal: np.ndarray,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    reach_bonus: float = DEFAULT_REACH_BONUS,
    goal_tolerance: float = DEFAULT_GOAL_TOLERANCE,
) -> dict:
    """Recompute reward/distance/reached for every row of a closed-loop trace.

    `x, y, heading, v, w` are 1-D arrays of equal length `T + 1` -- row 0 is
    the post-reset state (`v[0] == w[0] == 0`, no action applied yet), rows
    `1..T` are the state *after* applying `(v[i], w[i])`, matching the CSV
    schema `eval_seed_showcase.py::_write_trace` writes.

    Returns a dict of length-`T + 1` arrays: `reward` (row 0 forced to `0.0`
    -- no transition happened yet), `cum_reward` (`np.cumsum(reward)`, so
    also `0.0` at row 0), `dist` (`‖(x, y) - goal‖`), `reached`
    (`dist < goal_tolerance`).
    """
    Q = DEFAULT_Q if Q is None else np.asarray(Q, dtype=np.float64)
    R = DEFAULT_R if R is None else np.asarray(R, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    heading = np.asarray(heading, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64).reshape(2)

    dist = np.hypot(x - goal[0], y - goal[1])
    reached = dist < goal_tolerance

    y_ref = np.array([goal[0], goal[1], 0.0])
    err = np.stack([x, y, heading], axis=1) - y_ref
    u = np.stack([v, w], axis=1)
    cost = np.einsum("ti,ij,tj->t", err, Q, err) + np.einsum("ti,ij,tj->t", u, R, u)
    reward = -cost + np.where(reached, reach_bonus, 0.0)
    reward[0] = 0.0  # row 0 is the post-reset state -- no transition yet

    cum_reward = np.cumsum(reward)
    return {"reward": reward, "cum_reward": cum_reward, "dist": dist, "reached": reached}

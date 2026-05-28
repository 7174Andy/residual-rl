"""Offline trajectory collection for data-driven predictive control (DeePC).

A "trajectory" is a sequence of aligned `(u_t, y_t)` pairs of equal length T:
- `u_t` is the control input applied at time t (post-clip),
- `y_t` is the env output observed at time t *before* `u_t` is applied.

So `y_{t+1}` is the system response to `u_t` from `y_t`. Hankel matrices for
DeePC (in `controllers/hankel.py`) consume these arrays.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence, cast

import gymnasium as gym
import numpy as np

from two_wheel_robot.env.env import UnicycleGoalEnv


# ---- Paper (arXiv:2603.07395 Appendix D) data-collection settings -----------

PAPER_SAMPLE_BOUNDS: np.ndarray = np.array(
    [[10.0, 20.0], [-np.pi / 6, np.pi / 6]], dtype=np.float64
)
PAPER_INIT_HEADINGS: tuple[float, ...] = (
    np.pi / 4,
    3 * np.pi / 4,
    5 * np.pi / 4,
    7 * np.pi / 4,
)
PAPER_T: int = 1500


def paper_init_states(origin: tuple[float, float] = (0.0, 0.0)) -> list[np.ndarray]:
    """Paper's 4 initial states: same `origin`, headings from PAPER_INIT_HEADINGS."""
    ox, oy = origin
    return [np.array([ox, oy, h], dtype=np.float64) for h in PAPER_INIT_HEADINGS]


# ---- Core collection ---------------------------------------------------------


def collect_trajectory(
    env: gym.Env,
    T: int,
    *,
    init_state: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    sample_bounds: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out `env` for T steps with uniform PE inputs. Returns aligned (u, y).

    Args:
        env: gymnasium env wrapping `UnicycleGoalEnv`. Will be `reset()` first.
        T: trajectory length, >= 1.
        init_state: starting pose `(x, y, delta)`. If None, env samples its default.
        rng: numpy `Generator` for action sampling. Default: fresh, unseeded.
        sample_bounds: `((v_min, v_max), (w_min, w_max))` for uniform PE inputs.
            If None, uses `env.unwrapped.action_bounds`.

    Returns:
        u_traj: shape `(T, 2)`, float64. Actual (post-clip) actions applied.
        y_traj: shape `(T, 3)`, float64. `y_traj[t]` is the output observed
            *before* `u_traj[t]` is applied.
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if rng is None:
        rng = np.random.default_rng()

    options: dict[str, Any] = {
        # Goal far outside workspace so the env never terminates mid-collection.
        # Only (u, y) matter here.
        "goal": np.array([100.0, 100.0], dtype=np.float64),
    }
    if init_state is not None:
        options["state"] = np.asarray(init_state, dtype=np.float64).reshape(3)

    # The far goal pushes `distance` outside `observation_space.high`. Gym's
    # passive env-checker emits a UserWarning every reset/step in this state.
    # That's expected here (we don't consume obs during collection), so silence it.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*not within the observation space.*",
            category=UserWarning,
        )
        env.reset(options=options)
        base = cast(UnicycleGoalEnv, env.unwrapped)

        if sample_bounds is None:
            sample_bounds = base.action_bounds
        sample_bounds = np.asarray(sample_bounds, dtype=np.float64).reshape(2, 2)
        low = sample_bounds[:, 0]
        high = sample_bounds[:, 1]

        u_traj = np.zeros((T, 2), dtype=np.float64)
        y_traj = np.zeros((T, 3), dtype=np.float64)

        for t in range(T):
            y_traj[t] = base.y  # observed before applying u_t
            u_sample = rng.uniform(low=low, high=high)
            env.step(u_sample)
            # Record post-clip action so (u, y) reflects what dynamics actually saw.
            u_traj[t] = base.last_action

    return u_traj, y_traj


def collect_libraries(
    env: gym.Env,
    T: int,
    init_states: Sequence[np.ndarray],
    *,
    rng: Optional[np.random.Generator] = None,
    sample_bounds: Optional[np.ndarray] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One trajectory per `init_state`. Shares one rng across all trajectories.

    Returns a list of `(u_traj, y_traj)` tuples, parallel to `init_states`.
    """
    if rng is None:
        rng = np.random.default_rng()
    return [
        collect_trajectory(
            env, T, init_state=s, rng=rng, sample_bounds=sample_bounds
        )
        for s in init_states
    ]

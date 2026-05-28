"""Tests for two_wheel_robot.controllers.data_collection."""

from __future__ import annotations

from typing import cast

import gymnasium as gym
import numpy as np
import pytest

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from two_wheel_robot.controllers.data_collection import (
    PAPER_INIT_HEADINGS,
    PAPER_SAMPLE_BOUNDS,
    collect_libraries,
    collect_trajectory,
    paper_init_states,
)
from two_wheel_robot.env.dynamics import step_unicycle, wrap_to_pi
from two_wheel_robot.env.env import UnicycleGoalEnv


def _env() -> gym.Env:
    return gym.make("TwoWheelGoal-v0")


def _base(env: gym.Env) -> UnicycleGoalEnv:
    return cast(UnicycleGoalEnv, env.unwrapped)


# ---- collect_trajectory ------------------------------------------------------


def test_returns_correct_shapes():
    u, y = collect_trajectory(_env(), T=50, rng=np.random.default_rng(0))
    assert u.shape == (50, 2)
    assert y.shape == (50, 3)
    assert u.dtype == np.float64
    assert y.dtype == np.float64


def test_rejects_zero_T():
    with pytest.raises(ValueError):
        collect_trajectory(_env(), T=0, rng=np.random.default_rng(0))


def test_actions_within_paper_sample_bounds():
    u, _ = collect_trajectory(
        _env(),
        T=300,
        rng=np.random.default_rng(0),
        sample_bounds=PAPER_SAMPLE_BOUNDS,
    )
    assert (u[:, 0] >= 10.0 - 1e-9).all()
    assert (u[:, 0] <= 20.0 + 1e-9).all()
    assert (u[:, 1] >= -np.pi / 6 - 1e-9).all()
    assert (u[:, 1] <= np.pi / 6 + 1e-9).all()


def test_init_state_is_first_y():
    init = np.array([1.0, 2.0, 0.3])
    _, y = collect_trajectory(
        _env(), T=5, rng=np.random.default_rng(0), init_state=init
    )
    np.testing.assert_allclose(y[0], init)


def test_y_follows_dynamics_with_wall_clip_and_wrap():
    """y[t+1] == wrap_heading(wall_clip(step_unicycle(y[t], u[t], dt)))."""
    env = _env()
    base = _base(env)
    dt = base.dt
    bounds = base.workspace_bounds
    u, y = collect_trajectory(
        env,
        T=40,
        rng=np.random.default_rng(123),
        init_state=np.array([0.0, 0.0, 0.0]),
    )
    for t in range(len(u) - 1):
        expected = step_unicycle(y[t], u[t], dt)
        expected[0] = np.clip(expected[0], bounds[0, 0], bounds[0, 1])
        expected[1] = np.clip(expected[1], bounds[1, 0], bounds[1, 1])
        expected[2] = float(wrap_to_pi(expected[2]))
        np.testing.assert_allclose(y[t + 1], expected, atol=1e-9)


def test_seed_reproducibility():
    u1, y1 = collect_trajectory(
        _env(),
        T=20,
        rng=np.random.default_rng(42),
        init_state=np.array([0.0, 0.0, 0.0]),
    )
    u2, y2 = collect_trajectory(
        _env(),
        T=20,
        rng=np.random.default_rng(42),
        init_state=np.array([0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(u1, u2)
    np.testing.assert_array_equal(y1, y2)


def test_unwrapped_init_heading_is_normalized():
    """Caller passing heading outside [-pi, pi] gets a wrapped y[0]."""
    init = np.array([0.0, 0.0, 5 * np.pi / 4])  # > pi
    _, y = collect_trajectory(
        _env(), T=2, rng=np.random.default_rng(0), init_state=init
    )
    # Wrapping: 5pi/4 -> -3pi/4
    assert -np.pi - 1e-12 <= y[0, 2] <= np.pi + 1e-12
    assert y[0, 2] == pytest.approx(-3 * np.pi / 4)


# ---- collect_libraries -------------------------------------------------------


def test_libraries_one_per_init_state():
    libs = collect_libraries(
        _env(),
        T=10,
        init_states=paper_init_states(),
        rng=np.random.default_rng(0),
    )
    assert len(libs) == 4
    for u, y in libs:
        assert u.shape == (10, 2)
        assert y.shape == (10, 3)


def test_libraries_different_per_init_state():
    """Different initial orientations should produce different y trajectories."""
    libs = collect_libraries(
        _env(),
        T=20,
        init_states=paper_init_states(),
        rng=np.random.default_rng(0),
    )
    # Compare first vs others — heading distinct → trajectories must diverge.
    y0 = libs[0][1]
    for i in range(1, 4):
        assert not np.allclose(y0, libs[i][1])


# ---- paper_init_states -------------------------------------------------------


def test_paper_init_states_match_paper():
    states = paper_init_states()
    assert len(states) == 4
    for s, h in zip(states, PAPER_INIT_HEADINGS):
        np.testing.assert_allclose(s, [0.0, 0.0, h])

"""Tests for two_wheel_robot.rl.wrappers."""

from __future__ import annotations

import warnings
from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from two_wheel_robot.env.env import UnicycleGoalEnv
from two_wheel_robot.rl.wrappers import rescale_action_symmetric


def _make_wrapped() -> gym.Env:
    return rescale_action_symmetric(gym.make("TwoWheelGoal-v0"))


def _base(env: gym.Env) -> UnicycleGoalEnv:
    return cast(UnicycleGoalEnv, env.unwrapped)


def test_wrapped_action_space_is_symmetric_unit():
    env = _make_wrapped()
    space = cast(Box, env.action_space)
    assert np.allclose(space.low, [-1.0, -1.0])
    assert np.allclose(space.high, [1.0, 1.0])


def test_action_plus_one_maps_to_native_high():
    env = _make_wrapped()
    env.reset(seed=0, options={"state": [0.0, 0.0, 0.0], "goal": [9.0, 9.0]})
    env.step(np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(_base(env).last_action, [20.0, np.pi / 2], atol=1e-6)


def test_action_minus_one_maps_to_native_low():
    env = _make_wrapped()
    env.reset(seed=0, options={"state": [0.0, 0.0, 0.0], "goal": [9.0, 9.0]})
    env.step(np.array([-1.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(_base(env).last_action, [0.0, -np.pi / 2], atol=1e-6)


def test_action_zero_maps_to_native_midpoint():
    env = _make_wrapped()
    env.reset(seed=0, options={"state": [0.0, 0.0, 0.0], "goal": [9.0, 9.0]})
    env.step(np.array([0.0, 0.0], dtype=np.float32))
    # Midpoint of v in [0, 20] = 10; midpoint of w in [-pi/2, pi/2] = 0.
    np.testing.assert_allclose(_base(env).last_action, [10.0, 0.0], atol=1e-6)


def test_wrapped_env_checker_emits_no_symmetric_warning():
    env = _make_wrapped()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env, skip_render_check=True)
    flagged = [
        w
        for w in caught
        if "symmetric" in str(w.message).lower()
        or "normalized" in str(w.message).lower()
    ]
    assert not flagged, [str(w.message) for w in flagged]

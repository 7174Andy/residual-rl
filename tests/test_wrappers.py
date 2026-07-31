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
from two_wheel_robot.rl.wrappers import rescale_action_symmetric, vanilla_rl_env


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


def test_vanilla_env_action_space_is_the_deepc_u_bounds():
    """The vanilla agent emits u = (v, w) in physical units over DeePC's own box."""
    from two_wheel_robot.rl.deepc_setup import canonical_action_bounds

    bounds = canonical_action_bounds()
    env = vanilla_rl_env(bounds)
    try:
        space = cast(Box, env.action_space)
        np.testing.assert_allclose(space.low, bounds[:, 0])   # v = 0,  w = -pi/2
        np.testing.assert_allclose(space.high, bounds[:, 1])  # v = 20, w = +pi/2
        # A physical-unit command passes through untransformed.
        env.reset(seed=0)
        env.step(np.array([12.5, 0.4], dtype=np.float32))
        np.testing.assert_allclose(_base(env).last_action, [12.5, 0.4], atol=1e-6)
    finally:
        env.close()


def test_vanilla_env_observation_matches_the_residual_policy_view():
    """Obs is the residual actor's normalized body obs, minus its u_base block."""
    from two_wheel_robot.rl.residual_env import ResidualDeePCEnv

    res = ResidualDeePCEnv(include_base_in_obs=False)
    try:
        env = vanilla_rl_env(res.action_bounds)
        try:
            obs, _ = env.reset(seed=0)
            assert env.observation_space == res.observation_space
            assert res.observation_space.contains(obs)
        finally:
            env.close()
    finally:
        res.close()


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

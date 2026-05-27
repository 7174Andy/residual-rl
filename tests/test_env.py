"""Tests for UnicycleGoalEnv (TwoWheelGoal-v0)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from two_wheel_robot.env.env import UnicycleGoalEnv


# ---- Gymnasium API conformance ------------------------------------------------


def test_env_checker_default_config():
    env = UnicycleGoalEnv()
    check_env(env, skip_render_check=True)


def test_env_checker_custom_config():
    env = UnicycleGoalEnv(
        workspace_bounds=((-5.0, 5.0), (-5.0, 5.0)),
        action_bounds=((-2.0, 2.0), (-1.0, 1.0)),
        goal_tolerance=0.2,
        min_start_goal_dist=1.0,
        max_steps=50,
        reach_bonus=10.0,
    )
    check_env(env, skip_render_check=True)


def test_registered_gym_id_makeable():
    env = gym.make("TwoWheelGoal-v0")
    obs, info = env.reset(seed=0)
    assert env.action_space.contains(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert "distance" in info
    env.close()


# ---- Reset semantics ----------------------------------------------------------


def test_reset_obs_in_observation_space():
    env = UnicycleGoalEnv()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_reset_last_action_is_zero():
    env = UnicycleGoalEnv()
    env.reset(seed=0)
    np.testing.assert_array_equal(env.last_action, np.zeros(2))


def test_reset_state_and_goal_inside_workspace():
    env = UnicycleGoalEnv()
    for seed in range(20):
        env.reset(seed=seed)
        (xmin, xmax), (ymin, ymax) = env.workspace_bounds
        assert xmin <= env.state[0] <= xmax
        assert ymin <= env.state[1] <= ymax
        assert -np.pi <= env.state[2] <= np.pi
        assert xmin <= env.goal[0] <= xmax
        assert ymin <= env.goal[1] <= ymax


def test_reset_respects_min_start_goal_distance():
    env = UnicycleGoalEnv(min_start_goal_dist=5.0)
    for seed in range(20):
        env.reset(seed=seed)
        dist = float(np.linalg.norm(env.state[:2] - env.goal))
        assert dist >= 5.0 - 1e-9, f"seed={seed}: dist={dist}"


def test_reset_seed_reproducibility():
    env1 = UnicycleGoalEnv()
    env2 = UnicycleGoalEnv()
    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs2)
    np.testing.assert_array_equal(env1.state, env2.state)
    np.testing.assert_array_equal(env1.goal, env2.goal)


def test_reset_options_override_state_and_goal():
    env = UnicycleGoalEnv()
    env.reset(seed=0, options={"state": [1.0, 2.0, 0.5], "goal": [3.0, 4.0]})
    np.testing.assert_array_equal(env.state, [1.0, 2.0, 0.5])
    np.testing.assert_array_equal(env.goal, [3.0, 4.0])


# ---- Step semantics -----------------------------------------------------------


def test_step_returns_five_tuple():
    env = UnicycleGoalEnv()
    env.reset(seed=0)
    result = env.step(env.action_space.sample())
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_clips_action_to_bounds():
    env = UnicycleGoalEnv(action_bounds=((0.0, 5.0), (-1.0, 1.0)))
    env.reset(seed=0, options={"state": [0.0, 0.0, 0.0], "goal": [9.0, 9.0]})
    env.step(np.array([100.0, 100.0]))
    np.testing.assert_array_equal(env.last_action, [5.0, 1.0])

    env.reset(seed=0, options={"state": [0.0, 0.0, 0.0], "goal": [9.0, 9.0]})
    env.step(np.array([-100.0, -100.0]))
    np.testing.assert_array_equal(env.last_action, [0.0, -1.0])


def test_step_advances_step_idx():
    env = UnicycleGoalEnv()
    env.reset(seed=0)
    assert env.step_idx == 0
    env.step(env.action_space.sample())
    assert env.step_idx == 1


# ---- Termination and truncation ----------------------------------------------


def test_terminates_when_within_tolerance():
    env = UnicycleGoalEnv(goal_tolerance=0.5)
    # Place robot exactly at the goal: any action keeps it within tolerance.
    env.reset(seed=0, options={"state": [0.0, 0.0, 0.0], "goal": [0.0, 0.0]})
    _, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
    assert terminated is True
    assert truncated is False
    assert info["reached"] is True
    # Reach bonus added on top of (near-zero) stage cost.
    assert reward > 50.0


def test_truncates_at_max_steps_without_termination():
    env = UnicycleGoalEnv(
        max_steps=3,
        # Put goal far away and force min-distance constraint to clear.
        min_start_goal_dist=0.0,
    )
    env.reset(seed=0, options={"state": [0.0, 0.0, 0.0], "goal": [100.0, 100.0]})
    for step in range(3):
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
        if step < 2:
            assert not terminated and not truncated
    assert truncated is True
    assert terminated is False


def test_terminated_and_truncated_never_both_true():
    env = UnicycleGoalEnv(max_steps=1, goal_tolerance=0.5)
    env.reset(seed=0, options={"state": [0.0, 0.0, 0.0], "goal": [0.0, 0.0]})
    _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
    assert terminated is True
    assert truncated is False


# ---- Wall clipping ------------------------------------------------------------


def test_position_clipped_to_workspace_bounds():
    env = UnicycleGoalEnv(workspace_bounds=((-1.0, 1.0), (-1.0, 1.0)))
    env.reset(seed=0, options={"state": [0.9, 0.0, 0.0], "goal": [-0.5, -0.5]})
    # Push hard rightward; should clip at x=1.0 instead of overshooting.
    for _ in range(50):
        env.step(np.array([20.0, 0.0]))
    assert env.state[0] <= 1.0 + 1e-12
    assert env.state[0] >= -1.0 - 1e-12
    assert env.state[1] <= 1.0 + 1e-12
    assert env.state[1] >= -1.0 - 1e-12


# ---- Reward shape -------------------------------------------------------------


def test_reward_is_negative_quadratic_when_not_reached():
    # Place robot away from goal, take zero action: reward = -err^T Q err exactly.
    env = UnicycleGoalEnv(Q=np.eye(2), R=np.zeros((2, 2)), reach_bonus=0.0)
    env.reset(seed=0, options={"state": [3.0, 4.0, 0.0], "goal": [0.0, 0.0]})
    _, reward, terminated, _, _ = env.step(np.array([0.0, 0.0]))
    assert not terminated
    # After zero-action step, position unchanged; cost = 3^2 + 4^2 = 25.
    assert reward == pytest.approx(-25.0, abs=1e-9)

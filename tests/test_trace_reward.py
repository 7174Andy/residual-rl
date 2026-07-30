"""recompute_reward must exactly reproduce env.py's own per-step reward,
using only the trace columns eval_seed_showcase.py already writes."""
from __future__ import annotations

from typing import cast

import gymnasium as gym
import numpy as np

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from two_wheel_robot.env.env import UnicycleGoalEnv
from two_wheel_robot.rl.trace_reward import recompute_reward


def test_recompute_reward_matches_env_exactly():
    env = gym.make("TwoWheelGoal-v0")
    try:
        env.reset(seed=0)
        base = cast(UnicycleGoalEnv, env.unwrapped)
        x = [float(base.state[0])]
        y = [float(base.state[1])]
        heading = [float(base.state[2])]
        v = [0.0]
        w = [0.0]
        real_rewards = []
        rng = np.random.default_rng(0)
        for _ in range(5):
            action = rng.uniform(base.action_bounds[:, 0], base.action_bounds[:, 1])
            _, reward, term, trunc, _ = env.step(action)
            x.append(float(base.state[0]))
            y.append(float(base.state[1]))
            heading.append(float(base.state[2]))
            v.append(float(base.last_action[0]))
            w.append(float(base.last_action[1]))
            real_rewards.append(reward)
            if term or trunc:
                break
        goal = base.goal.copy()
    finally:
        env.close()

    out = recompute_reward(
        np.array(x), np.array(y), np.array(heading), np.array(v), np.array(w), goal,
    )
    assert out["reward"][0] == 0.0
    assert np.allclose(out["reward"][1:], real_rewards)
    assert np.allclose(out["cum_reward"], np.cumsum(out["reward"]))
    assert out["dist"].shape == out["reward"].shape == out["reached"].shape


def test_single_row_trace_has_zero_reward():
    goal = np.array([5.0, 5.0])
    out = recompute_reward(
        np.array([0.0]), np.array([0.0]), np.array([0.0]),
        np.array([0.0]), np.array([0.0]), goal,
    )
    assert out["reward"][0] == 0.0
    assert out["cum_reward"][0] == 0.0
    assert np.isclose(out["dist"][0], 5.0 * np.sqrt(2))


def test_recompute_reward_includes_reach_bonus_matching_env():
    env = gym.make("TwoWheelGoal-v0")
    try:
        env.reset(seed=0, options={"state": [0.0, 0.0, 0.0], "goal": [0.3, 0.0]})
        base = cast(UnicycleGoalEnv, env.unwrapped)
        x = [float(base.state[0])]
        y = [float(base.state[1])]
        heading = [float(base.state[2])]
        v = [0.0]
        w = [0.0]

        action = np.array([0.0, 0.0])
        _, reward, term, _, _ = env.step(action)
        assert term
        x.append(float(base.state[0]))
        y.append(float(base.state[1]))
        heading.append(float(base.state[2]))
        v.append(float(base.last_action[0]))
        w.append(float(base.last_action[1]))
        goal = base.goal.copy()
    finally:
        env.close()

    out = recompute_reward(
        np.array(x), np.array(y), np.array(heading), np.array(v), np.array(w), goal,
    )
    assert bool(out["reached"][1]) is True
    assert np.isclose(out["reward"][1], reward)


def test_reached_flag_uses_goal_tolerance():
    goal = np.array([0.0, 0.0])
    # distances: 1.0, 0.4 (inside default 0.5 tolerance), 2.0
    out = recompute_reward(
        np.array([1.0, 0.4, 2.0]), np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 3.0, 3.0]), np.array([0.0, 0.0, 0.0]),
        goal,
    )
    assert list(out["reached"]) == [False, True, False]

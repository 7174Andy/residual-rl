"""ReacherGoal-v0: Gym compliance, the reachability filter, and the null control."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import reacher  # noqa: F401  registers the Gym ID
from reacher.model import NQ_ARM, is_reachable, load_model


@pytest.fixture(scope="module")
def env():
    e = gym.make("ReacherGoal-v0")
    yield e
    e.close()


def test_env_checker_passes():
    e = gym.make("ReacherGoal-v0").unwrapped
    check_env(e, skip_render_check=True)


def test_spaces(env):
    assert env.action_space.shape == (NQ_ARM,)
    assert np.allclose(env.action_space.low, -1.0)
    assert np.allclose(env.action_space.high, 1.0)
    assert env.observation_space.shape == (8,)


def test_every_sampled_goal_is_reachable(env):
    """2.1% of a raw disc draw is outside the annulus; an unreachable goal is an
    invisible ceiling on every reach rate in the results table."""
    model, _ = load_model()
    for seed in range(50):
        env.reset(seed=seed)
        assert is_reachable(model, env.unwrapped.goal)


def test_zero_torque_holds_the_arm(env):
    """Planar arm, gravity perpendicular: u = 0 must mean 'hold'. This is the
    model assumption the whole Reacher rationale rests on."""
    env.reset(seed=0)
    q0 = env.unwrapped.state[:NQ_ARM].copy()
    for _ in range(50):
        env.step(np.zeros(NQ_ARM, dtype=np.float32))
    assert np.allclose(env.unwrapped.state[:NQ_ARM], q0, atol=1e-6)


def test_random_actions_almost_never_reach(env):
    """The mandatory null. Journey 11: run the random control FIRST on any new
    interface, or every later interpretation is against the wrong reference."""
    rng = np.random.default_rng(0)
    reached = 0
    for seed in range(20):
        env.reset(seed=seed)
        hit = False
        for _ in range(env.unwrapped.max_steps):
            _, _, _, _, info = env.step(rng.uniform(-1, 1, NQ_ARM).astype(np.float32))
            hit = hit or info["reached"]
        reached += int(hit)
    assert reached <= 2, f"random actions reached {reached}/20 — task may be trivial"


def test_never_terminates_on_reach(env):
    """Spec D4: terminating would make 'arrive and hold' indistinguishable from
    'arrive and leave', which is the drift this whole pipeline targets."""
    env.reset(seed=1, options={"qpos": np.zeros(NQ_ARM), "goal": np.array([0.15, 0.0])})
    for _ in range(env.unwrapped.max_steps - 1):
        _, _, terminated, _trunc, _ = env.step(np.zeros(NQ_ARM, dtype=np.float32))
        assert not terminated
    _, _, terminated, truncated, _ = env.step(np.zeros(NQ_ARM, dtype=np.float32))
    assert truncated and not terminated


def test_reset_options_place_arm_and_goal(env):
    q0 = np.array([0.3, -0.4])
    goal = np.array([0.12, 0.05])
    env.reset(seed=0, options={"qpos": q0, "goal": goal})
    assert np.allclose(env.unwrapped.state[:NQ_ARM], q0)
    assert np.allclose(env.unwrapped.goal, goal)


def test_y_and_y_ref_match_deepc_setup(env):
    from reacher.deepc_setup import y_ref_for
    env.reset(seed=3)
    u = env.unwrapped
    assert u.y.shape == (4,)
    assert np.allclose(u.y[:NQ_ARM], u.state[:NQ_ARM])
    assert np.allclose(u.y[NQ_ARM:], u.tip)
    assert np.allclose(u.y_ref, y_ref_for(u.goal))

import gymnasium as gym
import mujoco
import numpy as np
import pytest
import quadruped  # registers QuadrupedGoal-v0
from quadruped.env import QuadrupedGoalEnv

def test_make_and_spaces():
    env = gym.make("QuadrupedGoal-v0")
    assert env.action_space.shape == (3,) and env.action_space.dtype == np.float32
    np.testing.assert_allclose(env.action_space.low, [-0.25, -0.2, -0.8])
    np.testing.assert_allclose(env.action_space.high, [0.6, 0.2, 0.8])
    assert env.observation_space.shape == (6,)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert set(info) >= {"state", "goal", "y", "y_ref", "pos_error", "distance", "action",
                         "step_idx", "reached", "fell", "ctrl"}

def test_y_and_y_ref():
    env = QuadrupedGoalEnv(); env.reset(seed=1)
    assert env.y.shape == (3,) and -np.pi <= env.y[2] <= np.pi
    np.testing.assert_allclose(env.y_ref, [*env.goal, 0.0])
    np.testing.assert_allclose(np.diag(env.Q), [1, 1, 2])
    np.testing.assert_allclose(np.diag(env.R), [3.3e-2, 3.3e-2, 1.1e-4])

def test_seed_determinism():
    def roll():
        env = QuadrupedGoalEnv(); obs, _ = env.reset(seed=3)
        rng = np.random.default_rng(0); out = [obs]
        for _ in range(30):
            out.append(env.step(rng.uniform(env.action_space.low, env.action_space.high))[0])
        return np.array(out)
    np.testing.assert_array_equal(roll(), roll())

def test_reset_options_state_goal():
    env = QuadrupedGoalEnv()
    env.reset(seed=0, options={"state": (0.5, -0.7, 1.0), "goal": (-1.0, 1.0)})
    np.testing.assert_allclose(env.goal, [-1.0, 1.0])
    assert np.linalg.norm(env.y[:2] - [0.5, -0.7]) < 0.1        # settle drift ~5 cm
    assert abs(np.angle(np.exp(1j * (env.y[2] - 1.0)))) < 0.2

def test_min_start_goal_dist():
    env = QuadrupedGoalEnv()
    for s in range(20):
        env.reset(seed=s)
        assert np.all(np.abs(env.goal) <= 1.5)
        # checked against the requested start; settle drift is ~5 cm
        assert np.linalg.norm(env.goal - env.y[:2]) >= 0.5 - 0.1

def test_reach_terminates_with_bonus():
    env = QuadrupedGoalEnv()
    env.reset(seed=0, options={"state": (0.0, 0.0, 0.0), "goal": (0.0, 0.0)})
    _, r, term, trunc, info = env.step(np.zeros(3))
    assert info["reached"] and term and not trunc
    err = env.y - env.y_ref
    assert r == pytest.approx(-(err @ env.Q @ err) + 100.0, rel=1e-9)

def test_truncation_at_max_steps():
    env = QuadrupedGoalEnv(max_steps=5)
    env.reset(seed=0, options={"state": (-1.4, 0.0, 0.0), "goal": (1.4, 0.0)})
    for k in range(5):
        _, _, term, trunc, _ = env.step(np.zeros(3))
    assert trunc and not term

def test_action_clipped():
    env = QuadrupedGoalEnv(); env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([5, -5, 5]))          # int input
    np.testing.assert_allclose(info["action"], [0.6, -0.2, 0.8])

def test_linear_reward_and_bad_mode():
    env = QuadrupedGoalEnv(reward="linear")
    env.reset(seed=0, options={"state": (-1.0, 0.0, 0.0), "goal": (1.0, 0.0)})
    u = np.array([0.3, 0.0, 0.0])
    _, r, _, _, info = env.step(u)
    assert r == pytest.approx(-info["distance"] - u @ env.R @ u, rel=1e-9)
    with pytest.raises(ValueError):
        QuadrupedGoalEnv(reward="nope")

def test_obs_in_space_outside_box():
    env = QuadrupedGoalEnv()
    env.reset(seed=0, options={"state": (1.4, 1.4, np.pi / 4), "goal": (-1.4, -1.4)})
    for _ in range(300):
        obs, *_ = env.step(np.array([0.6, 0.0, 0.0]))
        assert env.observation_space.contains(obs)
    assert np.abs(env.y[:2]).max() > 1.5                       # it actually left the box

def test_reset_clears_state():
    env = QuadrupedGoalEnv()
    env.reset(seed=0)
    for _ in range(25): env.step(np.array([0.5, 0.1, 0.4]))
    obs, info = env.reset(seed=0)
    obs2, info2 = QuadrupedGoalEnv().reset(seed=0)
    np.testing.assert_array_equal(obs, obs2)
    assert info["step_idx"] == 0 and np.all(info["action"] == 0)


def _tip_over(env, roll=1.2):
    q = env.sim.data.qpos
    q[3:7] = [np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0]
    mujoco.mj_forward(env.sim.model, env.sim.data)


def test_fall_terminates():
    env = QuadrupedGoalEnv()
    env.reset(seed=0, options={"state": (0.0, 0.0, 0.0), "goal": (1.0, 1.0)})
    _tip_over(env)
    _, _, term, trunc, info = env.step(np.zeros(3))
    assert info["fell"] and term and not trunc and not info["reached"]


def test_fall_reward_scaled():
    env = QuadrupedGoalEnv(max_steps=400)
    env.reset(seed=0, options={"state": (0.0, 0.0, 0.0), "goal": (1.0, 1.0)})
    for _ in range(9):
        env.step(np.zeros(3))
    _tip_over(env)
    _, r, _, _, info = env.step(np.zeros(3))  # step_idx == 10
    err = env.y - env.y_ref
    stage = -(err @ env.Q @ err)  # u = 0
    assert r == pytest.approx(stage * (400 - 10 + 1), rel=1e-9)


def test_no_fall_flag_when_walking():
    env = QuadrupedGoalEnv()
    env.reset(seed=0)
    for _ in range(50):
        *_, info = env.step(np.array([0.4, 0.0, 0.2]))
        assert not info["fell"]


def test_render_rgb_array():
    env = QuadrupedGoalEnv(render_mode="rgb_array"); env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8 and frame.std() > 0
    env.close()
    assert QuadrupedGoalEnv().render() is None
    with pytest.raises(ValueError):
        QuadrupedGoalEnv(render_mode="human")


def test_reset_ctrl_is_real_and_in_range():
    env = QuadrupedGoalEnv()
    _, info = env.reset(seed=0)
    c = env.sim.data.ctrl[env.sim.act_idx]
    assert info["ctrl"].shape == (4, 3) and np.array_equal(info["ctrl"], c)
    r = env.sim.model.actuator_ctrlrange[env.sim.act_idx]
    assert np.all(info["ctrl"] >= r[..., 0]) and np.all(info["ctrl"] <= r[..., 1])


def test_reached_wins_over_fall():
    env = QuadrupedGoalEnv()
    env.reset(seed=0, options={"state": (0.0, 0.0, 0.0)})
    env.goal = env.y[:2].copy()
    _tip_over(env)
    _, r, term, _, info = env.step(np.zeros(3))
    err = env.y - env.y_ref
    assert info["reached"] and info["fell"] and term
    assert r == pytest.approx(-(err @ env.Q @ err) + env.reach_bonus, rel=1e-9)

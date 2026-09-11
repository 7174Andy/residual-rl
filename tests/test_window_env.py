"""The window env's contract: the observation IS the clone's feature vector, and
the buffer slides under the same convention the clone was labelled with. If the
second one drifts, the warm-started actor is reading a one-step-shifted history
and nothing downstream is valid."""
from __future__ import annotations

import numpy as np
import pytest

import reacher  # noqa: F401  registers the Gym ID
from reacher.clone_features import feature_dim, featurize
from reacher.model import NQ_ARM


@pytest.fixture(scope="module")
def win_env():
    from reacher.window_env import CloneWindowEnv
    e = CloneWindowEnv()
    yield e
    e.close()


def test_spaces_match_the_feature_layout(win_env):
    assert win_env.action_space.shape == (NQ_ARM,)
    assert win_env.observation_space.shape == (feature_dim(5, 0),)
    assert np.all(np.isfinite(win_env.observation_space.low))
    assert np.all(np.isfinite(win_env.observation_space.high))
    # Buffer validity is the last feature and is one-sided.
    assert win_env.observation_space.low[-1] == 0.0
    assert win_env.observation_space.high[-1] == 1.0


def test_observation_stays_inside_its_box(win_env):
    obs, _info = win_env.reset(seed=4)
    assert win_env.observation_space.contains(obs)
    rng = np.random.default_rng(0)
    for _ in range(30):
        obs, _r, _t, _tr, _info = win_env.step(
            rng.uniform(-1, 1, size=NQ_ARM).astype(np.float32))
        assert win_env.observation_space.contains(obs)


def test_reset_primes_the_buffer_the_way_the_clone_was_labelled(win_env):
    obs, _info = win_env.reset(seed=4)
    assert np.array_equal(win_env._u_buf, np.zeros((5, NQ_ARM)))
    assert np.array_equal(win_env._y_buf, np.tile(win_env.base.y, (5, 1)))
    # Buffer validity at step 0 is 0.0: the history is pure priming.
    assert obs[-1] == pytest.approx(0.0)


def test_buffer_slide_matches_labelling(win_env):
    """Replay the same actions by hand, sliding the buffer the way
    `reacher/clone_data.py::rollout` does, and demand the env's observation
    equals `featurize` of the hand-built buffers at every step."""
    obs, _info = win_env.reset(seed=9)
    base = win_env.base
    u_buf = np.zeros((5, NQ_ARM))
    y_buf = np.tile(base.y, (5, 1))

    rng = np.random.default_rng(3)
    for t in range(12):
        expected = featurize(u_buf, y_buf, base.y, base.goal, base.step_idx, None)
        assert np.allclose(obs, expected.astype(np.float32), atol=1e-6), f"step {t}"

        a = rng.uniform(-1, 1, size=NQ_ARM)
        y_pre = base.y
        obs, _r, _t, _tr, info = win_env.step(a.astype(np.float32))
        u_buf = np.roll(u_buf, -1, axis=0)
        u_buf[-1] = info["action"]          # APPLIED torque, post-clip
        y_buf = np.roll(y_buf, -1, axis=0)
        y_buf[-1] = y_pre                   # PRE-step measurement


def test_actions_are_clipped_into_the_torque_box(win_env):
    win_env.reset(seed=2)
    for sign in (+1.0, -1.0):
        _o, _r, _t, _tr, info = win_env.step(
            (sign * 5.0 * np.ones(NQ_ARM)).astype(np.float32))
        assert np.all(np.abs(info["action"]) <= 1.0 + 1e-9)


def test_exposes_max_steps_and_goal_for_shared_episode_loops(win_env):
    win_env.reset(seed=1)
    assert win_env.max_steps == win_env.base.max_steps
    assert np.array_equal(win_env.goal, win_env.base.goal)

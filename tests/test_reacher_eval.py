"""The single shared episode loop every results row is scored with."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import reacher  # noqa: F401  registers the Gym ID
from reacher.eval import run_episode


@pytest.fixture(scope="module")
def env():
    e = gym.make("ReacherGoal-v0")
    yield e
    e.close()


def _zero(_env, _info):
    return np.zeros(2)


def test_zero_policy_reports_need_best_and_final(env):
    """A do-nothing policy: the arm holds, so best == final == need and the tip
    travels no distance."""
    r = run_episode(env, _zero, qpos=np.array([0.0, 0.0]),
                    goal=np.array([0.15, 0.0]))
    assert not r["reached"]
    assert r["best"] == pytest.approx(r["need"], abs=1e-6)
    assert r["final"] == pytest.approx(r["need"], abs=1e-6)
    assert r["path"] == pytest.approx(0.0, abs=1e-6)


def test_runs_the_full_horizon_so_final_is_uncensored(env):
    """Journey 12: early stopping censors `final`. The loop must never stop at
    first contact -- even when the goal IS reached."""
    r = run_episode(env, _zero, qpos=np.array([0.0, 0.0]),
                    goal=np.array([0.15, 0.0]))
    assert r["steps_run"] == env.unwrapped.max_steps

    # And with a goal the held tip actually sits inside: still the full horizon.
    hit = run_episode(env, _zero, qpos=np.array([0.0, 0.0]),
                      goal=np.array([0.205, 0.0]))
    assert hit["reached"] and hit["steps"] == 1
    assert hit["steps_run"] == env.unwrapped.max_steps


def test_eff_is_nan_when_no_progress_is_made(env):
    """path/net with net ~ 0 is meaningless, not infinite."""
    r = run_episode(env, _zero, qpos=np.array([0.0, 0.0]),
                    goal=np.array([0.15, 0.0]))
    assert np.isnan(r["eff"])


def test_best_tracks_the_closest_approach_not_the_last_step(env):
    """The distinction the whole drift argument rests on: a policy that passes
    through the target and leaves must report best << final."""
    # A constant torque swings the arm through; best should beat final.
    def swing(_env, _info):
        return np.array([0.35, -0.25])

    r = run_episode(env, swing, qpos=np.array([0.0, 0.0]),
                    goal=np.array([0.10, 0.10]))
    # `best <= final` is true BY CONSTRUCTION -- best is a running min over a
    # window that includes the last step -- so asserting it tests nothing. Assert
    # the real property instead: this policy passes through and leaves, so best
    # must beat final by a wide margin. Measured 3.3x; 2x leaves headroom while
    # still failing if `best` stopped tracking closest approach.
    assert r["best"] < 0.5 * r["final"], (
        f"best {r['best']:.4f} not clearly better than final {r['final']:.4f}")
    assert r["path"] > 0.0


def test_need_matches_the_reset_distance(env):
    r = run_episode(env, _zero, qpos=np.array([0.0, 0.0]),
                    goal=np.array([0.15, 0.0]))
    # Held tip at q=(0,0) is (0.21, 0); goal at (0.15, 0) -> 0.06 m.
    assert r["need"] == pytest.approx(0.06, abs=1e-6)

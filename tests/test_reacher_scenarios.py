"""The frozen scenario file must contain only physically attainable goals."""
from __future__ import annotations

import numpy as np
import pytest

from reacher.model import NQ_ARM, is_reachable, load_model

SCENARIOS = "data/reacher_scenarios_v1.npz"


@pytest.fixture(scope="module")
def payload():
    try:
        with np.load(SCENARIOS) as z:
            return {k: z[k] for k in z.files}
    except FileNotFoundError:
        pytest.skip(f"{SCENARIOS} not generated; run scripts/make_reacher_scenarios.py")


def test_shapes_agree(payload):
    n = payload["qpos"].shape[0]
    assert payload["qpos"].shape == (n, NQ_ARM)
    assert payload["goal"].shape == (n, 2)
    assert payload["need"].shape == (n,)


def test_every_goal_is_reachable(payload):
    model, _ = load_model()
    assert all(is_reachable(model, g) for g in payload["goal"])


def test_no_scenario_starts_already_solved(payload):
    """A scenario whose start is inside tolerance measures nothing."""
    assert (payload["need"] > 0.01).all()


def test_recorded_need_matches_forward_kinematics(payload):
    """`need` must be the true tip-to-goal distance at the start configuration.

    Without this the metric every row is normalised by (`path/net`, `closed %`)
    could be silently wrong while every other test still passed -- `need` is
    stored, not recomputed, at evaluation time.
    """
    from reacher.model import fingertip, set_state
    model, data = load_model()
    for q0, g, need in zip(payload["qpos"], payload["goal"], payload["need"]):
        set_state(model, data, q0, goal=g)
        assert np.isclose(np.linalg.norm(fingertip(data) - g), need, atol=1e-9)


def test_scenarios_are_held_out_from_the_training_stream(payload):
    """THE GUARD THAT WAS MISSING.

    `ReacherGoalEnv.reset` draws goal-then-config from `self.np_random`, in the
    same order this generator does. So a scenario file built at seed 0 is
    bit-identical to training episodes 0..n-1 of any env seeded 0 -- and SB3 seeds
    once then auto-resets UNSEEDED, so training walks that same stream. Verified
    before this test existed: the mapping was the identity across all 40
    scenarios, meaning every evaluated episode was also a trained-on episode for
    both RL rows.

    Nothing else in the suite could see it: shapes, reachability and `need` were
    all still correct. Only provenance was wrong.
    """
    import gymnasium as gym

    import reacher  # noqa: F401

    env = gym.make("ReacherGoal-v0")
    try:
        _o, _i = env.reset(seed=0)
        stream = [(env.unwrapped.state[:NQ_ARM].copy(),
                   env.unwrapped.goal.copy())]
        for _ in range(40):                      # first 40 training episodes
            for _ in range(env.unwrapped.max_steps):
                _o, _r, _t, trunc, _i = env.step(np.zeros(NQ_ARM, dtype=np.float32))
                if trunc:
                    break
            _o, _i = env.reset()                 # unseeded, as SB3 auto-reset does
            stream.append((env.unwrapped.state[:NQ_ARM].copy(),
                           env.unwrapped.goal.copy()))
    finally:
        env.close()

    for q, g in zip(payload["qpos"][:40], payload["goal"][:40]):
        for qt, gt in stream:
            assert not (np.allclose(q, qt, atol=1e-12)
                        and np.allclose(g, gt, atol=1e-12)), (
                "frozen scenario appears in the seed-0 training stream; "
                "regenerate with a disjoint --seed")

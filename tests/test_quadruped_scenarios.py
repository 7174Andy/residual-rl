import numpy as np
import pytest

from quadruped import scenarios as S
from quadruped.env import QuadrupedGoalEnv

PINNED = "266ce66880971339a528b6bca4e99b6e70b8fe140e2a4ba63272fe4651ce4b8c"


def test_start_request_recorded():
    env = QuadrupedGoalEnv()
    s, g = np.array([0.3, -0.4, 1.0]), np.array([1.0, 1.0])
    env.reset(options={"state": s, "goal": g})
    np.testing.assert_array_equal(env.start_request, s)


def test_split_disjoint_and_sized():
    assert len(S.TUNE_IDS) == 40 and len(S.EVAL_IDS) == 78
    assert not set(S.TUNE_IDS) & set(S.EVAL_IDS)
    try:
        sc = S.load()
    except FileNotFoundError:
        pytest.skip("artefact absent")
    rows = np.hstack([sc["state"], sc["goal"]])
    assert len(np.unique(rows, axis=0)) == len(rows) == 118


def test_replay_identical():
    sc = S.generate(n=4)
    env = QuadrupedGoalEnv()
    for i in (0, 3):
        env.reset(seed=int(sc["seed"][i]))
        a = env.y.copy()
        S.reset_to(env, sc, i)
        np.testing.assert_array_equal(a, env.y)


def test_validate_against_env():
    sc = S.generate(n=2)
    S.validate_against_env(QuadrupedGoalEnv(), sc)
    with pytest.raises(ValueError):
        S.validate_against_env(QuadrupedGoalEnv(max_steps=399), sc)


def test_checksum_pinned():
    try:
        sc = S.load()
    except FileNotFoundError:
        pytest.skip("artefact absent")
    assert S.checksum(sc) == PINNED

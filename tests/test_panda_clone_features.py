import numpy as np
import pytest

from panda.clone_features import featurize, feature_dim


def test_feature_dim_at_T_ini_5():
    assert feature_dim(5) == 99


def test_layout_blocks_land_where_documented():
    rng = np.random.default_rng(0)
    u_ini = rng.normal(size=(5, 7))
    y_ini = rng.normal(size=(5, 10))
    y_cur = rng.normal(size=10)
    goal = rng.normal(size=3)
    f = featurize(u_ini, y_ini, y_cur, goal, step_idx=5)
    assert f.shape == (99,)
    assert np.allclose(f[0:35], u_ini.ravel())
    assert np.allclose(f[35:85], y_ini.ravel())
    assert np.allclose(f[85:95], y_cur)
    assert np.allclose(f[95:98], y_cur[7:] - goal)
    assert f[98] == 1.0


def test_buffer_validity_ramps_then_saturates():
    u_ini, y_ini = np.zeros((5, 7)), np.zeros((5, 10))
    y_cur, goal = np.zeros(10), np.zeros(3)
    vals = [featurize(u_ini, y_ini, y_cur, goal, t)[98] for t in [0, 2, 5, 40]]
    assert vals == [0.0, 0.4, 1.0, 1.0]


def test_T_ini_mismatch_raises_instead_of_broadcasting():
    with pytest.raises(ValueError, match="T_ini mismatch"):
        featurize(np.zeros((5, 7)), np.zeros((4, 10)), np.zeros(10),
                  np.zeros(3), 0)


def test_goal_enters_relative_but_state_stays_absolute():
    """Shifting tip and goal together must move ONLY the absolute blocks.

    The goal is relative (`tip - goal`) for the same generalization reason the
    env's observation is; the tip and `q` stay absolute because the arm is
    anchored at the origin, so an absolute tip IS physically meaningful state.
    A change that made the whole vector translation-invariant would delete
    information the clone needs, and this pins that it has not happened.
    """
    rng = np.random.default_rng(1)
    u_ini = rng.normal(size=(5, 7))
    y_ini = rng.normal(size=(5, 10))
    y_cur = rng.normal(size=10)
    goal = rng.normal(size=3)
    shift = np.array([0.1, -0.2, 0.3])

    y_cur_s = y_cur.copy()
    y_cur_s[7:] += shift
    a = featurize(u_ini, y_ini, y_cur, goal, 5)
    b = featurize(u_ini, y_ini, y_cur_s, goal + shift, 5)

    assert np.allclose(a[95:98], b[95:98])        # relative block unchanged
    assert not np.allclose(a[92:95], b[92:95])    # absolute tip block moved

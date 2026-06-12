# tests/test_deepc_setup.py
"""Canonical deep-lcc builder + bearing reference."""
from __future__ import annotations

import numpy as np
import pytest

from two_wheel_robot.rl.deepc_setup import bearing_y_ref, build_canonical_deepc


def test_bearing_y_ref_points_at_goal():
    state = np.array([0.0, 0.0, 0.0])
    goal = np.array([0.0, 5.0])  # straight "north" -> bearing +pi/2
    yref = bearing_y_ref(state, goal)
    assert yref.shape == (3,)
    assert np.allclose(yref[:2], goal)
    assert np.isclose(yref[2], np.pi / 2)

    # A non-degenerate case pins the atan2 argument order (catches an axis swap):
    # from (1, 1) to (2, 0) the bearing is atan2(0 - 1, 2 - 1) = -pi/4.
    yref2 = bearing_y_ref(np.array([1.0, 1.0, 0.0]), np.array([2.0, 0.0]))
    assert np.isclose(yref2[2], -np.pi / 4)


def test_build_canonical_deepc_has_four_libraries():
    try:
        deepc, info = build_canonical_deepc()
    except FileNotFoundError:
        pytest.skip("data/libraries_v0.npz not present")
    assert deepc._n_lib == 4
    assert deepc.T_ini == 5 and deepc.N == 12
    assert info["anchors"].shape == (4,)
    assert info["u_init_midpoint"].shape == (2,)
    assert info["action_bounds"].shape == (2, 2)
    # Q heading weight is 2 (paper / measured-baseline config).
    assert np.isclose(deepc.Q[2, 2], 2.0)

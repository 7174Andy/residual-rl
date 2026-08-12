# tests/test_clone_features.py
"""Featurizer: shape, heading continuity, library one-hot, hand-checked vector."""
from __future__ import annotations

import numpy as np
import pytest

from core.deepc import DeePC
from core.hankel import build_hankel
from two_wheel_robot.rl.features import featurize, select_library_index
from tests.deepc_scenarios import ANCHORS, N, Q, R, T_INI, multi_libraries

ANCHORS_ARR = np.asarray(ANCHORS, dtype=np.float64)


def _config(T_ini=T_INI):
    rng = np.random.default_rng(1)
    u_ini = rng.uniform(-1, 1, size=(T_ini, 2))
    y_ini = rng.standard_normal((T_ini, 3))
    y_current = np.array([0.5, -0.5, 0.3])
    y_ref = np.array([2.0, 3.0, 0.0])
    return u_ini, y_ini, y_current, y_ref


def test_output_dim():
    u_ini, y_ini, y_current, y_ref = _config()
    feat = featurize(u_ini, y_ini, y_current, y_ref, ANCHORS_ARR)
    assert feat.shape == (6 * T_INI + 6 + len(ANCHORS_ARR),)
    assert feat.shape == (40,)  # T_ini=5, N_lib=4


def test_heading_encoding_is_continuous_across_pi_wrap():
    # A tiny heading change near +/-pi must yield a tiny feature change (no jump).
    u_ini, y_ini, _, y_ref = _config()
    cur_a = np.array([0.0, 0.0, np.pi - 1e-4])
    cur_b = np.array([0.0, 0.0, -np.pi + 1e-4])  # ~2e-4 apart on the circle
    fa = featurize(u_ini, y_ini, cur_a, y_ref, ANCHORS_ARR)
    fb = featurize(u_ini, y_ini, cur_b, y_ref, ANCHORS_ARR)
    # sin/cos of y_current are features 6*T_ini+2 and +3.
    base = 6 * T_INI
    assert abs(fa[base + 2] - fb[base + 2]) < 1e-3  # sin
    assert abs(fa[base + 3] - fb[base + 3]) < 1e-3  # cos


def test_library_onehot_matches_deepc_select_index():
    libs = [build_hankel(u, y, T_ini=T_INI, N=N) for u, y in multi_libraries(4)]
    deepc = DeePC(libs, anchor_headings=ANCHORS_ARR, Q=Q, R=R, T_ini=T_INI, N=N,
                  lambda_g=1.0, lambda_y=1e3, solver="CLARABEL")
    rng = np.random.default_rng(7)
    for _ in range(50):
        h = float(rng.uniform(-np.pi, np.pi))
        idx_std = select_library_index(h, ANCHORS_ARR)
        # DeePC._select_index is the reference implementation.
        assert idx_std == deepc._select_index(h)


def test_mismatched_buffer_lengths_raise():
    # u_ini has 2 rows, y_ini has 1 — must raise, not silently broadcast.
    with pytest.raises(ValueError):
        featurize(
            np.zeros((2, 2)), np.zeros((1, 3)),
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0]), ANCHORS_ARR,
        )


def test_hand_checked_vector():
    u_ini = np.array([[1.0, 2.0]])           # T_ini = 1
    y_ini = np.array([[0.5, 1.5, 0.0]])
    y_current = np.array([3.0, 4.0, 0.3])
    y_ref = np.array([7.0, 8.0, 0.0])
    anchors = np.array([0.0, np.pi])          # heading 0.3 -> anchor 0 -> idx 0
    feat = featurize(u_ini, y_ini, y_current, y_ref, anchors)
    expected = np.array([
        0.5, 1.5, np.sin(0.0), np.cos(0.0), 1.0, 2.0,   # buffer step 0
        3.0, 4.0, np.sin(0.3), np.cos(0.3),             # y_current
        7.0, 8.0,                                        # goal
        1.0, 0.0,                                        # library one-hot
    ])
    assert feat.shape == (6 * 1 + 6 + 2,)
    assert np.allclose(feat, expected)

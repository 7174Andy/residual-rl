"""Tests for core.hankel."""

from __future__ import annotations

import numpy as np
import pytest

from core.hankel import build_hankel


# ---- Shapes ------------------------------------------------------------------


def test_shapes_match_formula():
    T, m_u, p_y = 100, 2, 3
    T_ini, N = 5, 12
    rng = np.random.default_rng(0)
    u = rng.standard_normal((T, m_u))
    y = rng.standard_normal((T, p_y))
    Up, Uf, Yp, Yf = build_hankel(u, y, T_ini=T_ini, N=N)
    n_cols = T - (T_ini + N) + 1
    assert Up.shape == (T_ini * m_u, n_cols)
    assert Uf.shape == (N * m_u, n_cols)
    assert Yp.shape == (T_ini * p_y, n_cols)
    assert Yf.shape == (N * p_y, n_cols)


def test_dtype_preserved():
    u = np.zeros((20, 2), dtype=np.float64)
    y = np.zeros((20, 3), dtype=np.float64)
    Up, _, Yp, _ = build_hankel(u, y, T_ini=2, N=3)
    assert Up.dtype == np.float64
    assert Yp.dtype == np.float64


# ---- Block content -----------------------------------------------------------


def test_block_content_indices():
    """Up[k*m:(k+1)*m, j] must equal u[j+k]; Uf the analogous suffix."""
    T, m_u = 12, 2
    u = np.arange(T * m_u).reshape(T, m_u).astype(np.float64)
    y = np.arange(T * 3).reshape(T, 3).astype(np.float64)
    T_ini, N = 2, 3
    n_cols = T - (T_ini + N) + 1  # = 8

    Up, Uf, Yp, Yf = build_hankel(u, y, T_ini=T_ini, N=N)
    for j in range(n_cols):
        for k in range(T_ini):
            np.testing.assert_array_equal(Up[k * m_u : (k + 1) * m_u, j], u[j + k])
            np.testing.assert_array_equal(Yp[k * 3 : (k + 1) * 3, j], y[j + k])
        for k in range(N):
            np.testing.assert_array_equal(
                Uf[k * m_u : (k + 1) * m_u, j], u[j + T_ini + k]
            )
            np.testing.assert_array_equal(
                Yf[k * 3 : (k + 1) * 3, j], y[j + T_ini + k]
            )


def test_concat_past_future_is_full_block_hankel():
    """vstack(Up, Uf) reconstructs the full L-block Hankel of u."""
    T, m_u = 30, 2
    u = np.arange(T * m_u).reshape(T, m_u).astype(np.float64)
    y = np.zeros((T, 3))
    T_ini, N = 4, 8
    Up, Uf, _, _ = build_hankel(u, y, T_ini=T_ini, N=N)
    full = np.vstack([Up, Uf])
    L = T_ini + N
    n_cols = T - L + 1
    assert full.shape == (L * m_u, n_cols)
    # Spot-check: column j, block k → u[j+k]
    for j in range(n_cols):
        for k in range(L):
            np.testing.assert_array_equal(full[k * m_u : (k + 1) * m_u, j], u[j + k])


# ---- Validation --------------------------------------------------------------


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        build_hankel(np.zeros((10, 2)), np.zeros((9, 3)), T_ini=2, N=3)


def test_rejects_too_short_trajectory():
    with pytest.raises(ValueError):
        build_hankel(np.zeros((10, 2)), np.zeros((10, 3)), T_ini=5, N=10)


def test_rejects_non_positive_horizons():
    u, y = np.zeros((20, 2)), np.zeros((20, 3))
    with pytest.raises(ValueError):
        build_hankel(u, y, T_ini=0, N=5)
    with pytest.raises(ValueError):
        build_hankel(u, y, T_ini=5, N=0)


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        build_hankel(np.zeros(20), np.zeros((20, 3)), T_ini=2, N=3)


# ---- End-to-end with real data ----------------------------------------------


def test_runs_on_real_collected_trajectory():
    """Sanity: build Hankels on a freshly collected trajectory."""
    import gymnasium as gym
    import two_wheel_robot.env  # noqa: F401
    from two_wheel_robot.controllers.data_collection import (
        DEFAULT_SAMPLE_BOUNDS,
        collect_trajectory,
    )

    env = gym.make("TwoWheelGoal-v0")
    u, y = collect_trajectory(
        env,
        T=200,
        rng=np.random.default_rng(0),
        init_state=np.array([0.0, 0.0, np.pi / 4]),
        sample_bounds=DEFAULT_SAMPLE_BOUNDS,
    )
    Up, Uf, Yp, Yf = build_hankel(u, y, T_ini=5, N=12)
    n_cols = 200 - 17 + 1  # = 184
    assert Up.shape == (10, n_cols)
    assert Uf.shape == (24, n_cols)
    assert Yp.shape == (15, n_cols)
    assert Yf.shape == (36, n_cols)

"""Tests for two_wheel_robot.controllers.deepc.DeePC and LibrarySwitchingDeePC."""

from __future__ import annotations

import numpy as np
import pytest

from two_wheel_robot.controllers.deepc import DeePC, LibrarySwitchingDeePC
from two_wheel_robot.controllers.hankel import build_hankel


# ---- Helpers -----------------------------------------------------------------


def _scalar_lti_data(T: int = 200, a: float = 0.9, y0: float = 5.0, seed: int = 0):
    """Generate (u, y) from `y_{t+1} = a*y_t + u_t` with PE inputs."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((T, 1))
    y = np.zeros((T, 1))
    y[0] = y0
    for t in range(T - 1):
        y[t + 1] = a * y[t] + u[t]
    return u, y, a


# ---- Construction / validation ----------------------------------------------


def test_constructor_validates_shapes():
    T = 50
    u = np.random.randn(T, 2)
    y = np.random.randn(T, 3)
    Up, Uf, Yp, Yf = build_hankel(u, y, T_ini=5, N=12)
    Q = np.eye(3)
    R = np.eye(2)
    # Wrong Q shape
    with pytest.raises(ValueError):
        DeePC(Up, Uf, Yp, Yf, Q=np.eye(2), R=R, T_ini=5, N=12)
    # Wrong R shape
    with pytest.raises(ValueError):
        DeePC(Up, Uf, Yp, Yf, Q=Q, R=np.eye(3), T_ini=5, N=12)
    # Mismatched T_ini vs Hankel rows
    with pytest.raises(ValueError):
        DeePC(Up, Uf, Yp, Yf, Q=Q, R=R, T_ini=4, N=12)


def test_act_before_reset_raises():
    u, y, _ = _scalar_lti_data(T=60)
    Up, Uf, Yp, Yf = build_hankel(u, y, T_ini=3, N=5)
    c = DeePC(Up, Uf, Yp, Yf, Q=np.eye(1), R=0.01 * np.eye(1), T_ini=3, N=5)
    with pytest.raises(RuntimeError):
        c.act(np.array([1.0]), np.array([0.0]))


# ---- API surface -------------------------------------------------------------


def test_act_returns_action_of_correct_shape():
    u, y, _ = _scalar_lti_data(T=80)
    Up, Uf, Yp, Yf = build_hankel(u, y, T_ini=3, N=5)
    c = DeePC(Up, Uf, Yp, Yf, Q=np.eye(1), R=0.01 * np.eye(1), T_ini=3, N=5)
    c.reset(y_initial=np.array([5.0]))
    u_t = c.act(np.array([5.0]), y_ref=np.array([0.0]))
    assert u_t.shape == (1,)
    assert np.isfinite(u_t).all()


def test_y_ref_per_step_horizon_accepted():
    u, y, _ = _scalar_lti_data(T=80)
    Up, Uf, Yp, Yf = build_hankel(u, y, T_ini=3, N=5)
    c = DeePC(Up, Uf, Yp, Yf, Q=np.eye(1), R=0.01 * np.eye(1), T_ini=3, N=5)
    c.reset(np.array([5.0]))
    y_ref_window = np.linspace(5.0, 0.0, 5).reshape(5, 1)
    u_t = c.act(np.array([5.0]), y_ref=y_ref_window)
    assert u_t.shape == (1,)


def test_buffer_slides_after_act():
    u, y, _ = _scalar_lti_data(T=60)
    Up, Uf, Yp, Yf = build_hankel(u, y, T_ini=3, N=5)
    c = DeePC(Up, Uf, Yp, Yf, Q=np.eye(1), R=0.01 * np.eye(1), T_ini=3, N=5)
    c.reset(np.array([5.0]))
    # After reset, y_buf is [5, 5, 5].
    assert c._y_buf is not None
    np.testing.assert_array_equal(c._y_buf.flatten(), [5.0, 5.0, 5.0])
    # Call act with y_current = 4.0; afterwards y_buf should be [5, 5, 4].
    c.act(np.array([4.0]), np.array([0.0]))
    np.testing.assert_array_equal(c._y_buf.flatten(), [5.0, 5.0, 4.0])
    # Another act with y_current = 3.0; y_buf should become [5, 4, 3].
    c.act(np.array([3.0]), np.array([0.0]))
    np.testing.assert_array_equal(c._y_buf.flatten(), [5.0, 4.0, 3.0])


# ---- Closed-loop behavior on simple LTI -------------------------------------


def test_closed_loop_drives_scalar_lti_to_reference():
    """Sanity: control a known scalar LTI system, observe y → 0 over a horizon."""
    u, y, a = _scalar_lti_data(T=400, a=0.9, y0=5.0, seed=0)
    Up, Uf, Yp, Yf = build_hankel(u, y, T_ini=4, N=8)
    c = DeePC(
        Up, Uf, Yp, Yf,
        Q=np.array([[1.0]]),
        R=np.array([[0.01]]),
        T_ini=4, N=8,
        lambda_g=1.0,
        lambda_y=1e4,
    )
    y_current = np.array([5.0])
    y_ref = np.array([0.0])
    c.reset(y_current)
    for _ in range(50):
        u_t = c.act(y_current, y_ref)
        y_current = a * y_current + u_t
    assert abs(y_current[0]) < 0.5, f"final y={y_current[0]:.3f} should be near 0"


def _build_3d_controller(seed: int, T_ini: int = 5, N: int = 8) -> DeePC:
    """Build a small DeePC with m_u=2, p_y=3 for use in switcher tests."""
    rng = np.random.default_rng(seed)
    T = 200
    u = rng.uniform(-1, 1, size=(T, 2))
    y = rng.standard_normal((T, 3))
    Up, Uf, Yp, Yf = build_hankel(u, y, T_ini=T_ini, N=N)
    return DeePC(
        Up, Uf, Yp, Yf,
        Q=np.eye(3), R=0.01 * np.eye(2),
        T_ini=T_ini, N=N,
        lambda_g=1.0, lambda_y=1e3,
    )


# ---- LibrarySwitchingDeePC selection logic ----------------------------------


def test_switcher_selects_quadrant_anchor():
    controllers = [_build_3d_controller(seed=i) for i in range(4)]
    anchors = [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4]
    sw = LibrarySwitchingDeePC(controllers, anchors)
    # Headings inside each quadrant route to that quadrant's anchor.
    assert sw._select_index(0.1) == 0       # Q0: [0, π/2)
    assert sw._select_index(np.pi / 3) == 0
    assert sw._select_index(2.0) == 1       # Q1: [π/2, π), 2.0 ≈ 0.64π
    assert sw._select_index(-2.0) == 2      # Q2: [-π, -π/2)
    assert sw._select_index(-0.3) == 3      # Q3: [-π/2, 0)


def test_switcher_wraps_around_pi():
    controllers = [_build_3d_controller(seed=i) for i in range(4)]
    anchors = [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4]
    sw = LibrarySwitchingDeePC(controllers, anchors)
    # Heading just below +π is closer to the -3π/4 anchor than 3π/4 (going the
    # other way round). 3.0 rad: distance to 3π/4 ≈ 0.71, to -3π/4 ≈ -0.71 (also
    # 0.71 abs); to π/4 ≈ 2.21; to -π/4 ≈ -3π or ≈ 2.93 → wraps to ~ -0.21... wait
    # actually need to be careful. Use a clearer case: 3.13 ≈ +π.
    # +π and -π are the same point. -3π/4 = -2.36 is closer (distance ≈ 0.78)
    # than 3π/4 = 2.36 (also distance 0.78 the other way). Tie — argmin picks the
    # first. Pick a heading slightly less than π: should still hit 3π/4 (Q1).
    assert sw._select_index(np.pi - 0.1) == 1
    # Heading slightly more than -π (i.e., +π side after wrap): hits -3π/4 (Q2).
    assert sw._select_index(-np.pi + 0.1) == 2


# ---- LibrarySwitchingDeePC API ----------------------------------------------


def test_switcher_act_before_reset_raises():
    controllers = [_build_3d_controller(seed=i) for i in range(2)]
    sw = LibrarySwitchingDeePC(controllers, [0.0, np.pi])
    with pytest.raises(RuntimeError):
        sw.act(np.array([0.0, 0.0, 0.0]), np.zeros(3))


def test_switcher_reset_and_act_returns_correct_shape():
    controllers = [_build_3d_controller(seed=i) for i in range(2)]
    sw = LibrarySwitchingDeePC(controllers, [0.0, np.pi])
    sw.reset(y_initial=np.array([0.0, 0.0, 0.1]))
    u_t = sw.act(np.array([0.0, 0.0, 0.1]), y_ref=np.array([1.0, 1.0, 0.1]))
    assert u_t.shape == (2,)
    assert np.isfinite(u_t).all()


def test_switcher_buffer_syncs_after_act():
    controllers = [_build_3d_controller(seed=i) for i in range(2)]
    sw = LibrarySwitchingDeePC(controllers, [0.0, np.pi])
    sw.reset(y_initial=np.array([0.0, 0.0, 0.1]))
    # Drive a step with heading near 0 → controller 0 is used.
    sw.act(np.array([0.0, 0.0, 0.1]), y_ref=np.array([1.0, 1.0, 0.1]))
    assert sw.last_library_idx == 0
    # The shared y_buf's most recent entry is the y_current we just passed in.
    assert sw._y_buf is not None
    np.testing.assert_allclose(sw._y_buf[-1], [0.0, 0.0, 0.1])
    # Now drive a step with heading near +π → controller 1 is used. Its buffer
    # must inherit what was just built on controller 0.
    sw.act(np.array([1.0, 1.0, np.pi - 0.1]), y_ref=np.array([1.0, 1.0, 0.1]))
    assert sw.last_library_idx == 1
    np.testing.assert_allclose(sw._y_buf[-1], [1.0, 1.0, np.pi - 0.1])
    # The most-recent-but-one slot still holds the previous y_current (proves
    # the buffer continued sliding through the library switch).
    np.testing.assert_allclose(sw._y_buf[-2], [0.0, 0.0, 0.1])


# ---- LibrarySwitchingDeePC validation ---------------------------------------


def test_switcher_rejects_empty_controllers():
    with pytest.raises(ValueError):
        LibrarySwitchingDeePC([], [])


def test_switcher_rejects_mismatched_anchor_count():
    controllers = [_build_3d_controller(seed=i) for i in range(2)]
    with pytest.raises(ValueError):
        LibrarySwitchingDeePC(controllers, [0.0, 1.0, 2.0])


def test_switcher_rejects_mismatched_inner_shapes():
    c1 = _build_3d_controller(seed=0, T_ini=5)
    c2 = _build_3d_controller(seed=1, T_ini=4)  # different T_ini
    with pytest.raises(ValueError):
        LibrarySwitchingDeePC([c1, c2], [0.0, 1.0])


def test_u_bounds_respected():
    """Control inputs returned by act() lie within u_bounds when set."""
    u, y, a = _scalar_lti_data(T=200, a=0.9, y0=10.0, seed=1)
    Up, Uf, Yp, Yf = build_hankel(u, y, T_ini=3, N=5)
    u_min = np.array([-0.5])
    u_max = np.array([0.5])
    # Paper defaults (lambda_g=2, lambda_y=3e6) over-regularize this 1-D toy
    # system under L1; use lighter weights so the QP is well-conditioned.
    c = DeePC(
        Up, Uf, Yp, Yf,
        Q=np.eye(1), R=0.01 * np.eye(1),
        T_ini=3, N=5,
        lambda_g=1.0,
        lambda_y=1e4,
        u_bounds=(u_min, u_max),
    )
    y_current = np.array([10.0])
    c.reset(y_current)
    for _ in range(20):
        u_t = c.act(y_current, np.array([0.0]))
        # Allow small numerical slack.
        assert u_t[0] >= u_min[0] - 1e-6, f"u={u_t[0]} below bound"
        assert u_t[0] <= u_max[0] + 1e-6, f"u={u_t[0]} above bound"
        y_current = a * y_current + u_t

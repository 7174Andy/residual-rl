"""Tests for the single parametric DeePC controller.

The controller holds all orientation-keyed libraries, builds one cached QP with
the Hankel matrices as `cp.Parameter`s, and swaps which library feeds the
predictor each step based on heading. These tests pin:

- equivalence with the legacy two-class implementation (golden fixtures in
  `fixtures/deepc_golden.npz`, generated from the old `DeePC` /
  `LibrarySwitchingDeePC`),
- constructor validation,
- the public API (reset/act, shapes, buffer sliding, bounds),
- orientation routing and warm-start-reset-on-switch behavior.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import deepc_scenarios as S  # noqa: E402

from two_wheel_robot.controllers.deepc import DeePC  # noqa: E402
from two_wheel_robot.controllers.hankel import build_hankel  # noqa: E402

_FIXTURE = np.load(pathlib.Path(__file__).resolve().parent / "fixtures" / "deepc_golden.npz")


# ---- Builders from shared scenarios -----------------------------------------


def _defaults(**override) -> dict:
    params = dict(
        Q=S.Q, R=S.R, T_ini=S.T_INI, N=S.N,
        lambda_g=S.LAMBDA_G, lambda_y=S.LAMBDA_Y, solver=S.SOLVER,
    )
    params.update(override)
    return params


def _build_single(**kw) -> DeePC:
    u, y = S.single_library()
    lib = build_hankel(u, y, T_ini=S.T_INI, N=S.N)
    return DeePC([lib], anchor_headings=[0.0], **_defaults(**kw))


def _build_multi(n: int = 4, **kw) -> DeePC:
    libs = [build_hankel(u, y, T_ini=S.T_INI, N=S.N) for (u, y) in S.multi_libraries(n)]
    return DeePC(libs, anchor_headings=S.ANCHORS[:n], **_defaults(**kw))


def _scalar_lti(T: int, a: float, y0: float, seed: int):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((T, 1))
    y = np.zeros((T, 1))
    y[0] = y0
    for t in range(T - 1):
        y[t + 1] = a * y[t] + u[t]
    return u, y


# ---- Equivalence with legacy implementation ---------------------------------


def test_single_library_matches_legacy_golden():
    c = _build_single()
    seq = S.single_input_sequence()
    c.reset(seq[0][0])
    got = np.array([c.act(yc, yr) for yc, yr in seq])
    np.testing.assert_allclose(got, _FIXTURE["single_u"], atol=1e-5, rtol=1e-4)


def test_multi_library_matches_legacy_golden_outputs_and_routing():
    c = _build_multi()
    seq = S.multi_input_sequence()
    c.reset(seq[0][0])
    got, idx = [], []
    for yc, yr in seq:
        got.append(c.act(yc, yr))
        idx.append(c.last_library_idx)
    np.testing.assert_allclose(np.array(got), _FIXTURE["multi_u"], atol=1e-5, rtol=1e-4)
    assert idx == _FIXTURE["multi_idx"].tolist()


# ---- Constructor validation -------------------------------------------------


def test_rejects_empty_library_list():
    with pytest.raises(ValueError):
        DeePC([], anchor_headings=[], Q=S.Q, R=S.R, T_ini=S.T_INI, N=S.N)


def test_rejects_mismatched_ncols_across_libraries():
    u0, y0 = S.library_data(0, T=200)
    u1, y1 = S.library_data(1, T=180)  # different length -> different n_cols
    libA = build_hankel(u0, y0, T_ini=S.T_INI, N=S.N)
    libB = build_hankel(u1, y1, T_ini=S.T_INI, N=S.N)
    with pytest.raises(ValueError):
        DeePC([libA, libB], anchor_headings=[0.0, np.pi], Q=S.Q, R=S.R,
              T_ini=S.T_INI, N=S.N, solver=S.SOLVER)


def test_rejects_wrong_Q_shape():
    with pytest.raises(ValueError):
        _build_single(Q=np.eye(2))


def test_rejects_wrong_R_shape():
    with pytest.raises(ValueError):
        _build_single(R=np.eye(3))


def test_rejects_mismatched_Tini_vs_hankel():
    u, y = S.single_library()
    lib = build_hankel(u, y, T_ini=S.T_INI, N=S.N)
    with pytest.raises(ValueError):
        DeePC([lib], anchor_headings=[0.0], Q=S.Q, R=S.R,
              T_ini=S.T_INI - 1, N=S.N, solver=S.SOLVER)


def test_rejects_mismatched_anchor_count():
    libs = [build_hankel(u, y, T_ini=S.T_INI, N=S.N) for (u, y) in S.multi_libraries(2)]
    with pytest.raises(ValueError):
        DeePC(libs, anchor_headings=[0.0, 1.0, 2.0], Q=S.Q, R=S.R,
              T_ini=S.T_INI, N=S.N, solver=S.SOLVER)


def test_rejects_out_of_range_heading_index_multi():
    with pytest.raises(ValueError):
        _build_multi(n=2, heading_index=5)


# ---- API surface ------------------------------------------------------------


def test_act_before_reset_raises():
    c = _build_single()
    with pytest.raises(RuntimeError):
        c.act(np.array([0.0, 0.0, 0.0]), np.zeros(3))


def test_act_returns_action_of_correct_shape():
    c = _build_single()
    c.reset(np.array([0.0, 0.0, 0.1]))
    u_t = c.act(np.array([0.0, 0.0, 0.1]), np.array([1.0, 1.0, 0.0]))
    assert u_t.shape == (2,)
    assert np.isfinite(u_t).all()


def test_y_ref_per_step_horizon_accepted():
    c = _build_single()
    c.reset(np.array([0.0, 0.0, 0.1]))
    y_ref_window = np.tile(np.array([1.0, 1.0, 0.0]), (S.N, 1))
    u_t = c.act(np.array([0.0, 0.0, 0.1]), y_ref_window)
    assert u_t.shape == (2,)


def test_buffer_slides_after_act():
    c = _build_single()
    c.reset(np.array([0.0, 0.0, 0.1]))
    assert c._y_buf is not None
    np.testing.assert_array_equal(c._y_buf[-1], [0.0, 0.0, 0.1])
    c.act(np.array([1.0, 2.0, 0.2]), np.array([1.0, 1.0, 0.0]))
    np.testing.assert_array_equal(c._y_buf[-1], [1.0, 2.0, 0.2])
    c.act(np.array([3.0, 4.0, 0.3]), np.array([1.0, 1.0, 0.0]))
    np.testing.assert_array_equal(c._y_buf[-1], [3.0, 4.0, 0.3])
    np.testing.assert_array_equal(c._y_buf[-2], [1.0, 2.0, 0.2])


def test_last_library_idx_is_negative_before_first_act():
    c = _build_multi()
    assert c.last_library_idx == -1


# ---- Orientation routing ----------------------------------------------------


def test_select_index_routes_by_quadrant():
    c = _build_multi()
    assert c._select_index(0.1) == 0
    assert c._select_index(2.0) == 1
    assert c._select_index(-2.0) == 2
    assert c._select_index(-0.3) == 3


def test_select_index_wraps_around_pi():
    c = _build_multi()
    assert c._select_index(np.pi - 0.1) == 1
    assert c._select_index(-np.pi + 0.1) == 2


def test_select_index_single_library_always_zero():
    c = _build_single()
    assert c._select_index(123.4) == 0
    assert c._select_index(-50.0) == 0


# ---- Warm-start reset on library switch -------------------------------------


def test_warm_start_retained_within_library_and_cleared_on_switch():
    c = _build_multi()
    seq = S.multi_input_sequence()  # routing: 0,0,1,2,3,0,1,2
    c.reset(seq[0][0])
    flags = []
    for yc, yr in seq:
        c.act(yc, yr)
        flags.append(c.last_warm_started)
    assert flags[0] is False          # first solve: nothing to warm-start from
    assert flags[1] is True           # stayed in library 0
    assert flags[2] is False          # switched 0 -> 1, warm-start cleared
    assert flags[3] is False          # switched 1 -> 2
    assert flags[5] is False          # switched 3 -> 0


# ---- Closed-loop behavior on a simple LTI system ----------------------------


def test_single_library_closed_loop_drives_scalar_lti_to_reference():
    u, y = _scalar_lti(T=400, a=0.9, y0=5.0, seed=0)
    lib = build_hankel(u, y, T_ini=4, N=8)
    c = DeePC(
        [lib], anchor_headings=[0.0],
        Q=np.array([[1.0]]), R=np.array([[0.01]]),
        T_ini=4, N=8, lambda_g=1.0, lambda_y=1e4, solver=S.SOLVER,
    )
    y_cur = np.array([5.0])
    y_ref = np.array([0.0])
    c.reset(y_cur)
    for _ in range(50):
        u_t = c.act(y_cur, y_ref)
        y_cur = 0.9 * y_cur + u_t
    assert abs(y_cur[0]) < 0.5, f"final y={y_cur[0]:.3f} should be near 0"


def test_u_bounds_respected():
    u, y = _scalar_lti(T=200, a=0.9, y0=10.0, seed=1)
    lib = build_hankel(u, y, T_ini=3, N=5)
    u_min, u_max = np.array([-0.5]), np.array([0.5])
    c = DeePC(
        [lib], anchor_headings=[0.0],
        Q=np.eye(1), R=0.01 * np.eye(1),
        T_ini=3, N=5, lambda_g=1.0, lambda_y=1e4,
        u_bounds=(u_min, u_max), solver=S.SOLVER,
    )
    y_cur = np.array([10.0])
    c.reset(y_cur)
    for _ in range(20):
        u_t = c.act(y_cur, np.array([0.0]))
        assert u_t[0] >= u_min[0] - 1e-6
        assert u_t[0] <= u_max[0] + 1e-6
        y_cur = 0.9 * y_cur + u_t

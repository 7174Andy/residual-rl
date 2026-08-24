"""The linearity harness, checked against a plant whose answer is known exactly.

`scripts/test_local_linearity.py` concludes things about the Panda from the SHAPE
of an error-vs-radius curve, so the harness has to be trustworthy before the curve
means anything -- an earlier version of it reported a spurious error floor that
came entirely from its own scaling choice, not from the plant.

The check: run it on a synthetic LTI system, where `y_f = f(y_p, u_p, u_f)` is
exactly linear and `E_self` must therefore be ~0 at EVERY radius, with no
floor and no U-shape. Anything the harness adds on its own -- conditioning,
rank truncation, scaling, the past/future split in `pack` -- shows up here as a
nonzero error on a problem that has none.
"""
from __future__ import annotations

import importlib.util

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "local_linearity", "scripts/test_local_linearity.py")
tll = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tll)


def lti_sys(n=6, m=3, p=4, L=12, seed=0):
    """A stable random LTI plant wrapped in the harness's `Sys` interface."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)); A *= 0.9 / np.abs(np.linalg.eigvals(A)).max()
    B, C = rng.standard_normal((n, m)), rng.standard_normal((p, n))
    state = {"x": np.zeros(n)}

    def set_state(q, qd):
        state["x"] = np.concatenate([q, qd])

    def step(u):
        state["x"] = A @ state["x"] + B @ u
        return u                      # no clipping: the map stays globally linear

    return tll.Sys(
        None, None, np.zeros(n // 2), np.zeros(n - n // 2),
        rng.standard_normal((L, m)),
        src_scale=(1.0, 1.0, 1.0),
        step=step, read_y=lambda: C @ state["x"], set_state=set_state,
        y_scale=np.ones(p), u_scale=np.ones(m),
    )


def test_exactly_linear_plant_has_no_error_at_any_radius():
    T_ini, N = 4, 8
    s = lti_sys(L=T_ini + N)
    sz, sy = s.zy_scales(T_ini, N)
    rng = np.random.default_rng(1)
    for eps in (1.0, 1e-2, 1e-4, 1e-6):
        dz, dy, _ = tll.ensemble(s, eps, 200, rng, T_ini)
        J, _, _ = tll.fit_jacobian(dz[:150] / sz, dy[:150] / sy, s.src_dim)
        err = tll.rel_err(dy[150:] / sy, dz[150:] / sz, J)
        # A linear plant has no second-order remainder, so the only error here is
        # the harness's own arithmetic. Scale-invariance matters as much as the
        # magnitude: a floor would show up as this growing at small eps.
        assert err < 1e-8, f"eps={eps}: harness invents error {err:.2e} on an LTI plant"


def test_pack_splits_past_and_future_at_T_ini():
    """`z = [y_p; u_p; u_f]` and `y_f` -- the alignment the whole test depends on."""
    T_ini, N, p, m = 3, 5, 2, 4
    y = np.arange((T_ini + N) * p, dtype=float).reshape(T_ini + N, p)
    u = np.arange((T_ini + N) * m, dtype=float).reshape(T_ini + N, m) + 100
    z, yf = tll.pack(y, u, T_ini)
    assert z.shape == (T_ini * p + (T_ini + N) * m,)
    assert yf.shape == (N * p,)
    np.testing.assert_array_equal(z[:T_ini * p], y[:T_ini].ravel())
    np.testing.assert_array_equal(z[T_ini * p:], u.ravel())
    np.testing.assert_array_equal(yf, y[T_ini:].ravel())


def test_fit_jacobian_truncates_to_the_known_rank():
    """The rank argument must bind, and discard only numerical noise."""
    rng = np.random.default_rng(2)
    basis = rng.standard_normal((40, 10))          # 40 columns, rank 10
    dz = rng.standard_normal((200, 10)) @ basis.T
    dy = dz @ rng.standard_normal((40, 6))
    J, k, tail = tll.fit_jacobian(dz, dy, rank=10)
    assert k == 10
    assert tail < 1e-20
    assert tll.rel_err(dy, dz, J) < 1e-10

"""Is the DeePC *implementation* right, as distinct from the libraries being good?

The reach rate on `PandaReach-v0` is 0.487, and a number like that cannot by
itself distinguish "the controller is miswired" from "the controller is fine but
a Hankel library is a poor model of a 7-DoF arm". These tests pin down the first
half so the second can be argued about honestly. Each one is exact -- it has a
known answer that does not depend on the Panda being linear, well-conditioned, or
well-excited.

The complementary *model-adequacy* numbers are not asserted here because they are
properties of the data, not the code: measured 1-step tip prediction error is
0.017-0.041 m on episodes that reach and 0.23-0.52 m on episodes that fail, with
median |g|_1 of 2.8-5.1 vs 28-49. See the DeePC section of the journey docs.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.deepc import DeePC
from core.hankel import build_hankel
from panda import data_collection as dc
from panda import deepc_setup as ds

T_INI, N = 5, 12
M_U, P_Y = 7, 3


def _panda_deepc(hankel, delta_max, **kw):
    opts = dict(
        anchor_headings=np.array([0.0]), Q=np.eye(P_Y), R=1.0e-2 * np.eye(M_U),
        T_ini=T_INI, N=N, lambda_g=5e-2, lambda_y=7.5e4,
        u_bounds=(-delta_max * np.ones(M_U), delta_max * np.ones(M_U)),
        key_fn=ds.azimuth_key, solver="SCS",
    )
    opts.update(kw)
    return DeePC([hankel], **opts)


@pytest.mark.integration
def test_hankel_column_is_the_recorded_trajectory_segment():
    """Column j of each block must BE the data segment starting at index j.

    This is the check that catches a transposed or off-by-one Hankel, and it is
    the one failure mode that no amount of lambda tuning would reveal: a
    misaligned library still yields a solvable QP and plausible-looking actions,
    just wrong ones.
    """
    with np.load(dc.LIBRARIES_PATH) as z:
        n_lib = int(z["anchor_q1"].shape[0])
        for i in range(n_lib):
            u, y = z[f"u_{i}"], z[f"y_{i}"]
            Up, Uf, Yp, Yf = build_hankel(u, y, T_ini=T_INI, N=N)
            for j in (0, 7, Up.shape[1] // 2, Up.shape[1] - 1):
                e = np.zeros(Up.shape[1])
                e[j] = 1.0
                assert Up @ e == pytest.approx(u[j:j + T_INI].flatten(), abs=0)
                assert Yp @ e == pytest.approx(y[j:j + T_INI].flatten(), abs=0)
                assert Uf @ e == pytest.approx(
                    u[j + T_INI:j + T_INI + N].flatten(), abs=0)
                assert Yf @ e == pytest.approx(
                    y[j + T_INI:j + T_INI + N].flatten(), abs=0)


@pytest.mark.integration
def test_replay_of_collected_data_recovers_the_single_true_column():
    """Primed with a real past from the library, the QP must find exactly `e_j`.

    Feeding back a past that IS one of the Hankel columns, and asking for that
    column's own recorded continuation, makes `g = e_j` the exact optimum: it
    attains zero slack AND zero tracking error, so nothing can beat it. Hence
    `|g|_1 == 1` is the sharp signature of correct wiring.

    Do NOT check this with a least-squares residual on `[Up; Yp; Yf] g = b`. That
    system has 86 rows and 384 columns, so it is underdetermined and fits an
    arbitrary `b` to ~1e-15 -- verified against a deliberately y-shifted Hankel,
    which it accepts just as happily. It is a check that cannot fail.

    Mutation-tested. Against the correct library `|g|_1 = 1.00` and prediction
    error 2e-4; with `y` shifted one step against `u`, 5.31 and 3.2e-2; with the
    past buffer time-reversed, 13.11 and 1.2e-1. `sigma_y` alone separates these
    by only ~20x (7.5e-6 vs 2.3e-4), which is why it is the weakest assertion here
    rather than the headline one.
    """
    with np.load(dc.LIBRARIES_PATH) as z:
        u, y = z["u_0"], z["y_0"]
        delta_max = float(z["delta_max"])
    hankel = build_hankel(u, y, T_ini=T_INI, N=N)

    for k in (10, 140, 300):
        u_ini, y_ini = u[k:k + T_INI], y[k:k + T_INI]
        y_fut = y[k + T_INI:k + T_INI + N]

        d = _panda_deepc(hankel, delta_max)
        d.reset(y_ini[0], u_initial=u_ini[0])
        d._u_buf, d._y_buf = u_ini.copy(), y_ini.copy()
        d.act(y_ini[-1], y_fut)

        assert float(np.abs(d._g.value).sum()) < 1.05          # recovered e_j
        assert np.abs(d.last_pred_y - y_fut).max() < 1e-3
        assert d.last_sigma_y_norm < 1e-4


def test_prediction_is_exact_on_an_lti_plant_at_panda_dimensions():
    """On a plant that IS exactly LTI, `Yf g` must equal the true future.

    `core/` already has a scalar LTI closed-loop test; this one runs at the
    Panda's actual shapes (7 inputs, 3 outputs, T_ini=5, N=12), which is where a
    kron/reshape bug in the horizon stacking would surface. The plant is the
    Panda's own linear core -- `q_{t+1} = q_t + u_t` with `y = J q` -- but with a
    random J, so passing cannot come from accidentally matching the real arm.
    """
    rng = np.random.default_rng(0)
    J = rng.normal(size=(P_Y, M_U)) * 0.3
    delta_max, T = 0.2, 600
    u = rng.uniform(-delta_max, delta_max, size=(T, M_U))
    y = np.zeros((T, P_Y))
    q = np.zeros(M_U)
    for t in range(T):
        y[t] = J @ q          # y observed BEFORE u_t, matching data_collection
        q = q + u[t]

    d = _panda_deepc(build_hankel(u, y, T_ini=T_INI, N=N), delta_max,
                     R=1e-6 * np.eye(M_U), lambda_g=1e-8, lambda_y=1e8)
    k = 50
    u_ini, y_ini = u[k:k + T_INI], y[k:k + T_INI]
    target = y[k + T_INI:k + T_INI + N]
    d.reset(y_ini[0], u_initial=u_ini[0])
    d._u_buf, d._y_buf = u_ini.copy(), y_ini.copy()
    d.act(y_ini[-1], target)

    assert d.last_sigma_y_norm < 1e-5
    assert np.abs(d.last_pred_y - target).max() < 1e-4


def test_first_predicted_output_is_the_current_one_not_the_next():
    """Pin the horizon convention: `Yf g`'s first row aligns with `y_current`.

    Because `data_collection` records `y_t` BEFORE applying `u_t`, the first future
    output of a Hankel column is the observation that follows the last past input
    -- i.e. the CURRENT one at solve time, with `Uf g`'s first row being the action
    to apply. An implementation that shifted this by one step would still control,
    just against a reference one step stale. Measured off-by-one alignment error
    across four Panda episodes selected off both alignments to 0.
    """
    rng = np.random.default_rng(1)
    J = rng.normal(size=(P_Y, M_U)) * 0.3
    delta_max, T = 0.2, 400
    u = rng.uniform(-delta_max, delta_max, size=(T, M_U))
    y = np.zeros((T, P_Y))
    q = np.zeros(M_U)
    for t in range(T):
        y[t] = J @ q
        q = q + u[t]

    d = _panda_deepc(build_hankel(u, y, T_ini=T_INI, N=N), delta_max,
                     R=1e-6 * np.eye(M_U), lambda_g=1e-8, lambda_y=1e8)
    k = 40
    u_ini, y_ini = u[k:k + T_INI], y[k:k + T_INI]
    d.reset(y_ini[0], u_initial=u_ini[0])
    d._u_buf, d._y_buf = u_ini.copy(), y_ini.copy()
    d.act(y_ini[-1], y[k + T_INI:k + T_INI + N])

    y_current = y[k + T_INI - 1] + J @ u_ini[-1]   # == y[k + T_INI] for this plant
    assert d.last_pred_y[0] == pytest.approx(y[k + T_INI], abs=1e-4)
    assert d.last_pred_y[0] == pytest.approx(y_current, abs=1e-4)
    # ...and it is NOT the step after that, which is what the shifted variant
    # would produce (guard against the assertion above passing degenerately).
    assert np.abs(d.last_pred_y[0] - y[k + T_INI + 1]).max() > 1e-3

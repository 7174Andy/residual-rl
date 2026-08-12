"""Panda PE data collection.

Two things here are non-obvious and were established by measurement, not theory:
the anchors must be interior to joint 1's safe box (the unicycle's heading
anchors are not), and the excitation needs a restoring term or an integrated
random walk pins a joint against the box edge -- 41% of steps, which would
silently corrupt every (u, y) pair it touched.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.hankel import build_hankel
from panda import data_collection as dc
from panda.env import PandaReachEnv
from panda.model import safe_box


@pytest.fixture(scope="module")
def env():
    e = PandaReachEnv(max_steps=10**9)   # collection must never terminate
    yield e
    e.close()


def test_anchors_are_interior_to_joint1_safe_box(env):
    """The unicycle's (pi/4, 3pi/4, 5pi/4, 7pi/4) map to q1 = +-0.785, +-2.356.
    Joint 1's safe box is +-2.318, so two of those are OUTSIDE it."""
    lo, hi = safe_box(env.model)
    for q1 in dc.PANDA_ANCHOR_Q1:
        assert lo[0] < q1 < hi[0]
    unicycle_q1 = [float(((a + np.pi) % (2 * np.pi)) - np.pi)
                   for a in (np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4)]
    assert sum(1 for q in unicycle_q1 if not (lo[0] < q < hi[0])) == 2


def test_anchor_qpos_is_home_with_q1_replaced(env):
    q = dc.anchor_qpos(env.model, -1.8)
    assert q.shape == (7,)
    assert q[0] == pytest.approx(-1.8)
    assert np.allclose(q[1:], env.model.key_qpos[0][1:])


def test_collect_trajectory_shapes_and_alignment(env):
    """y[t] must be the tip BEFORE u[t] is applied -- the same convention as
    two_wheel_robot/controllers/data_collection.py."""
    out = dc.collect_trajectory(env, dc.anchor_qpos(env.model, -0.6), T=30,
                                rng=np.random.default_rng(0))
    assert out["u"].shape == (30, 7)
    assert out["ctrl"].shape == (30, 7)
    assert out["y"].shape == (30, 3)
    # Replay the first step and check y[0] was the pre-step tip.
    dc._reset_to_anchor(env, dc.anchor_qpos(env.model, -0.6))
    assert np.allclose(out["y"][0], env.y)


def test_recorded_u_within_delta_max_and_ctrl_within_safe_box(env):
    lo, hi = safe_box(env.model)
    out = dc.collect_trajectory(env, dc.anchor_qpos(env.model, 0.6), T=50,
                                rng=np.random.default_rng(1))
    assert np.all(np.abs(out["u"]) <= env.delta_max + 1e-12)
    assert np.all(out["ctrl"] >= lo - 1e-9) and np.all(out["ctrl"] <= hi + 1e-9)


@pytest.mark.parametrize("seed", [0, 2, 3, 5, 7])
def test_clip_fraction_is_exactly_zero_by_construction(env, seed):
    """`collect_trajectory` bounds u to [lo - q, hi - q] before stepping, so the
    env's safe-box clip can never fire and the recorded u can never lie to the
    Hankel. These five seeds measured 2.69/0.88/1.75/1.69/1.75% clip fraction
    (seeds 0/2/3/5/7 respectively) before that bound existed -- restoring alone
    (k_ret=0.05) is not reliable at the outer anchors (q1=+-1.8), which sit only
    0.518 rad from joint 1's +-2.318 edge. With the structural bound, clip_frac
    must be exactly 0.0, not merely small: any nonzero value here means the
    bound itself is broken, a code defect rather than a data property."""
    out = dc.collect_trajectory(env, dc.anchor_qpos(env.model, 1.8), T=400,
                                rng=np.random.default_rng(seed))
    assert out["clip_frac"] == 0.0


def test_excitation_is_deterministic_under_a_seeded_rng(env):
    a = dc.collect_trajectory(env, dc.anchor_qpos(env.model, -1.8), T=25,
                              rng=np.random.default_rng(7))
    b = dc.collect_trajectory(env, dc.anchor_qpos(env.model, -1.8), T=25,
                              rng=np.random.default_rng(7))
    assert np.allclose(a["u"], b["u"]) and np.allclose(a["y"], b["y"])


def test_coverage_report_meets_the_rank_floor(env):
    payload = dc.collect_libraries(env, T=400, rng=np.random.default_rng(3))
    rep = dc.coverage_report(payload)
    assert len(rep["libraries"]) == 4
    for lib in rep["libraries"]:
        assert lib["n_cols"] == 384
        assert lib["rank"] >= dc.RANK_FLOOR
    assert rep["clip_frac"] < 0.01
    # Anchor azimuths are MEASURED from FK, not assumed equal to q1.
    assert len(rep["anchor_azimuths"]) == 4
    assert len(set(np.round(rep["anchor_azimuths"], 3))) == 4


def test_hankel_blocks_have_the_expected_panda_shapes(env):
    payload = dc.collect_libraries(env, T=400, rng=np.random.default_rng(4))
    Up, Uf, Yp, Yf = build_hankel(payload["u_0"], payload["y_0"], T_ini=5, N=12)
    assert Up.shape == (35, 384)
    assert Uf.shape == (84, 384)
    assert Yp.shape == (15, 384)
    assert Yf.shape == (36, 384)

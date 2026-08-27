"""Goal-directed task-bank collection: servo one valid config to another.

Unlike panda/data_collection.py's anchor libraries (excite around a fixed
joint-1 anchor), this collects trajectories that actually traverse the
task manifold -- start at one valid configuration, servo toward another
(the goal's generating configuration) -- which is what a DeePC library
needs to contain if the controller is ever going to be asked to move
between configurations that far apart.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from panda.env import PandaReachEnv
from panda.task_bank import collect_task_bank, for_select_dpc, servo_trajectory


@pytest.fixture(scope="module")
def env():
    e = PandaReachEnv()
    yield e
    e.close()


def test_servo_moves_toward_goal_config(env):
    rng = np.random.default_rng(0)
    out = servo_trajectory(env, T=60, rng=rng, alpha=0.35, noise_sigma=0.0)
    # noiseless servo must strictly reduce joint distance to the goal config
    d0 = np.linalg.norm(out["q"][0] - out["q_goal"])
    d1 = np.linalg.norm(out["q"][-1] - out["q_goal"])
    assert d1 < 0.5 * d0


def test_ctrl_is_recorded_and_consistent(env):
    rng = np.random.default_rng(1)
    out = servo_trajectory(env, T=20, rng=rng, alpha=0.35, noise_sigma=0.04)
    assert out["u"].shape == (20, 7) and out["ctrl"].shape == (20, 7)
    assert out["y"].shape[1] == 3 and out["yext"].shape[1] == 10


def test_bank_schema_matches_collect_libraries(env):
    bank = collect_task_bank(env, n_traj=3, T=15, seed=0)
    for i in range(3):
        for k in ("u", "ctrl", "y", "yext", "q", "tip"):
            assert f"{k}_{i}" in bank
    assert bank["n_traj"] == 3
    assert bank["anchors"].shape == (3, 7)


def test_bank_feeds_the_anchor_hankel_pipeline(env):
    """The u_i/y_i/yext_i keys match data_collection.collect_libraries's
    schema -- verified against data/panda_libraries_v2.npz's own key list --
    which is what panda/deepc_setup.py::build_canonical_panda_deepc actually
    reads (np.load(...)['u_i']/['y_i'] straight into core.hankel.build_hankel).

    u_0 must be the DELTA env.step() takes (bounded by delta_max), not the
    absolute q_des Select-DPC's u channel uses -- a fix-round-1 regression
    check: a shape-only test cannot catch that semantic swap.
    """
    from core.hankel import build_hankel

    bank = collect_task_bank(env, n_traj=3, T=30, seed=0)
    assert np.all(np.abs(bank["u_0"]) <= env.delta_max + 1e-9)
    for key in ("y_0", "yext_0"):
        Up, Uf, Yp, Yf = build_hankel(bank["u_0"], bank[key], T_ini=5, N=12)
        assert Uf.shape[1] > 0  # non-empty column bank


def test_bank_feeds_panda_bank_as_run_select_dpc_does(env, tmp_path):
    """panda/selectdpc.py::panda_bank (called the way scripts/run_select_dpc.py
    and scripts/measure_selection_distance.py call it) reads u_i as an
    ABSOLUTE q_des target (panda/qdes.py::collect_anchor's schema), which
    collides with collect_task_bank's own u_i (the DELTA env.step() takes) --
    see panda/task_bank.py's module docstring. Feeding the raw payload
    straight into panda_bank corrupts its Willems regression (measured:
    selection skill -77.6 instead of the -1.83 an even-uncollected-for bank
    got). for_select_dpc() is the fix: it re-keys u_i from ctrl_i (the
    recorded absolute target).

    Replicates scripts/run_select_dpc.py's load/call sequence, through the
    adapter every real caller of panda_bank on this bank must also use:
    round-trip through an npz, `for_select_dpc(payload)`, then
    `panda_bank(..., T_ini, N, stride=...)`.
    """
    from panda.selectdpc import panda_bank

    bank = collect_task_bank(env, n_traj=3, T=30, seed=0)
    p = tmp_path / "bank.npz"
    np.savez(p, **bank)
    with np.load(p) as z:
        payload = {k: z[k] for k in z.files}
    sdpc_payload = for_select_dpc(payload)

    # Semantic check: Select-DPC's u must be an absolute joint target, close
    # to q (ctrl = clip(q + delta, safe_box), |delta| <= delta_max) -- NOT
    # the raw delta bank["u_0"] would be bounded to (+-0.2 rad), which is far
    # smaller than |q_0| typically is. This is what would have caught the
    # delta-vs-absolute swap: a shape/KeyError-only test cannot.
    u0, q0 = sdpc_payload["u_0"], sdpc_payload["q_0"]
    assert np.all(np.abs(u0 - q0) <= env.delta_max + 1e-9)

    anchors = sdpc_payload["anchors"]
    assert len(anchors) == 3
    tb = panda_bank(sdpc_payload, T_ini=5, N=12, stride=2)
    assert tb["Up"].shape[1] > 0  # non-empty column bank


def test_panda_bank_guard_trips_on_raw_task_bank_payload(env):
    """Round-2 fix: panda_bank's runtime guard must reject a raw
    collect_task_bank payload BEFORE pooling -- its u_i is a delta
    (|u| <= delta_max), not the absolute q_des panda_bank requires. This is
    the disk-path hazard the guard closes: a caller pointing panda_bank at
    the wrong npz should get a loud error, not a silently corrupted bank.
    """
    from panda.selectdpc import panda_bank

    bank = collect_task_bank(env, n_traj=2, T=15, seed=2)
    with pytest.raises(ValueError, match="for_select_dpc"):
        panda_bank(bank, T_ini=5, N=8)


def test_panda_bank_still_works_on_converted_and_legacy_payloads(env):
    """The guard must NOT false-positive on real absolute-u payloads: a
    task-bank payload run through for_select_dpc, and (if present on disk) an
    actual panda/qdes.py collection from a previous experiment.
    """
    from panda.selectdpc import panda_bank

    bank = collect_task_bank(env, n_traj=2, T=30, seed=2)
    converted = for_select_dpc(bank)
    tb = panda_bank(converted, T_ini=5, N=12, stride=2)
    assert tb["Up"].shape[1] > 0

    legacy_path = "data/panda_anchors_k4_libs.npz"
    if os.path.exists(legacy_path):
        with np.load(legacy_path) as z:
            legacy = {k: z[k] for k in z.files}
        tb2 = panda_bank(legacy, T_ini=5, N=12, stride=4)
        assert tb2["Up"].shape[1] > 0

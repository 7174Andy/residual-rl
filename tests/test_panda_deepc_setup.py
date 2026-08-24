"""The canonical Panda DeePC configuration.

The key is tip azimuth, atan2(y[1], y[0]) -- the structural analog of the
unicycle's heading key. Note it is a *function* of y, not a component, which is
why DeePC needed a key_fn hook.
"""
from __future__ import annotations

import numpy as np
import pytest

from panda import data_collection as dc
from panda import deepc_setup as ds
from panda.env import PandaReachEnv

LIB_V0 = "data/panda_libraries_v0.npz"
LIB_V1 = "data/panda_libraries_v1.npz"


def test_azimuth_key_is_atan2_of_the_first_two_components():
    assert ds.azimuth_key(np.array([1.0, 0.0, 9.0])) == pytest.approx(0.0)
    assert ds.azimuth_key(np.array([0.0, 1.0, 9.0])) == pytest.approx(np.pi / 2)
    assert ds.azimuth_key(np.array([-1.0, 0.0, 9.0])) == pytest.approx(np.pi)


def test_anchor_headings_come_from_azimuths_not_q1(tmp_path):
    """anchor_q1 and anchor_azimuths coincide in the real npz (max diff 1.1e-16,
    a property of the home pose), so the routing test above cannot tell which
    field is read. This one can: the two are made to disagree.
    """
    T, m_u, p_y, n_lib = 17, 7, 3, 2   # T >= T_ini(5) + N(12)
    payload = {f"u_{i}": np.zeros((T, m_u)) for i in range(n_lib)}
    payload.update({f"y_{i}": np.zeros((T, p_y)) for i in range(n_lib)})
    payload["anchor_q1"] = np.array([0.0, 0.0])          # wrong field, if read
    payload["anchor_azimuths"] = np.array([-1.0, 1.0])   # must be what is used
    payload["delta_max"] = np.asarray(0.2)
    path = tmp_path / "fake.npz"
    np.savez(path, **payload)

    d, _ = ds.build_canonical_panda_deepc(libraries_path=str(path))
    assert np.allclose(d.anchor_headings, [-1.0, 1.0])


@pytest.mark.integration
def test_built_controller_has_the_panda_shapes():
    d, info = ds.build_canonical_panda_deepc()
    assert d.m_u == 7
    assert d.p_y == 3
    assert d.T_ini == 5 and d.N == 12
    assert len(d.anchor_headings) == 4
    assert np.allclose(info["Q"], np.eye(3))
    assert np.allclose(info["R"], 1.0e-2 * np.eye(7))
    assert d.u_bounds[0] == pytest.approx(-0.2)
    assert d.u_bounds[1] == pytest.approx(0.2)


@pytest.mark.integration
def test_each_anchor_configuration_selects_its_own_library():
    """The whole point of azimuth keying: an arm parked at anchor i must route to
    library i."""
    d, info = ds.build_canonical_panda_deepc()
    env = PandaReachEnv(max_steps=10**9)
    try:
        for i, q1 in enumerate(dc.PANDA_ANCHOR_Q1):
            dc._reset_to_anchor(env, dc.anchor_qpos(env.model, q1))
            assert d._select_index_for(env.y) == i
    finally:
        env.close()


@pytest.mark.integration
def test_one_closed_loop_step_returns_a_bounded_action():
    d, info = ds.build_canonical_panda_deepc()
    env = PandaReachEnv()
    try:
        env.reset(seed=0)
        d.reset(env.y, u_initial=info["u_init"])
        u = d.act(env.y, env.y_ref)
        assert u.shape == (7,)
        assert np.all(np.abs(u) <= 0.2 + 1e-9)
        assert np.all(np.isfinite(u))
    finally:
        env.close()


def test_rejects_unknown_output_mode():
    with pytest.raises(ValueError, match="output must be"):
        ds.build_canonical_panda_deepc(output="joints")


@pytest.mark.integration
def test_ext_output_widens_p_y_but_keeps_the_cost_identical():
    """Q must be diag(I_3, 0_7): the extra outputs inform prediction, not the cost.

    If the q block ever picked up weight, DeePC would start trading tip error for
    joint-space error against a y_ref whose q block is arbitrary zeros -- it would
    drive the arm toward the middle of its safe box and the reach rate would drop
    for a reason no reach-rate number could explain.
    """
    tip, i_tip = ds.build_canonical_panda_deepc(libraries_path=LIB_V1, output="tip")
    ext, i_ext = ds.build_canonical_panda_deepc(libraries_path=LIB_V1, output="ext")

    assert (tip.p_y, ext.p_y) == (3, 10)
    assert np.array_equal(i_ext["Q"][:3, :3], i_tip["Q"])
    assert np.array_equal(i_ext["Q"][3:, :], np.zeros((7, 10)))
    assert np.array_equal(i_ext["Q"][:, 3:], np.zeros((10, 7)))
    assert np.array_equal(i_ext["R"], i_tip["R"])
    assert (i_tip["y_attr"], i_ext["y_attr"]) == ("y", "y_ext")
    assert (i_tip["y_ref_attr"], i_ext["y_ref_attr"]) == ("y_ref", "y_ref_ext")


@pytest.mark.integration
def test_ext_output_rejects_a_libraries_file_without_yext():
    """v0 predates the extended output; failing loudly beats silently using tips.

    Pinned to v0 explicitly, not to `dc.LIBRARIES_PATH` -- the default now points at
    a file that *has* `yext`, so reading the constant would make this vacuous.
    """
    with pytest.raises(KeyError, match="yext_0"):
        ds.build_canonical_panda_deepc(libraries_path=LIB_V0, output="ext")


def test_policy_reads_the_output_its_info_names():
    """Teeth for the routing: a policy built 'ext' must not read env.y.

    Recorded via a stand-in env, because pairing an ext-mode controller with the
    3-D `env.y` raises a shape error only sometimes -- and a controller reading the
    wrong-but-conformable output would just steer badly and look like bad tuning.
    """
    class SpyEnv:
        step_idx = 0
        y = np.array([0.5, 0.1, 0.6])
        y_ref = np.array([0.4, 0.0, 0.5])
        y_ext = np.concatenate([y, np.zeros(7)])
        y_ref_ext = np.concatenate([y_ref, np.zeros(7)])

        def __init__(self):
            self.read = []

        def __getattribute__(self, name):
            if name in ("y", "y_ref", "y_ext", "y_ref_ext"):
                object.__getattribute__(self, "read").append(name)
            return object.__getattribute__(self, name)

    class FakeDeePC:
        last_library_idx = 0

        def reset(self, *a, **kw):
            pass

        def act(self, y, y_ref):
            return np.zeros(7)

    for out, expect in (("tip", {"y", "y_ref"}), ("ext", {"y_ext", "y_ref_ext"})):
        env = SpyEnv()
        info = {"u_init": np.zeros(7),
                "y_attr": "y" if out == "tip" else "y_ext",
                "y_ref_attr": "y_ref" if out == "tip" else "y_ref_ext"}
        ds.DeePCPolicy(FakeDeePC(), info)(env)
        assert set(env.read) == expect, f"output={out} read {set(env.read)}"


def test_policy_defaults_to_tip_for_info_dicts_predating_the_ext_output():
    class FakeDeePC:
        last_library_idx = 0

        def reset(self, *a, **kw):
            pass

        def act(self, y, y_ref):
            return np.zeros(7)

    pol = ds.DeePCPolicy(FakeDeePC(), {"u_init": np.zeros(7)})
    assert (pol._y_attr, pol._y_ref_attr) == ("y", "y_ref")

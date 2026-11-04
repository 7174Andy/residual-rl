import numpy as np
import pytest

from core.deepc import DeePC
from quadruped import deepc_setup as ds
from quadruped.data_collection import collect_libraries
from quadruped.env import QuadrupedGoalEnv
from quadruped.scenarios import generate

INFO_KEYS = {"kind", "anchors", "u_init", "action_bounds", "Q", "R", "T_ini", "N",
             "lambda_g", "lambda_y", "affine", "du_max", "n_cols", "libraries_path"}
EP_KEYS = {"id", "reached", "fell", "truncated", "qp_fail", "steps", "final_dist", "mean_solve_s"}


@pytest.fixture(scope="module")
def libs(tmp_path_factory):
    p = tmp_path_factory.mktemp("lib") / "libs.npz"
    np.savez(p, **collect_libraries(T=200, seed=0))
    return str(p)


@pytest.fixture(scope="module")
def scen():
    return generate(n=2)


Y0 = np.array([0.1, -0.2, 0.7])
REF = np.array([1.0, 0.5, 0.3])


def test_info_and_bounds(libs):
    _, info = ds.build_canonical_quadruped_deepc(libs)
    assert set(info) == INFO_KEYS
    assert np.all(info["u_init"] == 0)
    env = QuadrupedGoalEnv()
    np.testing.assert_array_equal(info["R"], env.R)
    np.testing.assert_array_equal(info["Q"], env.Q)
    np.testing.assert_array_equal(info["action_bounds"], env.action_bounds)


def test_pure_function(libs):
    outs = []
    for _ in range(2):
        c, info = ds.build_canonical_quadruped_deepc(libs)
        c.reset(Y0, info["u_init"])
        outs.append([c.act(Y0 + 0.01 * k, REF) for k in range(3)])
    np.testing.assert_array_equal(outs[0], outs[1])


def test_branch_invariance(libs):
    c, info = ds.build_canonical_quadruped_deepc(libs)
    c.reset(Y0, info["u_init"])
    a = c.act(Y0, REF)
    shift = np.array([0, 0, 2 * np.pi])
    c2, _ = ds.build_canonical_quadruped_deepc(libs)
    c2.reset(Y0 + shift, info["u_init"])
    np.testing.assert_allclose(c2.act(Y0 + shift, REF), a, atol=1e-6)


def test_ref_near_pi(libs, monkeypatch):
    seen = {}
    monkeypatch.setattr(DeePC, "act", lambda self, y, r: seen.update(y=y, r=r) or np.zeros(3))
    c, info = ds.build_canonical_quadruped_deepc(libs)
    y = np.array([0.0, 0.0, 3.1])
    c.reset(y, info["u_init"])
    c.act(y, np.array([1.0, 1.0, -3.1]))
    assert abs(seen["r"][2] - 3.1) <= np.pi


def test_select_kind(libs):
    c, info = ds.build_canonical_quadruped_deepc(libs, kind="select", n_cols=50)
    assert isinstance(c, ds.QuadSelectDPC) and c.carry_prediction is False
    c.reset(Y0, info["u_init"])
    assert c.act(Y0, REF).shape == (3,)


def test_affine_du_passthrough(libs):
    c, info = ds.build_canonical_quadruped_deepc(libs, affine=True, du_scale=25.0)
    ab = info["action_bounds"]
    assert c.affine
    np.testing.assert_allclose(c.du_max, (ab[:, 1] - ab[:, 0]) / 25.0)


class _Stub:
    def __init__(self, fail_at=None):
        self.n, self.fail_at = 0, fail_at

    def reset(self, *a):
        pass

    def act(self, y, r):
        self.n += 1
        if self.n == self.fail_at:
            raise RuntimeError("qp")
        return np.zeros(3)


def test_run_episode_stub(scen):
    env = QuadrupedGoalEnv(max_steps=10)
    r = ds.run_episode(_Stub(), {"u_init": np.zeros(3)}, env, scen, 0)
    assert set(r) == EP_KEYS and not r["qp_fail"]
    assert r["steps"] == 10 or r["reached"] or r["fell"]


def test_run_episode_qp_fail(scen):
    env = QuadrupedGoalEnv(max_steps=10)
    r = ds.run_episode(_Stub(fail_at=3), {"u_init": np.zeros(3)}, env, scen, 0)
    assert r["qp_fail"] and r["steps"] == 2 and not r["reached"]


def test_select_crossing_stays_principal(libs):
    c, info = ds.build_canonical_quadruped_deepc(libs, kind="select", n_cols=50,
                                              T_ini=5, N=12)  # 50 cols < 3*T_ini at 20
    c.reset(np.array([0.0, 0.0, 3.0]), info["u_init"])
    for yaw in (3.1, -3.1, -2.9, -2.0, -1.0):
        c.act(np.array([0.0, 0.0, yaw]), REF)
        # stored yaw must not drift 2*pi off the bank's branch after a +-pi crossing
        assert abs(c._y_buf[-1, 2]) <= np.pi + 0.5


def test_defaults_are_winner(libs):
    _, info = ds.build_canonical_quadruped_deepc(libs)
    assert (info["kind"], info["T_ini"], info["N"], info["lambda_g"], info["lambda_y"],
            info["affine"], info["n_cols"]) == ("deepc", 20, 25, 2.0, 3e6, False, 300)
    np.testing.assert_allclose(info["du_max"], (ds._ACTION_BOUNDS[:, 1] - ds._ACTION_BOUNDS[:, 0]) / 25.0)


def test_deepc_crossing_keeps_buffer_contiguous(libs, monkeypatch):
    c, info = ds.build_canonical_quadruped_deepc(libs, T_ini=5, N=12)
    seen = []

    def fake_act(self, y, ref):  # record, then shift the past window like DeePC.act does
        seen.append((self._select_index_for(y), self._y_buf[:, 2].copy()))
        self._y_buf = np.roll(self._y_buf, -1, axis=0)
        self._y_buf[-1] = y
        return np.zeros(3)

    monkeypatch.setattr(DeePC, "act", fake_act)
    c.reset(np.array([0.0, 0.0, 2.9]), info["u_init"])
    for yaw in (3.0, 3.1, -3.1, -3.0, -2.9):
        c.act(np.array([0.0, 0.0, yaw]), REF)
    used = [k for k, _ in seen]
    assert used[0] != used[-1]  # +-pi is the boundary between the 3pi/4 and -3pi/4 libraries
    for _, buf in seen:
        assert np.max(np.abs(np.diff(buf))) < 0.5  # no 2*pi jump in the past window

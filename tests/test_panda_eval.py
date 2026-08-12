"""The rollout harness. Its job is that DeePC, a clone and a residual all produce
comparable rows for identical scenarios -- so `steps`, `effort` and `reached`
must be defined here once, not three times."""
from __future__ import annotations

import os

import numpy as np
import pytest

from panda import eval as pe
from panda import scenarios as sc
from panda.env import PandaReachEnv


@pytest.fixture(scope="module")
def scen():
    return sc.generate(n=4)


def _hold(env):
    return np.zeros(7)


def test_one_row_per_scenario_with_every_documented_column(scen):
    env = PandaReachEnv()
    rows = pe.run_scenarios(env, _hold, [0, 1], scen, method="test")
    env.close()
    assert len(rows) == 2
    for r in rows:
        assert set(r) == set(pe.RESULT_COLUMNS)
        assert r["method"] == "test"
    assert [r["scenario_id"] for r in rows] == [0, 1]


def test_reached_and_steps_agree_with_the_env(scen):
    env = PandaReachEnv()
    rows = pe.run_scenarios(env, _hold, [0], scen, method="test")
    env.close()
    r = rows[0]
    assert r["reached"] is False        # holding still cannot reach
    assert r["steps"] == 150            # so it truncates at max_steps


def test_min_dist_never_exceeds_final_dist(scen):
    env = PandaReachEnv()
    rows = pe.run_scenarios(env, _hold, [0, 1, 2, 3], scen, method="test")
    env.close()
    for r in rows:
        assert r["min_dist"] <= r["final_dist"] + 1e-12


def test_effort_is_zero_for_a_hold_policy_and_positive_otherwise(scen):
    env = PandaReachEnv()
    zero = pe.run_scenarios(env, _hold, [0], scen, method="a")[0]
    moving = pe.run_scenarios(env, lambda e: np.full(7, 0.1), [0], scen, method="b")[0]
    env.close()
    assert zero["effort"] == pytest.approx(0.0)
    assert moving["effort"] > 0.0


def test_append_results_preserves_earlier_methods(tmp_path, scen):
    p = str(tmp_path / "results.csv")
    env = PandaReachEnv()
    pe.append_results(pe.run_scenarios(env, _hold, [0], scen, method="deepc"), p)
    pe.append_results(pe.run_scenarios(env, _hold, [0], scen, method="clone"), p)
    env.close()
    got = pe.read_results(p)
    assert [r["method"] for r in got] == ["deepc", "clone"]
    assert len(got) == 2


def test_traces_are_written_only_for_requested_ids(tmp_path, scen):
    env = PandaReachEnv()
    pe.run_scenarios(env, _hold, [0, 1], scen, method="deepc",
                     trace_ids=(1,), trace_dir=str(tmp_path))
    env.close()
    import os
    written = sorted(os.listdir(os.path.join(str(tmp_path), "deepc")))
    assert written == ["traj_1.csv"]


def test_trace_columns_round_trip(tmp_path, scen):
    from core.trace_io import read_columns
    env = PandaReachEnv()
    pe.run_scenarios(env, _hold, [0], scen, method="deepc",
                     trace_ids=(0,), trace_dir=str(tmp_path))
    env.close()
    got = read_columns(os.path.join(str(tmp_path), "deepc", "traj_0.csv"))
    for name in ("t", "qpos_0", "qvel_6", "tip_x", "goal_z", "u_3", "ctrl_3",
                 "reward", "distance", "lib_idx", "qp_status"):
        assert name in got
    assert len(got["t"]) == 150


def test_library_hist_degrades_gracefully_without_last_library_idx(scen):
    """A plain function has no `last_library_idx` -- the shape the clone and
    residual stages will actually pass. `getattr(policy, ..., -1)` then
    records -1 every step, and both aggregates must degrade to "nothing to
    report", not crash or fabricate a fake library 0."""
    env = PandaReachEnv()
    rows = pe.run_scenarios(env, _hold, [0], scen, method="test")
    env.close()
    assert rows[0]["library_hist"] == ""
    assert rows[0]["n_switches"] == 0


def test_library_hist_and_switches_for_a_policy_that_reports_them(scen):
    class _AlternatingLibraryPolicy:
        def __init__(self):
            self.last_library_idx = 0

        def __call__(self, env):
            self.last_library_idx = 1 - self.last_library_idx
            return np.zeros(7)

    env = PandaReachEnv()
    rows = pe.run_scenarios(env, _AlternatingLibraryPolicy(), [0], scen, method="test")
    env.close()
    assert rows[0]["library_hist"] != ""
    assert rows[0]["n_switches"] > 0


def test_library_hist_excludes_failed_steps(scen):
    """A real DeePC-style policy assigns `last_library_idx` only as the last
    statement of a *successful* solve, and fails occasionally. The histogram
    must not credit a failed step to whatever library the previous
    successful solve happened to leave behind -- it must sum to
    `steps - qp_failures`, not to `steps`."""
    class _FlakyLibraryPolicy:
        def __init__(self):
            self.n = 0
            self.last_library_idx = 0

        def __call__(self, env):
            self.n += 1
            if self.n % 3 == 0:
                raise RuntimeError("simulated solver failure")
            self.last_library_idx = self.n % 2
            return np.zeros(7)

    env = PandaReachEnv()
    rows = pe.run_scenarios(env, _FlakyLibraryPolicy(), [0], scen, method="test")
    env.close()
    r = rows[0]
    assert r["qp_failures"] > 0
    hist_sum = sum(int(x) for x in r["library_hist"].split(";")) if r["library_hist"] else 0
    assert hist_sum == r["steps"] - r["qp_failures"]


def test_append_results_raises_on_mismatched_header(tmp_path, scen):
    p = str(tmp_path / "results.csv")
    with open(p, "w", newline="") as fh:
        fh.write("method,scenario_id\n")
        fh.write("deepc,0\n")
    env = PandaReachEnv()
    rows = pe.run_scenarios(env, _hold, [0], scen, method="clone")
    env.close()
    with pytest.raises(ValueError, match="header"):
        pe.append_results(rows, p)

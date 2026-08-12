"""The frozen scenario set: the thing that makes DeePC, clone and residual
numbers describe the same episodes.

Why materialize instead of calling reset(seed=k)? Because reset(seed=k) maps a
seed to a scenario *through the env's sampling code*, and that code already
changed once during this project (the min_start_goal_dist rejection loop was
reordered). Any such change silently remaps seed -> scenario, and two numbers
recorded weeks apart stop being comparable with nothing to signal it.
"""
from __future__ import annotations

import numpy as np
import pytest

from panda import scenarios as sc
from panda.env import PandaReachEnv
from panda.model import MIN_TIP_Z, TIP_RADIUS_RANGE, safe_box


@pytest.fixture(scope="module")
def data():
    return sc.generate(n=sc.N_SCENARIOS)


def test_shapes_and_keys(data):
    assert data["qpos"].shape == (78, 7)
    assert data["goal"].shape == (78, 3)
    assert data["seed"].shape == (78,)
    for k in ("delta_max", "goal_tolerance", "min_start_goal_dist", "max_steps", "frame_skip"):
        assert k in data


def test_generation_is_deterministic():
    a, b = sc.generate(n=6), sc.generate(n=6)
    assert np.array_equal(a["qpos"], b["qpos"])
    assert np.array_equal(a["goal"], b["goal"])


def test_every_scenario_satisfies_the_env_invariants(data):
    env = PandaReachEnv()
    lo, hi = safe_box(env.model)
    lo_r, hi_r = TIP_RADIUS_RANGE
    for i in range(78):
        q, g = data["qpos"][i], data["goal"][i]
        assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9)
        assert g[2] >= MIN_TIP_Z
        assert lo_r <= float(np.linalg.norm(g)) <= hi_r
    env.close()


def test_start_goal_distance_respected(data):
    env = PandaReachEnv()
    for i in range(78):
        sc.reset_to(env, data, i)
        assert np.linalg.norm(env.y - env.goal) >= env.min_start_goal_dist
    env.close()


def test_reset_to_reproduces_the_recorded_scenario(data):
    env = PandaReachEnv()
    for i in (0, 37, 77):
        sc.reset_to(env, data, i)
        assert np.allclose(env.data.qpos, data["qpos"][i])
        assert np.allclose(env.goal, data["goal"][i])
    env.close()


def test_subsets_are_nested():
    """SWEEP must be a strict PREFIX of EVAL, or a lambda chosen on SWEEP cannot
    be cross-checked inside the EVAL run."""
    assert list(sc.SWEEP_IDS) == list(sc.EVAL_IDS)[: len(list(sc.SWEEP_IDS))]
    assert set(sc.SHOWCASE_IDS) <= set(sc.EVAL_IDS)
    assert len(list(sc.SWEEP_IDS)) == 20
    assert len(list(sc.EVAL_IDS)) == 78


def test_save_load_round_trip(tmp_path, data):
    p = str(tmp_path / "s.npz")
    sc.save(p, data)
    got = sc.load(p)
    assert np.array_equal(got["qpos"], data["qpos"])
    assert np.array_equal(got["goal"], data["goal"])
    assert sc.checksum(got) == sc.checksum(data)


def test_checksum_is_sensitive(data):
    perturbed = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    perturbed["goal"][0, 0] += 1e-9
    assert sc.checksum(perturbed) != sc.checksum(data)


def test_validate_against_env_passes_for_default_env(data):
    env = PandaReachEnv()
    sc.validate_against_env(env, data)  # must not raise
    env.close()


def test_validate_against_env_raises_naming_the_parameter(data):
    env = PandaReachEnv(max_steps=99)
    with pytest.raises(ValueError, match="max_steps"):
        sc.validate_against_env(env, data)
    env.close()


def test_validate_against_env_reports_multiple_mismatches(data):
    env = PandaReachEnv(max_steps=99, goal_tolerance=0.5)
    with pytest.raises(ValueError) as excinfo:
        sc.validate_against_env(env, data)
    assert "max_steps" in str(excinfo.value)
    assert "goal_tolerance" in str(excinfo.value)
    env.close()


# The freeze. Paste the checksum printed by scripts/make_panda_scenarios.py.
# If this test fails, data/panda_scenarios_v1.npz changed -- which invalidates
# every result already recorded against it. Do not update this constant to make
# the test pass; bump to v2 instead.
FROZEN_CHECKSUM = "995644eb72a8e96206e0bda95c89ee381941ac27f0732b64c5735c8f63d5e229"


@pytest.mark.integration
def test_committed_scenario_file_matches_the_frozen_checksum():
    assert sc.checksum(sc.load()) == FROZEN_CHECKSUM

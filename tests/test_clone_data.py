# tests/test_clone_data.py
"""Hybrid clone-dataset generation: shapes, both regimes present, determinism."""
from __future__ import annotations

import numpy as np
import pytest

from two_wheel_robot.rl.deepc_setup import build_canonical_deepc
from two_wheel_robot.rl import clone_data


@pytest.fixture(scope="module")
def setup():
    try:
        return build_canonical_deepc()
    except FileNotFoundError:
        pytest.skip("data/libraries_v0.npz not present")


def test_synthetic_config_shapes(setup):
    deepc, info = setup
    rng = np.random.default_rng(0)
    u_ini, y_ini, y_current, goal = clone_data.make_synthetic_config(
        rng, info["action_bounds"], dt=0.025, T_ini=info["T_ini"], degenerate=False
    )
    assert u_ini.shape == (info["T_ini"], 2)
    assert y_ini.shape == (info["T_ini"], 3)
    assert y_current.shape == (3,)
    assert goal.shape == (2,)


def test_degenerate_config_is_constant(setup):
    deepc, info = setup
    rng = np.random.default_rng(0)
    u_ini, y_ini, y_current, goal = clone_data.make_synthetic_config(
        rng, info["action_bounds"], dt=0.025, T_ini=info["T_ini"], degenerate=True
    )
    # A degenerate past holds still: all buffered y rows are identical.
    assert np.allclose(y_ini - y_ini[0], 0.0)


def test_generate_dataset_has_both_regimes_and_right_shapes(setup):
    deepc, info = setup
    ds = clone_data.generate_clone_dataset(
        deepc, info, n_synthetic=40, p_degenerate=0.5,
        n_onpolicy_episodes=2, seed=123, max_steps=30,
    )
    n = ds["features"].shape[0]
    expected_dim = 6 * info["T_ini"] + 6 + len(info["anchors"])
    assert n > 0
    assert ds["features"].shape == (n, expected_dim)
    assert ds["targets"].shape == (n, 2)
    assert ds["library_idx"].shape == (n,)
    assert ds["regime"].shape == (n,)
    regimes = set(np.unique(ds["regime"]).tolist())
    assert "synthetic" in regimes and "degenerate" in regimes and "onpolicy" in regimes


def test_generation_is_deterministic(setup):
    deepc, info = setup
    a = clone_data.generate_clone_dataset(
        deepc, info, n_synthetic=20, p_degenerate=0.5,
        n_onpolicy_episodes=1, seed=7, max_steps=20,
    )
    deepc2, info2 = build_canonical_deepc()
    b = clone_data.generate_clone_dataset(
        deepc2, info2, n_synthetic=20, p_degenerate=0.5,
        n_onpolicy_episodes=1, seed=7, max_steps=20,
    )
    assert np.allclose(a["features"], b["features"])
    assert np.allclose(a["targets"], b["targets"])

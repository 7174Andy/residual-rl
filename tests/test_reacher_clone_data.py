"""Clone-data collection: buffer alignment and dataset shape.

The alignment test is the one that matters. `y_t` must be recorded BEFORE `u_t`
is applied -- the convention every collection in this repo uses. A one-step shift
trains the clone on a map that does not exist and fails silently.
"""
from __future__ import annotations

import numpy as np
import pytest

import reacher  # noqa: F401  registers the Gym ID
from reacher.clone_features import feature_dim

# Deliberately tiny throughout: these tests check plumbing, not statistics. The
# real dataset comes from scripts/gen_reacher_clone_data.py.
SMALL = dict(grid=(3, 3), T=200, n_cols=60, stride=4)


@pytest.fixture(scope="module")
def dataset():
    from reacher.clone_data import generate_clone_dataset
    return generate_clone_dataset(n_episodes=2, seed=0, **SMALL)


def test_shapes(dataset):
    n = dataset["features"].shape[0]
    assert dataset["features"].shape == (n, feature_dim(5))
    assert dataset["actions"].shape == (n, 2)
    assert n > 0


def test_actions_are_inside_the_torque_box(dataset):
    assert np.all(np.abs(dataset["actions"]) <= 1.0 + 1e-9)


def test_meta_records_provenance(dataset):
    meta = dataset["meta"]
    for key in ("T_ini", "N", "n_cols", "n_max", "grid", "n_episodes",
                "n_dropped", "n_reached", "seed", "bank_columns", "T", "stride"):
        assert key in meta, f"meta is missing {key}"


def test_rows_per_episode_equal_the_full_horizon(dataset):
    """Episodes must NOT stop at first reach: the dataset has to contain the
    station-keeping regime, because that is what the residual is asked to
    improve. 2 episodes x 50 steps = 100 rows, exactly."""
    import gymnasium as gym
    e = gym.make("ReacherGoal-v0")
    horizon = e.unwrapped.max_steps
    e.close()
    n_ep = dataset["meta"]["n_episodes"] - dataset["meta"]["n_dropped"]
    assert dataset["features"].shape[0] == n_ep * horizon


def test_buffer_alignment_matches_deepc_convention():
    """Replay one episode by hand and assert the recorded feature at step 0 was
    built from the measurement taken BEFORE step 0's action was applied."""
    import gymnasium as gym

    from reacher.clone_data import build_bank, build_select_controller, rollout
    from reacher.clone_features import expand_y
    from reacher.model import load_model

    model, data = load_model()
    rng = np.random.default_rng(0)
    bank, _ = build_bank(model, data, rng, grid=SMALL["grid"], T=SMALL["T"],
                         stride=SMALL["stride"])
    ctrl = build_select_controller(bank, n_cols=SMALL["n_cols"], n_max=1)

    env = gym.make("ReacherGoal-v0")
    rec = rollout(env, ctrl, seed=0, T_ini=5)
    env.close()

    # Feature block [35:40] is the expanded CURRENT y. At step 0 it must equal
    # the expansion of the RESET measurement, not of the post-step one.
    env2 = gym.make("ReacherGoal-v0")
    env2.reset(seed=0)
    y0 = env2.unwrapped.y
    env2.close()
    assert np.allclose(rec["features"][0][35:40], expand_y(y0))

    # And the step-0 past buffer must be the priming fill, not real history:
    # u_ini all zeros, y_ini all copies of y0.
    assert np.allclose(rec["features"][0][0:10], 0.0)
    assert np.allclose(rec["features"][0][10:35], np.tile(expand_y(y0), 5))
    assert rec["features"][0][42] == 0.0        # validity: pure priming


def test_step_one_buffer_holds_step_zero_measurement_and_action():
    """The slide itself: at step 1 the newest buffer entry must be step 0's
    PRE-step y and step 0's APPLIED action -- an off-by-one here is the silent
    failure this module's docstring warns about."""
    import gymnasium as gym

    from reacher.clone_data import build_bank, build_select_controller, rollout
    from reacher.clone_features import expand_y
    from reacher.model import load_model

    model, data = load_model()
    rng = np.random.default_rng(0)
    bank, _ = build_bank(model, data, rng, grid=SMALL["grid"], T=SMALL["T"],
                         stride=SMALL["stride"])
    ctrl = build_select_controller(bank, n_cols=SMALL["n_cols"], n_max=1)

    env = gym.make("ReacherGoal-v0")
    rec = rollout(env, ctrl, seed=0, T_ini=5)
    env.close()

    env2 = gym.make("ReacherGoal-v0")
    env2.reset(seed=0)
    y0 = env2.unwrapped.y
    env2.close()

    # y_ini's LAST row at step 1 == y0 (the pre-step measurement of step 0).
    y_ini_rows = rec["features"][1][10:35].reshape(5, 5)
    assert np.allclose(y_ini_rows[-1], expand_y(y0))
    # u_ini's LAST row at step 1 == the action recorded at step 0.
    u_ini_rows = rec["features"][1][0:10].reshape(5, 2)
    assert np.allclose(u_ini_rows[-1], rec["actions"][0], atol=1e-9)

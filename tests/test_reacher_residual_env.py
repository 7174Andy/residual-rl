"""The zero-residual invariant is the no-regression floor. If it does not hold
bit-for-bit, "the residual can only improve on the clone" is not true."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import reacher  # noqa: F401

CLONE = "data/reacher_clone_600.pt"

def _skip_if_stale(path):
    """Skip with a legible reason if the checkpoint predates the feature change.

    The 43rd feature (buffer validity) widened `featurize`'s output, so a clone
    trained before it raises a broadcast error deep inside `ClonePredictor`
    rather than saying what is wrong. `data/` is gitignored, so stale checkpoints
    linger locally and this is a real trap, not a hypothetical.
    """
    import torch

    from reacher.clone_features import feature_dim
    want = feature_dim(5)
    try:
        got = int(torch.load(path, map_location="cpu",
                             weights_only=False)["stats"]["input_dim"])
    except FileNotFoundError:
        pytest.skip(f"{path} not trained; run scripts/train_reacher_clone.py")
    if got != want:
        pytest.skip(f"{path} has input_dim {got}, features are now {want}-D "
                    f"(stale checkpoint -- retrain on a 43-D dataset)")



@pytest.fixture(scope="module")
def res_env():
    from reacher.residual_env import ResidualSelectEnv
    _skip_if_stale(CLONE)
    e = ResidualSelectEnv(clone_path=CLONE)
    yield e
    e.close()


def test_spaces(res_env):
    assert res_env.action_space.shape == (2,)
    assert res_env.observation_space.shape == (10,)
    assert np.allclose(res_env.observation_space.low, -1.0)
    assert np.allclose(res_env.observation_space.high, 1.0)


def test_zero_residual_reproduces_the_clone_bit_for_bit(res_env):
    """Roll the residual env with action 0 and the raw clone policy on the same
    seed; the trajectories must be IDENTICAL, not merely close. `array_equal`,
    not `allclose` -- a drifting buffer convention shows up as a tiny difference
    that grows, and `allclose` would hide the first few steps of it."""
    from reacher.eval import ClonePolicy
    from rl.clone import load_clone

    res_env.reset(seed=7)
    res_traj = []
    for _ in range(20):
        _o, _r, _t, _tr, info = res_env.step(np.zeros(2, dtype=np.float32))
        res_traj.append(info["y"].copy())

    env = gym.make("ReacherGoal-v0")
    _o, info = env.reset(seed=7)
    policy = ClonePolicy(load_clone(CLONE, device="cpu"))
    clone_traj = []
    for _ in range(20):
        _o, _r, _t, _tr, info = env.step(policy(env, info))
        clone_traj.append(info["y"].copy())
    env.close()

    assert np.array_equal(np.array(res_traj), np.array(clone_traj))


def test_residual_saturation_stays_in_the_torque_box(res_env):
    for sign in (+1.0, -1.0):
        res_env.reset(seed=3)
        for _ in range(10):
            _o, _r, _t, _tr, info = res_env.step(
                sign * np.ones(2, dtype=np.float32))
            assert np.all(np.abs(info["action"]) <= 1.0 + 1e-9)


def test_u_base_is_cached_and_appears_in_the_observation(res_env):
    obs, _info = res_env.reset(seed=5)
    assert res_env.u_base is not None
    # u_base occupies the last two slots; the box is [-1,1] and the raw bounds
    # for that block are also [-1,1], so normalization is the identity there.
    assert np.allclose(obs[-2:], res_env.u_base, atol=1e-6)


def test_residual_frac_scales_the_correction(res_env):
    """`residual_frac` is the fraction of full authority the policy may add. At
    0.0 the residual must be inert even for a saturated action."""
    from reacher.residual_env import ResidualSelectEnv

    inert = ResidualSelectEnv(clone_path=CLONE, residual_frac=0.0)
    inert.reset(seed=11)
    u_base_first = inert.u_base.copy()
    _o, _r, _t, _tr, info = inert.step(np.ones(2, dtype=np.float32))
    inert.close()
    assert np.allclose(info["action"], u_base_first, atol=1e-12)

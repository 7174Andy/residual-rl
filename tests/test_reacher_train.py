"""Training plumbing: the zero-init contract, on a budget small enough for CI."""
from __future__ import annotations

import numpy as np
import pytest

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



def test_zero_init_makes_the_initial_residual_exactly_zero():
    """The no-regression floor: at step 0 the policy IS the clone.

    `zero_init_actor` zeroes the actor's mean head, and `tanh(0) == 0`, so a
    deterministic prediction must be exactly the zero residual. Without this the
    residual starts by perturbing a known-good baseline in a random direction.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    from rl.sb3 import build_model, zero_init_actor

    _skip_if_stale(CLONE)
    from reacher.residual_env import ResidualSelectEnv
    env = ResidualSelectEnv(clone_path=CLONE)

    venv = DummyVecEnv([lambda: env])
    model = build_model("sac", venv, 3e-4, "cpu", 0, 0, 0.1)
    zero_init_actor(model)
    obs = venv.reset()
    action, _ = model.predict(obs, deterministic=True)
    venv.close()
    assert np.allclose(action, 0.0, atol=1e-6)


def test_without_zero_init_the_initial_residual_is_generally_nonzero():
    """The control for the test above: if a fresh actor also emitted zero, the
    zero-init assertion would be vacuous."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    from rl.sb3 import build_model

    _skip_if_stale(CLONE)
    from reacher.residual_env import ResidualSelectEnv
    env = ResidualSelectEnv(clone_path=CLONE)

    venv = DummyVecEnv([lambda: env])
    model = build_model("sac", venv, 3e-4, "cpu", 0, 0, 0.1)
    obs = venv.reset()
    action, _ = model.predict(obs, deterministic=True)
    venv.close()
    assert not np.allclose(action, 0.0, atol=1e-6)

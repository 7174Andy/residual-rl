"""The two invariants the Panda RL arms rest on.

* Residual: a zero residual reproduces the clone's closed loop (`ClonePolicy`).
* Warm start: the transferred actor's deterministic action IS the squash clone.
"""
import numpy as np
import pytest

from panda.rl_env import PandaResidualEnv, PandaWindowEnv

CLONE = "data/panda_clone_box_r2.pt"
SQUASH = "data/panda_clone_box_r2_squash.pt"


def _need(path):
    import os
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")


def test_zero_residual_is_the_clone():
    _need(CLONE)
    from scripts.validate_panda_clone import ClonePolicy

    env = PandaResidualEnv(CLONE)
    env.reset(seed=3)
    ref = PandaResidualEnv(CLONE)          # an independent copy driven by ClonePolicy
    ref.reset(seed=3)
    pol = ClonePolicy(ref.pred, 5, ref.dmax)
    for _ in range(30):
        d_ref = pol(ref.base)
        _, _, term, trunc, _ = env.step(np.zeros(7))
        ref.base.step(d_ref)
        np.testing.assert_allclose(env.base.data.qpos, ref.base.data.qpos, atol=1e-10)
        if term or trunc:
            break


def test_warm_actor_is_the_squash_clone():
    _need(SQUASH)
    from stable_baselines3.common.vec_env import DummyVecEnv

    from rl.clone import load_clone
    from rl.sb3 import build_model, init_actor_from_clone

    env = PandaWindowEnv()
    obs, _ = env.reset(seed=5)
    model = build_model("sac", DummyVecEnv([PandaWindowEnv]), 3e-4, "cpu", 0, 0, 0.1)
    init_actor_from_clone(model, SQUASH)
    a, _ = model.predict(obs, deterministic=True)
    pred = load_clone(SQUASH)
    np.testing.assert_allclose(env.to_delta(a),
                               pred.predict(env.window.features()) * env.dmax, atol=1e-4)

"""The anneal must actually move `target_entropy`, and only inside its window.

Driven without `learn()`: `AnnealTargetEntropyCallback` reads nothing but
`model.num_timesteps`, so the ramp is testable by setting that attribute
directly — the same way SB3's `on_step` would see it mid-run.
"""
from __future__ import annotations

import pytest

import reacher  # noqa: F401  registers the Gym ID


def _sac_with_callback(start=100, total=200, final=-6.0):
    from stable_baselines3 import SAC

    from reacher.window_env import CloneWindowEnv
    from rl.sb3 import AnnealTargetEntropyCallback

    env = CloneWindowEnv()
    model = SAC("MlpPolicy", env, policy_kwargs=dict(net_arch=[32]),
                device="cpu", seed=0, verbose=0)
    cb = AnnealTargetEntropyCallback(start, total, final)
    cb.init_callback(model)
    cb.on_training_start(locals_={}, globals_={})
    return model, cb, env


def _step_at(model, cb, t):
    model.num_timesteps = t
    cb.on_step()
    return float(model.target_entropy)


def test_ramp_is_flat_then_linear_then_clamped():
    model, cb, env = _sac_with_callback(start=100, total=200, final=-6.0)
    try:
        initial = float(model.target_entropy)
        assert initial == pytest.approx(-2.0)  # SAC's auto = -dim(A)

        assert _step_at(model, cb, 50) == pytest.approx(initial)   # before
        assert _step_at(model, cb, 100) == pytest.approx(initial)  # boundary
        assert _step_at(model, cb, 150) == pytest.approx(-4.0)     # midpoint
        assert _step_at(model, cb, 200) == pytest.approx(-6.0)     # end
        assert _step_at(model, cb, 999) == pytest.approx(-6.0)     # clamped
    finally:
        env.close()


def test_alpha_loss_sees_the_mutated_target():
    # The mechanism the callback relies on: SAC.train() reads
    # `self.target_entropy` each gradient step. Guard the attribute name so an
    # SB3 rename fails here rather than silently disabling the anneal.
    model, cb, env = _sac_with_callback()
    try:
        import inspect

        from stable_baselines3 import SAC
        assert "target_entropy" in inspect.getsource(SAC.train)
    finally:
        env.close()

"""Gymnasium wrappers that prep TwoWheelGoal for RL training.

The base env (`two_wheel_robot.env.UnicycleGoalEnv`) keeps actions in physical
units so classical controllers can talk to it in their natural language. SB3 and
most actor-critic libraries train more reliably with a symmetric, unit-bounded
action space — this module provides that adaptation.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RescaleAction, RescaleObservation

import two_wheel_robot.env  # noqa: F401  registers Gym ID


def rescale_action_symmetric(env: gym.Env) -> gym.Env:
    """Wrap `env` so its action space is `Box([-1, -1], [1, 1])`.

    Actions sampled in `[-1, 1]^d` are linearly mapped to the underlying env's
    native bounds before being passed to `step`. Suitable for SB3 PPO / SAC / TD3
    out of the box.
    """
    return RescaleAction(env, min_action=np.float32(-1.0), max_action=np.float32(1.0))


def vanilla_rl_env(action_bounds, render_mode: str | None = None, perturb=None) -> gym.Env:
    """Raw `TwoWheelGoal-v0`, normalized for from-scratch RL. No controller involved.

    The agent learns the whole policy against the env's own DeePC-form reward, and
    emits `u = (v, w)` in physical units over `action_bounds` — pass the canonical
    DeePC bounds and the action space *is* DeePC's `u_bounds`. No `RescaleAction`
    here: SB3's off-policy algos already normalize a non-symmetric Box internally
    (`policy.scale_action` / `unscale_action`) for the actor head, action noise and
    replay buffer, so wrapping would only hide `(v, w)` behind a second identical
    affine map.

    The 5-D body observation *is* min-max normalized to `[-1, 1]` — obs bounds are
    not a physical contract the way action bounds are, and this matches what the
    residual policy network sees (minus its `u_base` block).

    `perturb` (optional): callable wrapping the raw env, for robustness benchmarking
    (see `env/perturbations.py`). Applied *inside* the obs normalization so the
    observation space is unchanged and the policy needs no retraining to be evaluated.
    """
    env = gym.make("TwoWheelGoal-v0", action_bounds=action_bounds, render_mode=render_mode)
    if perturb is not None:
        env = perturb(env)
    return RescaleObservation(env, min_obs=np.float32(-1.0), max_obs=np.float32(1.0))

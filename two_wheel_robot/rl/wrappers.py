"""Gymnasium wrappers that prep TwoWheelGoal for RL training.

The base env (`two_wheel_robot.env.UnicycleGoalEnv`) keeps actions in physical
units so classical controllers can talk to it in their natural language. SB3 and
most actor-critic libraries train more reliably with a symmetric, unit-bounded
action space — this module provides that adaptation.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RescaleAction


def rescale_action_symmetric(env: gym.Env) -> gym.Env:
    """Wrap `env` so its action space is `Box([-1, -1], [1, 1])`.

    Actions sampled in `[-1, 1]^d` are linearly mapped to the underlying env's
    native bounds before being passed to `step`. Suitable for SB3 PPO / SAC / TD3
    out of the box.
    """
    return RescaleAction(env, min_action=np.float32(-1.0), max_action=np.float32(1.0))

"""Franka Emika Panda end-effector reaching environment.

Importing this package registers Gym ID `PandaReach-v0`.
"""

from gymnasium.envs.registration import register

from .env import PandaReachEnv

register(
    id="PandaReach-v0",
    entry_point="panda.env:PandaReachEnv",
    # The env handles its own truncation via `max_steps`; don't let Gym stack a
    # second TimeLimit on top. Matches two_wheel_robot/env/__init__.py.
    max_episode_steps=None,
)

__all__ = ["PandaReachEnv"]

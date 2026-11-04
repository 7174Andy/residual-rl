"""Unitree Go1 goal-reaching environment.

Importing this package registers Gym ID `QuadrupedGoal-v0`.
"""

from gymnasium.envs.registration import register

from .env import QuadrupedGoalEnv

register(
    id="QuadrupedGoal-v0",
    entry_point="quadruped.env:QuadrupedGoalEnv",
    # The env handles its own truncation via `max_steps`; no second TimeLimit.
    max_episode_steps=None,
)

__all__ = ["QuadrupedGoalEnv"]

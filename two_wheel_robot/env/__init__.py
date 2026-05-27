"""Two-wheel robot goal-reaching environment.

Importing this package registers Gym ID `TwoWheelGoal-v0`.
"""

from gymnasium.envs.registration import register

from .env import UnicycleGoalEnv

register(
    id="TwoWheelGoal-v0",
    entry_point="two_wheel_robot.env.env:UnicycleGoalEnv",
    # The env handles its own truncation via `max_steps`, so don't let Gym
    # impose an additional TimeLimit on top.
    max_episode_steps=None,
)

__all__ = ["UnicycleGoalEnv"]

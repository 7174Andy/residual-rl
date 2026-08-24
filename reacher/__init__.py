"""Gymnasium `Reacher-v5` (2-link planar arm) as a tractable control for the Panda.

2 DoF, no redundancy, no gravity term, direct torque, and a 2-D configuration
space -- so the local-library DeePC hypothesis can be tested at a data budget of
tens of trajectories rather than the ~10^5 the Panda's ~5.7-dimensional
configuration set demands.

Importing this package registers Gym ID `ReacherGoal-v0`.
"""

from gymnasium.envs.registration import register

from .env import ReacherGoalEnv

register(
    id="ReacherGoal-v0",
    entry_point="reacher.env:ReacherGoalEnv",
    # The env truncates itself via `max_steps`; a Gym TimeLimit on top would
    # truncate twice. Matches panda/__init__.py.
    max_episode_steps=None,
)

__all__ = ["ReacherGoalEnv"]

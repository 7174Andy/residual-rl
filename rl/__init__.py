"""System-agnostic RL machinery: the behavioral clone, torch device selection,
stable-baselines3 plumbing, and the paired statistics.

This is the RL counterpart of `core/`: it knows nothing about any particular
robot, so all three systems (`two_wheel_robot/`, `panda/`, `reacher/`) share it.
Per CLAUDE.md this package is the ONLY place that may import torch or
stable_baselines3, and it imports no `gymnasium`, no `mujoco`, and nothing from
the system packages.
"""

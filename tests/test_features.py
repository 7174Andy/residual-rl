"""Tests for the unicycle clone's robot-centred feature transform."""

from __future__ import annotations

import numpy as np

from two_wheel_robot.rl.features import egocentric, featurize

ANCHORS = np.array([0.785, 2.356, -2.356, -0.785])


def _scenario(rng, theta=0.0, shift=(0.0, 0.0)):
    """One random window, optionally rotated by `theta` about the origin and shifted."""
    u = rng.uniform([0, -1.5], [20, 1.5], size=(5, 2))
    y = np.c_[rng.uniform(-5, 5, size=(6, 2)), rng.uniform(-np.pi, np.pi, size=6)]
    goal = rng.uniform(-5, 5, size=2)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    y[:, :2] = y[:, :2] @ R.T + shift
    y[:, 2] += theta
    goal = R @ goal + shift
    return featurize(u, y[:5], y[5], np.r_[goal, 0.0], ANCHORS)


def test_egocentric_invariant_to_world_rotation_and_translation():
    a = egocentric(_scenario(np.random.default_rng(0)), T_ini=5)
    b = egocentric(_scenario(np.random.default_rng(0), theta=1.1, shift=(3.0, -7.0)), T_ini=5)
    # The library one-hot keys on absolute heading by design, so it is excluded.
    np.testing.assert_allclose(a[:-4], b[:-4], atol=1e-9)


def test_egocentric_puts_robot_at_origin_facing_forward():
    f = egocentric(_scenario(np.random.default_rng(1)), T_ini=5)
    np.testing.assert_allclose(f[30:34], [0.0, 0.0, 0.0, 1.0])

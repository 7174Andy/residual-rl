"""Pure-numpy kinematic unicycle dynamics. No Gym dependency."""

from __future__ import annotations

import numpy as np


def step_unicycle(state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
    """One forward-Euler step of the kinematic unicycle.

    state: (x, y, delta) — position and heading (radians).
    action: (v, w) — tangential and angular velocity.
    """
    x, y, delta = state
    v, w = action
    return np.array(
        [
            x + dt * np.cos(delta) * v,
            y + dt * np.sin(delta) * v,
            delta + dt * w,
        ],
        dtype=state.dtype,
    )


def wrap_to_pi(angle):
    """Wrap angle(s) to [-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi

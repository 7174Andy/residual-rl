# two_wheel_robot/rl/features.py
"""Heading-safe featurization of DeePC's inputs for the behavioral clone.

The clone sees exactly what `DeePC.act` conditions on: the T_ini-step past
buffer (u_ini, y_ini), the current measurement y_current (which selects the
library), the goal, and a one-hot of the selected library. Headings are encoded
as (sin, cos) so there is no +/-pi wrap discontinuity, and absolute heading --
which library selection depends on -- is preserved. Output dim 6*T_ini + 6 + N_lib.
"""
from __future__ import annotations

import numpy as np


def select_library_index(heading: float, anchor_headings: np.ndarray) -> int:
    """Index of the library whose anchor is closest to `heading` on the circle.

    Mirrors `DeePC._select_index` exactly (a test pins the two together).
    """
    anchor_headings = np.asarray(anchor_headings, dtype=np.float64)
    diffs = (heading - anchor_headings + np.pi) % (2 * np.pi) - np.pi
    return int(np.argmin(np.abs(diffs)))


def featurize(
    u_ini: np.ndarray,
    y_ini: np.ndarray,
    y_current: np.ndarray,
    y_ref: np.ndarray,
    anchor_headings: np.ndarray,
) -> np.ndarray:
    """Build the clone feature vector.

    Args:
        u_ini: `(T_ini, 2)` past actions.
        y_ini: `(T_ini, 3)` past outputs `(x, y, delta)`.
        y_current: `(3,)` current measurement `(x, y, delta)`.
        y_ref: reference; only `y_ref[:2]` (the goal) is used.
        anchor_headings: `(N_lib,)` library anchor headings.

    Returns:
        `(6*T_ini + 6 + N_lib,)` float64 feature vector. Layout (row-major):
          ``[0 : 6*T_ini]``                 buffer — T_ini rows of
                                            ``(x, y, sin δ, cos δ, v, w)``
          ``[6*T_ini : 6*T_ini+4]``         y_current ``(x, y, sin δ, cos δ)``
          ``[6*T_ini+4 : 6*T_ini+6]``       goal ``(g_x, g_y)``
          ``[6*T_ini+6 : 6*T_ini+6+N_lib]`` library one-hot

    Raises:
        ValueError: if `u_ini` and `y_ini` disagree on `T_ini` (which would
            otherwise broadcast silently into a corrupt vector).
    """
    u_ini = np.asarray(u_ini, dtype=np.float64)
    y_ini = np.asarray(y_ini, dtype=np.float64)
    y_current = np.asarray(y_current, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64)
    anchor_headings = np.asarray(anchor_headings, dtype=np.float64)

    T_ini = u_ini.shape[0]
    if y_ini.shape[0] != T_ini:
        raise ValueError(
            f"u_ini and y_ini must share T_ini; got {T_ini} vs {y_ini.shape[0]}"
        )
    buf = np.empty((T_ini, 6), dtype=np.float64)
    buf[:, 0] = y_ini[:, 0]
    buf[:, 1] = y_ini[:, 1]
    buf[:, 2] = np.sin(y_ini[:, 2])
    buf[:, 3] = np.cos(y_ini[:, 2])
    buf[:, 4] = u_ini[:, 0]
    buf[:, 5] = u_ini[:, 1]

    cur = np.array(
        [y_current[0], y_current[1], np.sin(y_current[2]), np.cos(y_current[2])],
        dtype=np.float64,
    )
    goal = np.array([y_ref[0], y_ref[1]], dtype=np.float64)

    idx = select_library_index(float(y_current[2]), anchor_headings)
    onehot = np.zeros(anchor_headings.shape[0], dtype=np.float64)
    onehot[idx] = 1.0

    return np.concatenate([buf.reshape(-1), cur, goal, onehot])

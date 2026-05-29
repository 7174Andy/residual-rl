"""Shared, controller-agnostic scenario definitions for DeePC equivalence tests.

Pure numpy: generates the offline `(u, y)` data and the closed-loop input
sequences used to characterize DeePC behavior. Both the golden-fixture
generator (run against the *old* implementation) and the refactored tests (run
against the *new* implementation) import these, so the two are guaranteed to
exercise identical problems. Any drift here would invalidate the equivalence
claim, so keep it dependency-free and deterministic.
"""

from __future__ import annotations

import numpy as np

# Pinned solver so old/new solves are bit-comparable (within tolerance).
SOLVER = "CLARABEL"

# Shared horizons / weights for the 3-D (unicycle-shaped) scenarios.
T_DATA = 200
T_INI = 5
N = 8
Q = np.eye(3)
R = 0.01 * np.eye(2)
LAMBDA_G = 1.0
LAMBDA_Y = 1e3

# Paper-style quadrant anchors for the multi-library scenario.
ANCHORS = [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4]


def library_data(seed: int, T: int = T_DATA) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic offline `(u, y)` for one library (m_u=2, p_y=3)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, size=(T, 2))
    y = rng.standard_normal((T, 3))
    return u, y


def single_library() -> tuple[np.ndarray, np.ndarray]:
    """The single-library scenario's offline data (seed 0)."""
    return library_data(seed=0)


def multi_libraries(n: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """`n` libraries (seeds 0..n-1), parallel to ANCHORS[:n]."""
    return [library_data(seed=i) for i in range(n)]


def single_input_sequence() -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed `(y_current, y_ref)` steps for the single-library scenario."""
    y_ref = np.array([1.0, 1.0, 0.0])
    headings = np.linspace(0.0, 0.4, 10)
    return [
        (np.array([0.1 * t, 0.05 * t, h]), y_ref)
        for t, h in enumerate(headings)
    ]


def multi_input_sequence() -> list[tuple[np.ndarray, np.ndarray]]:
    """`(y_current, y_ref)` steps whose headings deliberately cross quadrants.

    Exercises library switching (and, for the new controller, the
    warm-start-reset-on-switch path): consecutive headings jump between
    quadrants and also include a same-quadrant repeat.
    """
    y_ref = np.array([1.0, 1.0, 0.0])
    headings = [
        0.2,            # Q0 (anchor pi/4)
        0.3,            # Q0 again (no switch)
        2.0,            # Q1 (anchor 3pi/4)
        -2.0,           # Q2 (anchor -3pi/4)
        -0.3,           # Q3 (anchor -pi/4)
        0.1,            # back to Q0
        np.pi - 0.1,    # Q1 (near +pi)
        -np.pi + 0.1,   # Q2 (near -pi)
    ]
    return [
        (np.array([0.1 * t, 0.05 * t, h]), y_ref)
        for t, h in enumerate(headings)
    ]

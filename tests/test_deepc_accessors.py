# tests/test_deepc_accessors.py
"""prime_buffer / past_buffer: arbitrary-past priming, behavior-preserving."""
from __future__ import annotations

import numpy as np
import pytest

from two_wheel_robot.controllers.deepc import DeePC
from tests.deepc_scenarios import ANCHORS, N, Q, R, T_INI, multi_libraries
from two_wheel_robot.controllers.hankel import build_hankel


def _controller() -> DeePC:
    libs = [build_hankel(u, y, T_ini=T_INI, N=N) for u, y in multi_libraries(4)]
    return DeePC(libs, anchor_headings=ANCHORS, Q=Q, R=R, T_ini=T_INI, N=N,
                 lambda_g=1.0, lambda_y=1e3, solver="CLARABEL")


def test_prime_then_read_roundtrips():
    c = _controller()
    rng = np.random.default_rng(0)
    u_buf = rng.uniform(-1, 1, size=(T_INI, c.m_u))
    y_buf = rng.standard_normal((T_INI, c.p_y))
    c.prime_buffer(u_buf, y_buf)
    u_out, y_out = c.past_buffer
    assert np.allclose(u_out, u_buf)
    assert np.allclose(y_out, y_buf)
    # Must be copies, not internal references.
    u_out[0, 0] = 999.0
    assert not np.allclose(c.past_buffer[0], u_out)


def test_prime_buffer_matches_reset_constant_past():
    # reset() builds a *constant* past; prime_buffer with the same tiled values
    # must yield an identical act() output (proves it's behavior-preserving).
    c1 = _controller()
    c2 = _controller()
    y0 = np.array([0.3, -0.2, 0.1])
    u0 = np.array([0.5, 0.0])
    y_ref = np.array([1.0, 1.0, 0.0])

    c1.reset(y0, u_initial=u0)
    c2.prime_buffer(np.tile(u0, (T_INI, 1)), np.tile(y0, (T_INI, 1)))

    u1 = c1.act(y0, y_ref)
    u2 = c2.act(y0, y_ref)
    assert np.allclose(u1, u2, atol=1e-6)


def test_prime_buffer_validates_shape():
    c = _controller()

    with pytest.raises(ValueError):
        c.prime_buffer(np.zeros((T_INI + 1, 2)), np.zeros((T_INI, 3)))
    with pytest.raises(ValueError):
        c.prime_buffer(np.zeros((T_INI, 2)), np.zeros((T_INI, 2)))

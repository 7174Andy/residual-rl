"""Unit tests for two_wheel_robot.env.dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from two_wheel_robot.env.dynamics import step_unicycle, wrap_to_pi


def _state(x: float, y: float, delta: float) -> np.ndarray:
    return np.array([x, y, delta], dtype=np.float64)


def _action(v: float, w: float) -> np.ndarray:
    return np.array([v, w], dtype=np.float64)


class TestStepUnicycle:
    def test_zero_action_preserves_state(self):
        s = _state(1.0, 2.0, 0.3)
        s_next = step_unicycle(s, _action(0.0, 0.0), dt=0.025)
        np.testing.assert_allclose(s_next, s)

    def test_forward_at_zero_heading_increases_x(self):
        s = _state(0.0, 0.0, 0.0)
        s_next = step_unicycle(s, _action(1.0, 0.0), dt=0.025)
        np.testing.assert_allclose(s_next, [0.025, 0.0, 0.0], atol=1e-12)

    def test_forward_at_pi_over_two_increases_y(self):
        s = _state(0.0, 0.0, np.pi / 2)
        s_next = step_unicycle(s, _action(1.0, 0.0), dt=0.025)
        np.testing.assert_allclose(s_next, [0.0, 0.025, np.pi / 2], atol=1e-12)

    def test_forward_at_pi_decreases_x(self):
        s = _state(0.0, 0.0, np.pi)
        s_next = step_unicycle(s, _action(1.0, 0.0), dt=0.025)
        np.testing.assert_allclose(s_next, [-0.025, 0.0, np.pi], atol=1e-12)

    def test_pure_rotation_only_changes_heading(self):
        s = _state(1.5, -2.5, 0.0)
        s_next = step_unicycle(s, _action(0.0, 1.0), dt=0.1)
        np.testing.assert_allclose(s_next[:2], s[:2])
        assert s_next[2] == pytest.approx(0.1)

    def test_does_not_mutate_input(self):
        s = _state(3.0, 4.0, 0.1)
        s_copy = s.copy()
        step_unicycle(s, _action(5.0, 0.5), dt=0.025)
        np.testing.assert_array_equal(s, s_copy)

    def test_velocity_scales_displacement_linearly(self):
        s = _state(0.0, 0.0, 0.0)
        s_slow = step_unicycle(s, _action(1.0, 0.0), dt=0.025)
        s_fast = step_unicycle(s, _action(10.0, 0.0), dt=0.025)
        assert s_fast[0] == pytest.approx(10.0 * s_slow[0])

    def test_dt_scales_displacement_linearly(self):
        s = _state(0.0, 0.0, 0.0)
        s_short = step_unicycle(s, _action(2.0, 0.0), dt=0.01)
        s_long = step_unicycle(s, _action(2.0, 0.0), dt=0.05)
        assert s_long[0] == pytest.approx(5.0 * s_short[0])


class TestWrapToPi:
    def test_zero(self):
        assert wrap_to_pi(0.0) == pytest.approx(0.0)

    @pytest.mark.parametrize("angle", [0.5, -0.5, 1.0, -1.0, 3.0, -3.0])
    def test_inside_range_unchanged(self, angle: float):
        assert wrap_to_pi(angle) == pytest.approx(angle)

    def test_above_pi_wraps_to_negative(self):
        result = float(wrap_to_pi(np.pi + 0.1))
        assert result == pytest.approx(-np.pi + 0.1)

    def test_below_negative_pi_wraps_to_positive(self):
        result = float(wrap_to_pi(-np.pi - 0.1))
        assert result == pytest.approx(np.pi - 0.1)

    def test_multiple_revolutions(self):
        result = float(wrap_to_pi(4 * np.pi + 0.2))
        assert result == pytest.approx(0.2, abs=1e-10)

    def test_array_input(self):
        angles = np.array([0.0, np.pi + 0.1, -np.pi - 0.2, 4 * np.pi])
        result = wrap_to_pi(angles)
        assert result.shape == angles.shape
        for r in result:
            assert -np.pi - 1e-9 <= r <= np.pi + 1e-9

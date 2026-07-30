"""Pure CSV read/write round-trip for closed-loop trace files -- no gym/
torch/sb3 imports anywhere in this test or in trace_io.py itself."""
from __future__ import annotations

import numpy as np

from two_wheel_robot.rl.trace_io import (
    clone_trace_path,
    read_trace,
    residual_trace_path,
    write_trace,
)


def test_clone_and_residual_trace_paths():
    assert clone_trace_path("figs", 7) == "figs/traj_7_clone.csv"
    assert residual_trace_path("figs", 7) == "figs/traj_7_residual.csv"


def test_write_then_read_trace_round_trips(tmp_path):
    traj = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.1], [2.0, 1.0, 0.2]])
    actions = np.array([[3.0, 0.05], [4.0, -0.05]])
    goal = np.array([5.0, 5.0])
    path = tmp_path / "traj_1_clone.csv"

    write_trace(str(path), traj, actions, goal)
    trace = read_trace(str(path))

    assert list(trace["step"]) == [0, 1, 2]
    assert np.allclose(trace["x"], [0.0, 1.0, 2.0])
    assert np.allclose(trace["y"], [0.0, 0.5, 1.0])
    assert np.allclose(trace["heading"], [0.0, 0.1, 0.2])
    assert np.allclose(trace["v"], [0.0, 3.0, 4.0])  # row 0 has no action -> 0.0
    assert np.allclose(trace["w"], [0.0, 0.05, -0.05])
    assert tuple(trace["goal"]) == (5.0, 5.0)

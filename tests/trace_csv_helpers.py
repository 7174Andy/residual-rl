# tests/trace_csv_helpers.py
"""Shared synthetic trace-CSV writer for tests that only need *some*
schema-correct trace to exist, not a specific trajectory shape."""
from __future__ import annotations

import csv


def write_synthetic_trace(path, n_rows, goal=(5.0, 5.0), x_scale=0.1):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "x", "y", "heading", "v", "w", "goal_x", "goal_y"])
        for t in range(n_rows):
            w.writerow(
                [t, float(t) * x_scale, 0.0, 0.0, 1.0 if t else 0.0, 0.0, goal[0], goal[1]]
            )

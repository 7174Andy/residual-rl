"""Smoke test for scripts/plot_seed_traces.py using synthetic trace CSVs --
no real checkpoints needed, confirming the trace_io refactor didn't change
behavior: the script still reads two traces and writes one PNG."""
from __future__ import annotations

import subprocess
import sys

from tests.trace_csv_helpers import write_synthetic_trace

SCRIPT = "scripts/plot_seed_traces.py"


def test_plot_seed_traces_from_synthetic_csvs(tmp_path):
    seed = 7
    write_synthetic_trace(tmp_path / f"traj_{seed}_clone.csv", n_rows=6)
    write_synthetic_trace(tmp_path / f"traj_{seed}_residual.csv", n_rows=4)
    out = tmp_path / "metrics.png"

    r = subprocess.run(
        [sys.executable, SCRIPT, "--seed", str(seed), "--figdir", str(tmp_path), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.stat().st_size > 0

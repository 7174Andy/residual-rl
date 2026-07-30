"""Integration smoke test for scripts/eval_seed_showcase.py -- exercises the
real closed loops (needs data/clone.pt, data/residual_td3.zip,
data/libraries_v0.npz), confirming the trace_io refactor didn't change
behavior: the script still writes both CSVs and prints the same message."""
from __future__ import annotations

import subprocess
import sys

import pytest

SCRIPT = "scripts/eval_seed_showcase.py"


@pytest.mark.integration
def test_eval_seed_showcase_writes_traces(tmp_path):
    seed = 4104626029  # known-good showcased seed
    r = subprocess.run(
        [sys.executable, SCRIPT, "--outdir", str(tmp_path), "--trace-seeds", str(seed)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    clone_csv = tmp_path / f"traj_{seed}_clone.csv"
    residual_csv = tmp_path / f"traj_{seed}_residual.csv"
    assert clone_csv.exists() and residual_csv.exists()
    assert f"wrote {clone_csv}" in r.stdout
    assert f"wrote {residual_csv}" in r.stdout

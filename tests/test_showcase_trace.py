"""ensure_traces: cache-hit short-circuit (must never touch checkpoints when
both CSVs already exist) and the real generate-and-cache path."""
from __future__ import annotations

import os

import pytest

from tests.trace_csv_helpers import write_synthetic_trace
from two_wheel_robot.rl.showcase_trace import ensure_traces
from two_wheel_robot.rl.trace_io import clone_trace_path, residual_trace_path

CLONE = "data/clone.pt"
RESIDUAL_MODEL = "data/residual_td3.zip"
LIB = "data/libraries_v0.npz"


def test_ensure_traces_cache_hit_never_touches_models(tmp_path):
    seed = 999
    write_synthetic_trace(clone_trace_path(str(tmp_path), seed), n_rows=3, x_scale=1.0)
    write_synthetic_trace(residual_trace_path(str(tmp_path), seed), n_rows=5, x_scale=1.0)

    clone, residual = ensure_traces(
        seed, str(tmp_path),
        clone_path="/nonexistent/clone.pt",
        residual_model_path="/nonexistent/model.zip",
        libraries_path="/nonexistent/libraries.npz",
    )
    assert len(clone["step"]) == 3
    assert len(residual["step"]) == 5


@pytest.mark.integration
def test_ensure_traces_generates_and_caches_when_missing(tmp_path):
    seed = 4104626029  # known-good showcased seed (see docs/journey/08-residual-rl.md)
    clone, residual = ensure_traces(
        seed, str(tmp_path),
        clone_path=CLONE, residual_model_path=RESIDUAL_MODEL, libraries_path=LIB,
    )
    assert len(clone["step"]) >= 2
    assert len(residual["step"]) >= 2
    assert os.path.exists(clone_trace_path(str(tmp_path), seed))
    assert os.path.exists(residual_trace_path(str(tmp_path), seed))

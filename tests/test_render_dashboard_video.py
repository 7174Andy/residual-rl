"""Smoke test for scripts/render_dashboard_video.py: given pre-cached trace
CSVs (the fast path -- no model loading), it must render a correctly-sized
MP4 without touching clone/residual checkpoints at all."""
from __future__ import annotations

import subprocess
import sys

from tests.trace_csv_helpers import write_synthetic_trace

SCRIPT = "scripts/render_dashboard_video.py"


def _run(args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)


def test_renders_video_from_cached_traces(tmp_path):
    seed = 42
    write_synthetic_trace(tmp_path / f"traj_{seed}_clone.csv", n_rows=8)
    write_synthetic_trace(tmp_path / f"traj_{seed}_residual.csv", n_rows=5)

    r = _run([
        "--seeds", str(seed),
        "--figdir", str(tmp_path), "--outdir", str(tmp_path),
        "--clone", "/nonexistent/clone.pt",
        "--residual-model", "/nonexistent/model.zip",
        "--libraries", "/nonexistent/libraries.npz",
        "--fps", "5",
    ])
    assert r.returncode == 0, r.stderr

    out = tmp_path / f"dashboard-{seed}.mp4"
    assert out.exists() and out.stat().st_size > 0

    import imageio.v2 as imageio
    reader = imageio.get_reader(str(out))
    n_frames = reader.count_frames()
    reader.close()
    assert n_frames == 8  # max(8, 5) rows -> steps 0..7 -> 8 frames

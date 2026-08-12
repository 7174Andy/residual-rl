"""Shared MP4 frame encoder (imageio + ffmpeg).

Used by `scripts/run_clone.py`, `scripts/run_residual.py`, and
`scripts/render_dashboard_video.py` -- their frame sources differ (pygame vs
matplotlib) but the encoding step is identical.
"""
from __future__ import annotations

import sys

import numpy as np


def encode_video(frames: list, path: str, fps: int) -> bool:
    """Encode a list of `(H, W, 3)` uint8 RGB frames to an MP4 at `path`."""
    if not frames:
        print(f"  warning: no frames to write for {path}", file=sys.stderr)
        return False
    try:
        import imageio.v2 as imageio
    except ImportError:
        print(
            "  warning: imageio not installed; cannot record. "
            "Install with `uv add imageio imageio-ffmpeg`.",
            file=sys.stderr,
        )
        return False
    writer = imageio.get_writer(
        path, mode="I", fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None,
    )
    try:
        for fr in frames:
            writer.append_data(np.ascontiguousarray(fr, dtype=np.uint8))
    finally:
        writer.close()
    return True

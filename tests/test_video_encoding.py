"""encode_video: writes a real, correctly-sized MP4 from synthetic frames;
returns False (no crash) when there are no frames to write."""
from __future__ import annotations

import numpy as np

from two_wheel_robot.rl.video_encoding import encode_video


def test_encode_video_writes_playable_mp4(tmp_path):
    frames = [np.full((64, 64, 3), i * 20, dtype=np.uint8) for i in range(5)]
    out = tmp_path / "clip.mp4"

    ok = encode_video(frames, str(out), fps=10)

    assert ok is True
    assert out.exists() and out.stat().st_size > 0
    import imageio.v2 as imageio
    reader = imageio.get_reader(str(out))
    n = reader.count_frames()
    reader.close()
    assert n == 5


def test_encode_video_empty_frames_returns_false(tmp_path):
    out = tmp_path / "empty.mp4"
    ok = encode_video([], str(out), fps=10)
    assert ok is False
    assert not out.exists()

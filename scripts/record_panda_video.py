"""Record PandaReach-v0 under random actions: MP4 + validity report.

    uv run python scripts/record_panda_video.py --episodes 3 --out data/panda_random.mp4

fps equals the 50 Hz control rate, so playback is real time.
"""
from __future__ import annotations

import argparse

from panda.env import PandaReachEnv
from panda.validity import format_report, record

# encode_video lives in core/ (system-agnostic, shared with the unicycle
# pipeline) — it already handles the optional-imageio path and the
# libx264/yuv420p/macro_block_size settings a playable MP4 needs, so there's
# no reason to duplicate it here.
from core.video_encoding import encode_video


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--out", default="data/panda_random.mp4")
    args = ap.parse_args()

    env = PandaReachEnv(render_mode="rgb_array", max_steps=args.max_steps)
    try:
        frames, report = record(env, episodes=args.episodes, seed=args.seed)
        fps = int(env.metadata["render_fps"])
    finally:
        env.close()

    print(format_report(report))
    if encode_video(frames, args.out, fps=fps):
        print(f"\nwrote {args.out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()

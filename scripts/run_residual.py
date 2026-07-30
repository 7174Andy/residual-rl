# scripts/run_residual.py
"""Closed-loop run of the clone+residual policy, with rendering/recording.

Mirrors scripts/run_clone.py but steps a ResidualDeePCEnv driven by the trained
TD3 residual. Record videos with:
    uv run python scripts/run_residual.py --record docs/journey/videos --seeds 4104626061
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from two_wheel_robot.rl.residual_env import ResidualDeePCEnv
from two_wheel_robot.rl.train_sb3 import load_residual
from two_wheel_robot.rl.video_encoding import encode_video


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/residual_td3.zip")
    p.add_argument("--clone", default="data/clone.pt")
    p.add_argument("--libraries", default="data/libraries_v0.npz")
    p.add_argument("--seeds", default="4104626029")
    p.add_argument("--residual-frac", type=float, default=1.0)
    p.add_argument("--record", default=None, metavar="DIR")
    p.add_argument("--algo", default="td3", choices=["td3", "sac"])
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    recording = args.record is not None
    render_mode = "rgb_array" if recording else "human"
    if recording:
        os.makedirs(args.record, exist_ok=True)

    model = load_residual(args.model, algo=args.algo, device=args.device)
    env = ResidualDeePCEnv(
        clone_path=args.clone, libraries_path=args.libraries,
        residual_frac=args.residual_frac, device=args.device, render_mode=render_mode,
    )
    fps = int(env.env.metadata.get("render_fps", 40))
    try:
        for s in seeds:
            obs, _ = env.reset(seed=s)
            frames = [np.asarray(env.render(), dtype=np.uint8)] if recording else []
            term = trunc = False
            steps = 0
            info: dict = {}
            while not (term or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                if recording:
                    frames.append(np.asarray(env.render(), dtype=np.uint8))
                steps += 1
            outcome = "REACHED" if info.get("reached") else "truncated"
            print(f"seed {s}: {outcome:9s} after {steps:3d} steps  "
                  f"final_dist={info['distance']:.2f}")
            if recording:
                out = os.path.join(args.record, f"episode_{s}.mp4")
                if encode_video(frames, out, fps):
                    print(f"  wrote {out} ({len(frames)} frames @ {fps} fps)")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

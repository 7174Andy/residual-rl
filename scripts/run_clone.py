"""Closed-loop run of the trained deep-lcc clone on TwoWheelGoal-v0, with rendering.

The mirror of `scripts/run_deepc.py`, but the QP controller is replaced by the
amortized NN clone (`data/clone.pt`). Loads the canonical config so the env
action bounds, anchors, and buffer priming match how the clone was trained and
validated, then runs the clone closed-loop:

    featurize(buffer, y_current, goal) -> clone.predict -> clip -> step -> slide

Usage:
    # live pygame window (opens on your display)
    uv run python scripts/run_clone.py --episodes 3 --seed 0

    # record MP4s headless (no display needed)
    uv run python scripts/run_clone.py --record videos/clone --seeds 4104626029,4104626034

    # aggregate performance, no window
    uv run python scripts/run_clone.py --headless --episodes 78 --seed 4104626029
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import cast

import gymnasium as gym
import numpy as np

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from two_wheel_robot.env.env import UnicycleGoalEnv
from two_wheel_robot.rl.clone import load_clone
from two_wheel_robot.rl.deepc_setup import bearing_y_ref, build_canonical_deepc
from two_wheel_robot.rl.features import featurize
from two_wheel_robot.rl.video_encoding import encode_video


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone", default="data/clone.pt")
    parser.add_argument("--libraries", default="data/libraries_v0.npz")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed; episode ep resets with seed = base + ep.")
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="comma-separated exact seeds to run (overrides --episodes/--seed); "
             "one video per seed named episode_<seed>.mp4.",
    )
    parser.add_argument(
        "--random", action="store_true",
        help="draw the base seed from OS entropy (printed for reproducibility).",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="no pygame window; run as fast as possible (for aggregate metrics).",
    )
    parser.add_argument(
        "--record", type=str, default=None, metavar="DIR",
        help="record each episode to DIR/episode_<seed>.mp4 (forces rgb_array).",
    )
    args = parser.parse_args()

    # Resolve the seed list. --seeds wins; else base (+ --random) plus episode idx.
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    else:
        if args.random:
            base_seed = int(np.random.default_rng().integers(0, 2**32))
            print(f"--random: drew base seed {base_seed} "
                  f"(rerun with --seed {base_seed} to reproduce)")
        else:
            base_seed = args.seed
        seeds = [base_seed + ep for ep in range(args.episodes)]

    # Recording forces offscreen rgb_array; else human window unless --headless.
    recording = args.record is not None
    if recording:
        render_mode = "rgb_array"
        os.makedirs(args.record, exist_ok=True)
    else:
        render_mode = None if args.headless else "human"

    try:
        clone = load_clone(args.clone, device=args.device)
    except FileNotFoundError:
        print(f"error: could not find clone checkpoint {args.clone}. Train it first:\n"
              f"  uv run python scripts/train_clone.py --out {args.clone}",
              file=sys.stderr)
        return 1

    # Canonical config (action bounds, anchors, buffer priming) — the same the
    # clone was trained/validated under.
    deepc, info = build_canonical_deepc(libraries_path=args.libraries)
    del deepc  # only the info dict is needed to drive the clone
    T_ini = info["T_ini"]
    a_low, a_high = info["action_bounds"][:, 0], info["action_bounds"][:, 1]

    env = gym.make(
        "TwoWheelGoal-v0", action_bounds=info["action_bounds"], render_mode=render_mode
    )
    base = cast(UnicycleGoalEnv, env.unwrapped)
    record_fps = int(env.metadata.get("render_fps", 40))

    records: list[dict] = []
    try:
        for s in seeds:
            env.reset(seed=s)
            u_buf = np.tile(info["u_init_midpoint"], (T_ini, 1))
            y_buf = np.tile(base.y, (T_ini, 1))
            frames: list[np.ndarray] = []
            if recording:
                frames.append(np.asarray(env.render(), dtype=np.uint8))  # initial pose

            terminated = truncated = False
            steps = 0
            total_reward = 0.0
            applied: list[np.ndarray] = []
            while not (terminated or truncated):
                y_cur = base.y
                y_ref = bearing_y_ref(base.state, base.goal)
                feat = featurize(u_buf, y_buf, y_cur, y_ref, info["anchors"])
                u = np.clip(clone.predict(feat), a_low, a_high)
                _, reward, terminated, truncated, step_info = env.step(u)
                if recording:
                    frames.append(np.asarray(env.render(), dtype=np.uint8))
                # Slide the buffer with (applied u, pre-step measurement) — the
                # same convention used to generate the clone's training labels.
                u_buf = np.roll(u_buf, -1, axis=0); u_buf[-1] = u
                y_buf = np.roll(y_buf, -1, axis=0); y_buf[-1] = y_cur
                applied.append(u.copy())
                total_reward += float(reward)
                steps += 1

            outcome = "REACHED" if terminated else "truncated"
            records.append({"reached": terminated, "steps": steps,
                            "final_dist": float(step_info["distance"])})
            print(f"seed {s}: {outcome:9s} after {steps:3d} steps  "
                  f"return={total_reward:+10.1f}  final_dist={step_info['distance']:.2f}")
            if applied:
                arr = np.asarray(applied)
                print(f"  v: mean={arr[:,0].mean():+.2f} std={arr[:,0].std():.2f}   "
                      f"w: mean={arr[:,1].mean():+.2f} std={arr[:,1].std():.2f}")
            if recording:
                out_path = os.path.join(args.record, f"episode_{s}.mp4")
                if encode_video(frames, out_path, record_fps):
                    print(f"  wrote {out_path} ({len(frames)} frames @ {record_fps} fps)")

        if records:
            n = len(records)
            n_reached = sum(r["reached"] for r in records)
            finals = np.array([r["final_dist"] for r in records])
            print(f"\n=== summary over {n} episodes ===")
            print(f"  reached : {n_reached}/{n} = {n_reached / n:.1%}")
            print(f"  final_dist: mean={finals.mean():.2f}  max={finals.max():.2f}")

        if render_mode == "human":
            time.sleep(1.5)  # hold the final frame briefly
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

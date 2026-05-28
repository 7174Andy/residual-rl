"""Closed-loop DeePC on TwoWheelGoal-v0 with pygame rendering.

Loads pre-collected (u, y) libraries from `scripts/collect_data.py`, builds the
past/future Hankels, and runs the DeePC controller across a few episodes. A
pygame window shows the robot's trail, the goal, and a HUD.

The env is constructed with the *same* action bounds the data was collected
under (paper PE bounds: v ∈ [10, 20], w ∈ [-π/6, π/6]) so DeePC stays inside
its data envelope. If you trained data with different bounds, override with
`--action_bounds`.

Usage:
    uv run python scripts/collect_data.py --out data/libraries.npz   # first
    uv run python scripts/run_deepc.py
    uv run python scripts/run_deepc.py --library 0 --episodes 5 --seed 42
    uv run python scripts/run_deepc.py --T_ini 5 --N 9              # paper variants
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import cast

import gymnasium as gym
import numpy as np

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from two_wheel_robot.controllers.data_collection import (
    PAPER_INIT_HEADINGS,
    PAPER_SAMPLE_BOUNDS,
)
from two_wheel_robot.controllers.deepc import DeePC, LibrarySwitchingDeePC
from two_wheel_robot.controllers.hankel import build_hankel
from two_wheel_robot.env.dynamics import wrap_to_pi
from two_wheel_robot.env.env import UnicycleGoalEnv


def _resolve_sample_bounds(data) -> np.ndarray:
    """Read `sample_bounds` from the .npz; fall back to paper bounds for old files."""
    if "sample_bounds" in data.files:
        bounds = np.asarray(data["sample_bounds"], dtype=np.float64)
        return bounds
    print("warning: no sample_bounds key in libraries file; assuming PAPER_SAMPLE_BOUNDS.")
    return PAPER_SAMPLE_BOUNDS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libraries", default="data/libraries.npz")
    parser.add_argument(
        "--single_library",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help=(
            "use only one library (its index 0..3); skip orientation switching. "
            "Default: use all 4 libraries with quadrant-based switching."
        ),
    )
    parser.add_argument("--T_ini", type=int, default=5)
    parser.add_argument("--N", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lambda_g",
        type=float,
        default=2.0,
        help="L1 regularizer on g (paper default 2.0)",
    )
    parser.add_argument(
        "--lambda_y",
        type=float,
        default=3e6,
        help="L2 regularizer on past-output slack (paper default 3e6)",
    )
    parser.add_argument(
        "--Q_heading",
        type=float,
        default=1.0,
        help=(
            "weight on heading deviation in Q (default 1.0). Set to 0 to "
            "reproduce paper's 'heading don't-care' Q = diag(1, 1, 0)."
        ),
    )
    parser.add_argument(
        "--no_bearing_ref",
        action="store_true",
        help=(
            "use the env's default y_ref = (g_x, g_y, 0) instead of "
            "(g_x, g_y, bearing_to_goal). Combine with --Q_heading 0 for paper-faithful."
        ),
    )
    args = parser.parse_args()

    # Load offline data
    try:
        data = np.load(args.libraries)
    except FileNotFoundError:
        print(
            f"error: could not find {args.libraries}. Generate it first:\n"
            f"  uv run python scripts/collect_data.py --out {args.libraries}",
            file=sys.stderr,
        )
        return 1

    sample_bounds = _resolve_sample_bounds(data)
    print(
        f"sample bounds (from data): v in [{sample_bounds[0,0]:.3f}, {sample_bounds[0,1]:.3f}], "
        f"w in [{sample_bounds[1,0]:.3f}, {sample_bounds[1,1]:.3f}]"
    )

    # Match env action bounds to the data collection bounds so the controller
    # stays inside its empirical envelope (no extrapolation to unseen actions).
    env = gym.make(
        "TwoWheelGoal-v0",
        action_bounds=sample_bounds,
        render_mode="human",
    )
    base = cast(UnicycleGoalEnv, env.unwrapped)

    # Override Q's heading weight (default 1.0). With paper's Q[2,2]=0, the QP
    # has no direct cost gradient on heading and tends to saturate w.
    Q = base.Q.copy()
    Q[2, 2] = args.Q_heading
    print(f"Q = diag({Q[0,0]:.3f}, {Q[1,1]:.3f}, {Q[2,2]:.3f})")

    u_bounds = (base.action_bounds[:, 0], base.action_bounds[:, 1])

    def _build_deepc(u_data: np.ndarray, y_data: np.ndarray) -> DeePC:
        Up, Uf, Yp, Yf = build_hankel(u_data, y_data, T_ini=args.T_ini, N=args.N)
        return DeePC(
            Up, Uf, Yp, Yf,
            Q=Q, R=base.R,
            T_ini=args.T_ini, N=args.N,
            lambda_g=args.lambda_g, lambda_y=args.lambda_y,
            u_bounds=u_bounds,
        )

    if args.single_library is None:
        # Build all 4 libraries → 4 DeePCs → orientation-keyed switcher.
        sub_controllers = [
            _build_deepc(data[f"u_{i}"], data[f"y_{i}"]) for i in range(4)
        ]
        # Anchor headings = paper init states wrapped to [-pi, pi].
        anchors = np.asarray(
            [float(wrap_to_pi(h)) for h in PAPER_INIT_HEADINGS], dtype=np.float64
        )
        controller = LibrarySwitchingDeePC(sub_controllers, anchors)
        print(
            f"library-switching DeePC: 4 libraries, anchors = "
            f"{[round(a, 3) for a in anchors]}"
        )
        Up0 = sub_controllers[0].Up
        Uf0 = sub_controllers[0].Uf
        print(
            f"Hankels per library (T_ini={args.T_ini}, N={args.N}): "
            f"Up {Up0.shape}, Uf {Uf0.shape}"
        )
    else:
        i = args.single_library
        u_data = data[f"u_{i}"]
        y_data = data[f"y_{i}"]
        controller = _build_deepc(u_data, y_data)
        print(f"single-library DeePC: library {i} (u {u_data.shape}, y {y_data.shape})")

    # Prime the controller's past-action buffer at the midpoint of action_bounds.
    # Zero-initialization makes the QP try to satisfy Up·g = 0, which (for data
    # with non-negative v) locks the controller into outputting u ≈ 0 — see the
    # cold-start discussion in CLAUDE.md.
    u_init_midpoint = 0.5 * (base.action_bounds[:, 0] + base.action_bounds[:, 1])
    print(f"u_initial (midpoint): v={u_init_midpoint[0]:.3f}, w={u_init_midpoint[1]:.3f}")

    try:
        for ep in range(args.episodes):
            _, info = env.reset(seed=args.seed + ep)
            controller.reset(base.y, u_initial=u_init_midpoint)
            steps = 0
            total_reward = 0.0
            terminated = truncated = False
            qp_failed = False
            applied = []
            lib_usage = np.zeros(4, dtype=np.int64)
            while not (terminated or truncated):
                if args.no_bearing_ref:
                    y_ref_step = base.y_ref
                else:
                    # Heading reference = bearing from robot to goal, updated each step.
                    dx_g = base.goal[0] - base.state[0]
                    dy_g = base.goal[1] - base.state[1]
                    bearing = float(np.arctan2(dy_g, dx_g))
                    y_ref_step = np.array(
                        [base.goal[0], base.goal[1], bearing], dtype=np.float64
                    )
                try:
                    u_t = controller.act(base.y, y_ref_step)
                except RuntimeError as exc:
                    print(f"  QP failure at step {steps}: {exc}")
                    qp_failed = True
                    break
                if isinstance(controller, LibrarySwitchingDeePC):
                    lib_usage[controller.last_library_idx] += 1
                _, reward, terminated, truncated, info = env.step(u_t)
                applied.append(u_t.copy())
                total_reward += float(reward)
                steps += 1
            if qp_failed:
                outcome = "QP-FAIL"
            elif terminated:
                outcome = "REACHED"
            else:
                outcome = "truncated"
            print(
                f"episode {ep}: {outcome:9s} after {steps:3d} steps  "
                f"return={total_reward:+10.1f}  final_dist={info['distance']:.2f}"
            )
            if applied:
                arr = np.asarray(applied)
                v_col, w_col = arr[:, 0], arr[:, 1]
                print(
                    f"  v: min={v_col.min():+.3f} max={v_col.max():+.3f} "
                    f"mean={v_col.mean():+.3f} std={v_col.std():.3f}"
                )
                print(
                    f"  w: min={w_col.min():+.3f} max={w_col.max():+.3f} "
                    f"mean={w_col.mean():+.3f} std={w_col.std():.3f}"
                )
                if isinstance(controller, LibrarySwitchingDeePC):
                    print(f"  library usage: {lib_usage.tolist()}")
        # Hold the final frame briefly so the last state is visible.
        time.sleep(1.5)
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

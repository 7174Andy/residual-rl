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
    uv run python scripts/run_deepc.py --random --headless --episodes 100  # perf eval
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
from two_wheel_robot.controllers.data_collection import (
    PAPER_INIT_HEADINGS,
    DEFAULT_SAMPLE_BOUNDS,
)
from core.deepc import DeePC
from core.hankel import build_hankel
from two_wheel_robot.env.dynamics import wrap_to_pi
from two_wheel_robot.env.env import UnicycleGoalEnv


def _encode_video(frames: list[np.ndarray], path: str, fps: int) -> bool:
    """Encode `(H, W, 3)` uint8 RGB frames to an MP4 at `path`.

    Uses imageio with its bundled ffmpeg (imageio-ffmpeg), so it does not depend
    on a system ffmpeg install. yuv420p output keeps the file playable in browsers
    (MkDocs docs pages); the 600x600 renderer satisfies the even-dimension need.
    `macro_block_size=None` preserves the exact 600x600 frame size.
    """
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
        path, mode="I", fps=fps,
        codec="libx264", pixelformat="yuv420p", macro_block_size=None,
    )
    try:
        for fr in frames:
            writer.append_data(np.ascontiguousarray(fr, dtype=np.uint8))
    finally:
        writer.close()
    return True


def _resolve_sample_bounds(data) -> np.ndarray:
    """Read `sample_bounds` from the .npz; fall back to the default for old files."""
    if "sample_bounds" in data.files:
        bounds = np.asarray(data["sample_bounds"], dtype=np.float64)
        return bounds
    print("warning: no sample_bounds key in libraries file; assuming DEFAULT_SAMPLE_BOUNDS.")
    return DEFAULT_SAMPLE_BOUNDS


def _trace_step(step, cur_dist, goal, u_t, controller, diag, trace_rows) -> None:
    """Print one per-step trace line and accumulate it for the episode verdict.

    Columns: the applied `(v, w)`; the QP's own predicted distance reduction over
    the horizon (`plan Δd`, positive = predicted to get closer); and the
    forced-forward counterfactual's predicted reduction (`forced Δd`) and slack
    when the first-step `v` is pinned `>= v_floor`.
    """
    goal = np.asarray(goal, dtype=np.float64)
    # The QP's own plan: predicted distance at the horizon end vs now.
    plan_dd = float("nan")
    if controller.last_pred_y is not None:
        plan_end = float(np.linalg.norm(controller.last_pred_y[-1, :2] - goal))
        plan_dd = cur_dist - plan_end
    # Forced-forward counterfactual.
    forced_dd = float("nan")
    forced_sig = float("nan")
    forced_v = float("nan")
    status = "n/a"
    if diag is not None:
        status = diag["status"]
        if "pred_y" in diag:
            forced_end = float(np.linalg.norm(diag["pred_y"][-1, :2] - goal))
            forced_dd = cur_dist - forced_end
            forced_sig = diag["sigma_y_norm"]
            forced_v = float(diag["u_first"][0])
    trace_rows.append(
        {"plan_dd": plan_dd, "forced_dd": forced_dd, "forced_status": status}
    )
    print(
        f"  step {step:3d} d={cur_dist:6.2f} lib={controller.last_library_idx} "
        f"v={u_t[0]:6.2f} w={u_t[1]:+5.2f} σy={controller.last_sigma_y_norm:8.1f} | "
        f"plan Δd={plan_dd:+6.2f} | forced(v≥) v={forced_v:5.1f} "
        f"Δd={forced_dd:+6.2f} σy={forced_sig:8.1f} [{status}]"
    )


def _trace_verdict(trace_rows, applied) -> None:
    """Summarize a traced episode into a data-vs-cost-shaping verdict."""
    if not trace_rows:
        return
    import statistics as st

    v_applied = [float(a[0]) for a in applied]
    plan = [r["plan_dd"] for r in trace_rows if r["plan_dd"] == r["plan_dd"]]
    forced = [r["forced_dd"] for r in trace_rows if r["forced_dd"] == r["forced_dd"]]
    n_forced_ok = sum(1 for r in trace_rows if r["forced_status"].startswith("optimal"))
    med_v = st.median(v_applied) if v_applied else float("nan")
    med_plan = st.median(plan) if plan else float("nan")
    med_forced = st.median(forced) if forced else float("nan")
    print(
        f"  --- trace verdict ---\n"
        f"  median applied v       : {med_v:.2f}  (collapse if ~0)\n"
        f"  median plan Δd (horizon): {med_plan:+.2f}  "
        f"(QP's own predicted approach per step)\n"
        f"  median forced Δd       : {med_forced:+.2f}  over {n_forced_ok}/"
        f"{len(trace_rows)} solved forced probes\n"
        f"  reading: forced Δd ~0 (or worse) while v≈0  => DATA COVERAGE "
        f"(library can't represent driving in)\n"
        f"           forced Δd strongly + while v≈0     => COST-SHAPING / "
        f"COLD-START (data can drive in; QP declined)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libraries", default="data/libraries_v0.npz")
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
        "--random",
        action="store_true",
        help=(
            "draw a base seed from OS entropy instead of using --seed, so start "
            "poses and goals are genuinely random. The drawn seed is printed; "
            "rerun with --seed <that value> (no --random) to reproduce the run."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "disable the pygame window and run as fast as possible. Use for "
            "measuring aggregate performance over many episodes."
        ),
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "record each episode to DIR/episode_<ep>.mp4 via system ffmpeg. "
            "Forces rgb_array rendering (overrides --headless / the human window)."
        ),
    )
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
        default=2.0,
        help=(
            "weight on heading deviation in Q (default 2.0, matching the "
            "paper's Q_z = diag(1, 1, 2)). Set to 0 for a 'heading don't-care' "
            "Q = diag(1, 1, 0)."
        ),
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "per-step diagnostic trace: applied v/w, selected library, past-output "
            "slack ‖σ_y‖, the QP's own predicted distance reduction, and a "
            "forced-forward counterfactual (re-solve with first-step v >= "
            "--trace_v_floor). Separates data-coverage vs cost-shaping vs cold-start "
            "for the v-collapse failure. Best with --episodes 1."
        ),
    )
    parser.add_argument(
        "--trace_v_floor",
        type=float,
        default=12.0,
        help="forced first-step v in the --trace counterfactual (default 12.0).",
    )
    parser.add_argument(
        "--no_bearing_ref",
        action="store_true",
        help=(
            "use the env's default y_ref = (g_x, g_y, 0) instead of "
            "(g_x, g_y, bearing_to_goal). The paper tracks a tangent-heading "
            "reference, so the default (bearing ref + --Q_heading 2) is the "
            "paper-faithful setting; this flag disables it."
        ),
    )
    args = parser.parse_args()

    # Resolve the per-episode seeding. --random draws a base seed from OS entropy
    # (printed so the run stays reproducible); otherwise use the deterministic
    # --seed. Either way episode `ep` resets with base_seed + ep.
    if args.random:
        base_seed = int(np.random.default_rng().integers(0, 2**32))
        print(
            f"--random: drew base seed {base_seed} "
            f"(rerun with --seed {base_seed} to reproduce)"
        )
    else:
        base_seed = args.seed

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

    # Recording forces offscreen rgb_array rendering; otherwise human window
    # unless --headless. (rgb_array takes precedence so we can capture frames.)
    recording = args.record is not None
    if recording:
        render_mode = "rgb_array"
        os.makedirs(args.record, exist_ok=True)
    else:
        render_mode = None if args.headless else "human"

    # Match env action bounds to the data collection bounds so the controller
    # stays inside its empirical envelope (no extrapolation to unseen actions).
    env = gym.make(
        "TwoWheelGoal-v0",
        action_bounds=sample_bounds,
        render_mode=render_mode,
    )
    base = cast(UnicycleGoalEnv, env.unwrapped)
    record_fps = int(env.metadata.get("render_fps", 40))

    # Override Q's heading weight (default 1.0). With paper's Q[2,2]=0, the QP
    # has no direct cost gradient on heading and tends to saturate w.
    Q = base.Q.copy()
    Q[2, 2] = args.Q_heading
    print(f"Q = diag({Q[0,0]:.3f}, {Q[1,1]:.3f}, {Q[2,2]:.3f})")

    u_bounds = (base.action_bounds[:, 0], base.action_bounds[:, 1])

    def _hankels(u_data: np.ndarray, y_data: np.ndarray):
        return build_hankel(u_data, y_data, T_ini=args.T_ini, N=args.N)

    # Anchor headings = paper init states wrapped to [-pi, pi].
    anchors = np.asarray(
        [float(wrap_to_pi(h)) for h in PAPER_INIT_HEADINGS], dtype=np.float64
    )

    if args.single_library is None:
        # All 4 orientation-keyed libraries fed into one parametric controller.
        libraries = [_hankels(data[f"u_{i}"], data[f"y_{i}"]) for i in range(4)]
        controller_anchors = anchors
        print(
            f"library-switching DeePC: 4 libraries, anchors = "
            f"{[round(a, 3) for a in anchors]}"
        )
        Up0, Uf0 = libraries[0][0], libraries[0][1]
        print(
            f"Hankels per library (T_ini={args.T_ini}, N={args.N}): "
            f"Up {Up0.shape}, Uf {Uf0.shape}"
        )
    else:
        i = args.single_library
        u_data = data[f"u_{i}"]
        y_data = data[f"y_{i}"]
        libraries = [_hankels(u_data, y_data)]
        controller_anchors = anchors[i : i + 1]
        print(f"single-library DeePC: library {i} (u {u_data.shape}, y {y_data.shape})")

    n_lib = len(libraries)
    controller = DeePC(
        libraries,
        anchor_headings=controller_anchors,
        Q=Q, R=base.R,
        T_ini=args.T_ini, N=args.N,
        lambda_g=args.lambda_g, lambda_y=args.lambda_y,
        u_bounds=u_bounds,
    )

    # Prime the controller's past-action buffer at the midpoint of action_bounds.
    # Zero-initialization makes the QP try to satisfy Up·g = 0, which (for data
    # with non-negative v) locks the controller into outputting u ≈ 0 — see the
    # cold-start discussion in CLAUDE.md.
    u_init_midpoint = 0.5 * (base.action_bounds[:, 0] + base.action_bounds[:, 1])
    print(f"u_initial (midpoint): v={u_init_midpoint[0]:.3f}, w={u_init_midpoint[1]:.3f}")

    # Per-episode records for the aggregate summary.
    records: list[dict] = []
    try:
        for ep in range(args.episodes):
            _, info = env.reset(seed=base_seed + ep)
            controller.reset(base.y, u_initial=u_init_midpoint)
            steps = 0
            total_reward = 0.0
            terminated = truncated = False
            qp_failed = False
            applied = []
            lib_usage = np.zeros(n_lib, dtype=np.int64)
            trace_rows: list[dict] = []
            frames: list[np.ndarray] = []
            if recording:
                frames.append(np.asarray(env.render(), dtype=np.uint8))  # initial pose
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
                # Forced-forward counterfactual must read the buffer BEFORE act()
                # slides it, so probe first (it does not mutate controller state).
                diag = (
                    controller.diagnose_forward(base.y, y_ref_step, args.trace_v_floor)
                    if args.trace
                    else None
                )
                cur_dist = float(np.linalg.norm(base.state[:2] - base.goal))
                try:
                    u_t = controller.act(base.y, y_ref_step)
                except RuntimeError as exc:
                    print(f"  QP failure at step {steps}: {exc}")
                    qp_failed = True
                    break
                lib_usage[controller.last_library_idx] += 1
                if args.trace:
                    _trace_step(
                        steps, cur_dist, base.goal, u_t, controller, diag, trace_rows
                    )
                _, reward, terminated, truncated, info = env.step(u_t)
                if recording:
                    frames.append(np.asarray(env.render(), dtype=np.uint8))
                applied.append(u_t.copy())
                total_reward += float(reward)
                steps += 1
            if qp_failed:
                outcome = "QP-FAIL"
            elif terminated:
                outcome = "REACHED"
            else:
                outcome = "truncated"
            records.append(
                {
                    "reached": outcome == "REACHED",
                    "qp_failed": qp_failed,
                    "steps": steps,
                    "return": total_reward,
                    "final_dist": float(info["distance"]),
                }
            )
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
                if n_lib > 1:
                    print(f"  library usage: {lib_usage.tolist()}")
            if args.trace:
                _trace_verdict(trace_rows, applied)
            if recording:
                out_path = os.path.join(args.record, f"episode_{ep}.mp4")
                if _encode_video(frames, out_path, record_fps):
                    print(f"  wrote {out_path} ({len(frames)} frames @ {record_fps} fps)")

        # Aggregate summary across all episodes.
        if records:
            n = len(records)
            n_reached = sum(r["reached"] for r in records)
            n_qp_fail = sum(r["qp_failed"] for r in records)
            returns = np.array([r["return"] for r in records])
            finals = np.array([r["final_dist"] for r in records])
            reached_steps = [r["steps"] for r in records if r["reached"]]
            seed_desc = (
                f"random base_seed={base_seed}" if args.random else f"seed={base_seed}"
            )
            print(
                f"\n=== summary over {n} episodes ({seed_desc}) ===\n"
                f"  success rate : {n_reached}/{n} = {n_reached / n:.1%}"
                + (f"   (QP failures: {n_qp_fail})" if n_qp_fail else "")
                + "\n"
                f"  return       : mean={returns.mean():+.1f}  std={returns.std():.1f}\n"
                f"  final_dist   : mean={finals.mean():.2f}  std={finals.std():.2f}  "
                f"max={finals.max():.2f}\n"
                f"  steps(reached): "
                + (
                    f"mean={np.mean(reached_steps):.1f}  "
                    f"min={min(reached_steps)}  max={max(reached_steps)}"
                    if reached_steps
                    else "n/a (none reached)"
                )
            )

        # Hold the final frame briefly so the last state is visible (windowed only).
        if not args.headless:
            time.sleep(1.5)
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

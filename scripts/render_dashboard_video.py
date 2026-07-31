#!/usr/bin/env python
"""Side-by-side clone vs clone+residual dashboard video.

Two columns (clone left, clone+TD3-residual right), each with an animated
trajectory panel (trail, goal, tolerance circle, heading-marker robot, text
HUD) and three live sparklines underneath (v(t), w(t), cumulative reward),
playing in real time (fps = env's render_fps = 40, matching Delta t = 0.025s).

Reuses the traj_<seed>_{clone,residual}.csv cache scripts/eval_seed_showcase.py
originates (via two_wheel_robot.rl.showcase_trace) -- if a seed hasn't been
showcased yet, this script runs both closed loops itself and writes the cache
as a byproduct, so a new seed is one command:

    uv run python scripts/render_dashboard_video.py --seeds 4104626029,4104626034

Design spec: docs/superpowers/specs/2026-07-30-dashboard-video-design.md
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless: encode frames straight to video, never open a window
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from two_wheel_robot.rl.showcase_trace import ensure_traces, ensure_vanilla_trace  # noqa: E402
from two_wheel_robot.rl.trace_reward import DEFAULT_GOAL_TOLERANCE, recompute_reward  # noqa: E402
from two_wheel_robot.rl.video_encoding import encode_video  # noqa: E402

_GRAY = "#898781"
_BLUE = "#3987e5"
_INK = "#52514e"
_RED = "#e34948"
# vanilla-TD3 arm, distinct from the residual's blue. Validated with the dataviz
# skill alongside #898781/#3987e5: clears the 3:1 surface-contrast check that the
# lighter #d9822b failed. Same hex as scripts/plot_robustness.py, so the arm keeps
# one identity across figures and videos.
_ORANGE = "#c1701c"
_TOLERANCE = DEFAULT_GOAL_TOLERANCE  # two_wheel_robot/rl/trace_reward.py's copy of env.py's default


def _axis_limits(xs: list, ys: list, goal, tolerance, pad_frac=0.15):
    """Shared (xlim, ylim) covering every point in `xs`/`ys` plus the goal +/-
    tolerance, padded by `pad_frac` of the larger span. Computed once from the
    full, already-known trajectories of BOTH controllers -- never re-fit per
    frame (would visibly pan/zoom during playback) and never fit per-panel
    independently (would make the two columns' viewports incomparable).
    """
    all_x = np.concatenate([*xs, [goal[0] - tolerance, goal[0] + tolerance]])
    all_y = np.concatenate([*ys, [goal[1] - tolerance, goal[1] + tolerance]])
    x_lo, x_hi = float(all_x.min()), float(all_x.max())
    y_lo, y_hi = float(all_y.min()), float(all_y.max())
    span = max(x_hi - x_lo, y_hi - y_lo, 1e-6)
    pad = pad_frac * span
    return (x_lo - pad, x_hi + pad), (y_lo - pad, y_hi + pad)


def _setup_panel(fig, gs, trace, reward_data, color, title, xlim, ylim, full_len):
    """Create this panel's axes and every artist once. Static elements (goal,
    tolerance circle, faint full-episode sparkline curves, axis limits/title)
    never change frame to frame and are drawn here only; `_update_panel`
    mutates the returned handles in place instead of rebuilding any of this.
    """
    ax_traj = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1])
    ax_w = fig.add_subplot(gs[2])
    ax_r = fig.add_subplot(gs[3])

    goal = trace["goal"]
    ax_traj.scatter([trace["x"][0]], [trace["y"][0]], color=_INK, s=25, zorder=4)
    ax_traj.scatter([goal[0]], [goal[1]], color=_RED, s=90, marker="*", zorder=4)
    ax_traj.add_patch(
        mpatches.Circle(goal, _TOLERANCE, fill=False, ls="--", color=_RED, lw=1.0, zorder=1)
    )
    (trail,) = ax_traj.plot([], [], color=color, lw=1.8, zorder=3)
    heading_marker = mpatches.Polygon(
        [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)], closed=True, color=color, zorder=5,
    )
    ax_traj.add_patch(heading_marker)
    ax_traj.set_xlim(*xlim)
    ax_traj.set_ylim(*ylim)
    ax_traj.set_aspect("equal")
    ax_traj.grid(alpha=0.25)
    ax_traj.set_title(title, fontsize=11, color=_INK, fontweight="bold")

    hud_text = ax_traj.text(
        0.02, 0.98, "", transform=ax_traj.transAxes, ha="left", va="top",
        fontsize=8.5, color=_INK, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="#d8d6cd", alpha=0.85),
    )
    banner_text = ax_traj.text(
        0.98, 0.98, "", transform=ax_traj.transAxes, ha="right", va="top",
        fontsize=9.5, color="white", fontweight="bold", visible=False,
        bbox=dict(boxstyle="round", fc="#3a9d5d", ec="none"),
    )

    sparklines = []
    for ax, series, name in (
        (ax_v, trace["v"], "v(t)"),
        (ax_w, trace["w"], "w(t)"),
        (ax_r, reward_data["cum_reward"], "reward(t)"),
    ):
        # Faint full-episode curve, drawn once -- pins the axis limits from
        # frame 1 onward (autoscale sees the whole range immediately), so the
        # sparkline axes never jitter even though only the solid overlay
        # (updated per frame below) grows frame to frame.
        ax.plot(trace["step"], series, color=color, lw=1.0, alpha=0.25, zorder=1)
        (solid,) = ax.plot([], [], color=color, lw=1.4, zorder=2)
        vline = ax.axvline(0, color=_INK, lw=0.8, alpha=0.6, zorder=3)
        ax.set_xlim(0, full_len)
        ax.set_ylabel(name, fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.2)
        sparklines.append((solid, vline, trace["step"], series))

    return {
        "trail": trail,
        "heading_marker": heading_marker,
        "hud_text": hud_text,
        "banner_text": banner_text,
        "sparklines": sparklines,
    }


def _update_panel(handles, trace, reward_data, t, full_len):
    ep_len = len(trace["step"])
    tt = min(t, ep_len - 1)
    finished = t >= ep_len - 1

    handles["trail"].set_data(trace["x"][: tt + 1], trace["y"][: tt + 1])

    x, y, hd = trace["x"][tt], trace["y"][tt], trace["heading"][tt]
    size = 0.5
    tip = (x + size * np.cos(hd), y + size * np.sin(hd))
    left = (x + 0.45 * size * np.cos(hd + 2.4), y + 0.45 * size * np.sin(hd + 2.4))
    right = (x + 0.45 * size * np.cos(hd - 2.4), y + 0.45 * size * np.sin(hd - 2.4))
    handles["heading_marker"].set_xy([tip, left, right])

    dist = reward_data["dist"][tt]
    handles["hud_text"].set_text(
        f"step {tt}/{full_len}   dist {dist:.2f}   "
        f"v {trace['v'][tt]:+.2f}   w {trace['w'][tt]:+.2f}"
    )

    if finished:
        reached = bool(reward_data["reached"][ep_len - 1])
        banner = "REACHED" if reached else "TRUNCATED"
        bc = "#3a9d5d" if reached else "#b5423f"
        banner_text = handles["banner_text"]
        banner_text.set_text(banner)
        banner_text.set_bbox(dict(boxstyle="round", fc=bc, ec="none"))
        banner_text.set_visible(True)

    for solid, vline, steps, series in handles["sparklines"]:
        solid.set_data(steps[: tt + 1], series[: tt + 1])
        vline.set_xdata([tt, tt])


def _fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].reshape(h, w, 3).copy()


def _render_seed_video(
    seed: int, clone: dict, residual: dict, outdir: str, fps: int,
    left_label: str = "CLONE", right_label: str = "CLONE + TD3 RESIDUAL",
    left_color: str = _GRAY, right_color: str = _BLUE, prefix: str = "dashboard",
    fail_word: str = "stalls",
) -> str:
    """Render a two-column dashboard. `clone`/`residual` are just the left/right traces;
    the labels/colors/prefix parameterize which arms they are (defaults reproduce the
    original clone-vs-residual video exactly).
    """
    reward_c = recompute_reward(
        clone["x"], clone["y"], clone["heading"], clone["v"], clone["w"], clone["goal"],
    )
    reward_r = recompute_reward(
        residual["x"], residual["y"], residual["heading"], residual["v"], residual["w"],
        residual["goal"],
    )
    full_len = max(len(clone["step"]), len(residual["step"])) - 1

    xlim, ylim = _axis_limits(
        [clone["x"], residual["x"]], [clone["y"], residual["y"]], clone["goal"], _TOLERANCE,
    )

    clone_reached = bool(reward_c["reached"][-1])
    residual_reached = bool(reward_r["reached"][-1])
    clone_title = f"{left_label} ({'reaches' if clone_reached else fail_word})"
    residual_title = f"{right_label} ({'reaches' if residual_reached else fail_word})"

    fig = plt.figure(figsize=(9.5, 7.5), dpi=100)
    outer = fig.add_gridspec(1, 2, wspace=0.28)
    gs_left = outer[0].subgridspec(4, 1, height_ratios=[2.6, 1, 1, 1], hspace=0.55)
    gs_right = outer[1].subgridspec(4, 1, height_ratios=[2.6, 1, 1, 1], hspace=0.55)
    left = _setup_panel(
        fig, gs_left, clone, reward_c, left_color, clone_title, xlim, ylim, full_len
    )
    right = _setup_panel(
        fig, gs_right, residual, reward_r, right_color, residual_title, xlim, ylim, full_len
    )
    fig.suptitle(f"seed {seed}", fontsize=11, color=_INK)

    frames = []
    try:
        for t in range(full_len + 1):
            _update_panel(left, clone, reward_c, t, full_len)
            _update_panel(right, residual, reward_r, t, full_len)
            frames.append(_fig_to_rgb(fig))
    finally:
        plt.close(fig)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{prefix}-{seed}.mp4")
    encode_video(frames, out_path, fps)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", required=True, help="comma-separated seeds")
    parser.add_argument("--clone", default="data/clone.pt")
    parser.add_argument("--residual-model", default="data/residual_td3.zip")
    parser.add_argument("--algo", default="td3", choices=["td3", "sac"])
    parser.add_argument("--libraries", default="data/libraries_v0.npz")
    parser.add_argument("--figdir", default="docs/journey/figures")
    parser.add_argument("--outdir", default="docs/journey/videos")
    parser.add_argument("--fps", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--compare", default="clone-residual",
        choices=["clone-residual", "residual-vanilla", "clone-vanilla"],
        help="which two arms to put side by side (default: clone vs clone+residual)",
    )
    parser.add_argument("--vanilla-model", default="data/vanilla_td3_400k.zip")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for seed in seeds:
        try:
            clone, residual = ensure_traces(
                seed, args.figdir,
                clone_path=args.clone, residual_model_path=args.residual_model,
                algo=args.algo, libraries_path=args.libraries, device=args.device,
            )
            if args.compare in ("residual-vanilla", "clone-vanilla"):
                vanilla = ensure_vanilla_trace(
                    seed, args.figdir, vanilla_model_path=args.vanilla_model,
                    algo=args.algo, libraries_path=args.libraries, device=args.device,
                )
        except FileNotFoundError as e:
            print(
                f"error: {e}. Train the missing checkpoint first "
                f"(scripts/train_clone.py / scripts/train_residual.py / "
                f"scripts/train_vanilla.py).",
                file=sys.stderr,
            )
            return 1
        if args.compare == "residual-vanilla":
            out_path = _render_seed_video(
                seed, residual, vanilla, args.outdir, args.fps,
                left_label="CLONE + TD3 RESIDUAL", right_label="VANILLA TD3 (from scratch)",
                left_color=_BLUE, right_color=_ORANGE,
                prefix="residual-vs-vanilla", fail_word="misses",
            )
        elif args.compare == "clone-vanilla":
            out_path = _render_seed_video(
                seed, clone, vanilla, args.outdir, args.fps,
                left_label="CLONE (DeePC surrogate)", right_label="VANILLA TD3 (from scratch)",
                left_color=_GRAY, right_color=_ORANGE,
                prefix="clone-vs-vanilla", fail_word="stalls",
            )
        else:
            out_path = _render_seed_video(seed, clone, residual, args.outdir, args.fps)
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

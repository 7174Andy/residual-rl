#!/usr/bin/env python
"""Trajectory + forward-velocity companion figure for a showcase seed's video.

Default inputs are the committed per-step trace CSVs written by
`scripts/eval_seed_showcase.py` (`docs/journey/figures/traj_<seed>_{clone,
residual}.csv`, columns ``step,x,y,heading,v,w,goal_x,goal_y``), so the figure
regenerates with no model inference:

    uv run python scripts/plot_seed_traces.py --seed 4104626029

Two panels sharing one x-position-free layout (never a dual-axis chart): the
XY closed-loop trajectory (top) and the forward-velocity trace v(t) (bottom) --
the "v-collapse" failure mode is a story about the *v* channel specifically,
so the companion figure makes that channel explicit instead of leaving it as
prose next to the video.
"""
from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_GRAY = "#898781"
_INK = "#52514e"
_TOLERANCE = 0.5


def _read_trace(path: str) -> dict:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {
        "step": np.array([int(r["step"]) for r in rows]),
        "x": np.array([float(r["x"]) for r in rows]),
        "y": np.array([float(r["y"]) for r in rows]),
        "v": np.array([float(r["v"]) for r in rows]),
        "goal": np.array([float(rows[0]["goal_x"]), float(rows[0]["goal_y"])]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--figdir", default="docs/journey/figures")
    p.add_argument("--residual-label", default="clone + TD3 (200k)")
    p.add_argument("--residual-color", default="#3987e5")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    clone = _read_trace(f"{args.figdir}/traj_{args.seed}_clone.csv")
    residual = _read_trace(f"{args.figdir}/traj_{args.seed}_residual.csv")
    out = args.out or f"{args.figdir}/seed_{args.seed}_metrics.png"

    fig, (ax_xy, ax_v) = plt.subplots(2, 1, figsize=(4.2, 6.2), height_ratios=[1.3, 1])

    ax_xy.plot(clone["x"], clone["y"], color=_GRAY, lw=1.6, ls="--", label="clone", zorder=2)
    ax_xy.plot(residual["x"], residual["y"], color=args.residual_color, lw=1.8,
               label=args.residual_label, zorder=3)
    ax_xy.scatter([clone["x"][0]], [clone["y"][0]], color=_INK, s=30, zorder=4, marker="o")
    ax_xy.annotate("start", (clone["x"][0], clone["y"][0]), textcoords="offset points",
                   xytext=(6, 4), fontsize=8, color=_INK)
    goal = clone["goal"]
    ax_xy.scatter([goal[0]], [goal[1]], color="#e34948", s=70, zorder=4, marker="*")
    ax_xy.add_patch(mpatches.Circle(goal, _TOLERANCE, fill=False, ls="--",
                                     color="#e34948", lw=1.0, zorder=1))
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_title(f"seed {args.seed} — trajectory", fontsize=10)
    ax_xy.grid(alpha=0.25)
    ax_xy.legend(frameon=False, loc="best", fontsize=8)

    ax_v.plot(clone["step"], clone["v"], color=_GRAY, lw=1.4, ls="--", label="clone", zorder=2)
    ax_v.plot(residual["step"], residual["v"], color=args.residual_color, lw=1.6,
              label=args.residual_label, zorder=3)
    ax_v.axhline(0.0, color="#c3c2b7", lw=0.8, zorder=1)
    ax_v.set_xlabel("step")
    ax_v.set_ylabel("forward velocity  v")
    ax_v.set_title("v(t) — the v-collapse channel", fontsize=10)
    ax_v.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

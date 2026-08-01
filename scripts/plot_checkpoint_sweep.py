#!/usr/bin/env python
"""Plot deterministic reach rate and return vs training steps, residual vs vanilla.

Two panels over the CSV `sweep_checkpoints.py` writes (one row per arm x training
seed x checkpoint), bands = +/-1 std over the 5 training seeds:

    uv run python scripts/plot_checkpoint_sweep.py --csv data/checkpoint_sweep.csv

DeePC/clone are drawn as flat reference lines: both are deterministic and have no
training curve, so their reach rate is a constant the RL arms must clear.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

# Same palette and rationale as scripts/plot_robustness.py, so the arms keep one
# identity across every figure in the journey. The SAC arm's violet is slot 7 of the
# dataviz-skill categorical order, validated:
#   node scripts/validate_palette.js "#898781,#3987e5,#c1701c,#4a3aa7" --mode light
#   -> every check PASSes (worst adjacent CVD dE 15.9, all >= 3:1 contrast) except
#      the gray's chroma floor, the same deliberate baseline exception as elsewhere.
#      Aqua and magenta failed contrast; green collides with orange under protanopia.
_RESIDUAL = "#3987e5"
_VANILLA = "#c1701c"
_RESIDUAL_SAC = "#4a3aa7"
_BASELINE = "#898781"
_INK = "#52514e"
_GRID = "#e1e0d9"

# Fixed order and fixed hue per arm -- never cycled, so an arm keeps its colour
# whether or not the other arms are present in the CSV.
ARMS = [("residual", _RESIDUAL, "clone + residual TD3 (frac 2.0)"),
        ("residual_sac", _RESIDUAL_SAC, "clone + residual SAC (frac 2.0)"),
        ("vanilla", _VANILLA, "vanilla TD3")]
DEEPC_REACH = 30 / 78  # docs/journey/figures/reach_rates.csv -- DeePC and clone both


def _read(paths: str) -> list[dict]:
    """Rows from one or more comma-separated CSVs (one sweep invocation per arm)."""
    rows = []
    for path in paths.split(","):
        with open(path.strip()) as f:
            rows += [{"arm": r["arm"], "train_seed": int(r["train_seed"]),
                      "steps": int(r["steps"]), "reach_rate": float(r["reach_rate"]),
                      "return_mean": float(r["return_mean"]), "n": int(r["n"])}
                     for r in csv.DictReader(f)]
    return rows


def _series(rows, arm: str, field: str):
    by_step = defaultdict(list)
    for r in rows:
        if r["arm"] == arm:
            by_step[r["steps"]].append(r[field])
    xs = sorted(by_step)
    return (np.array(xs),
            np.array([np.mean(by_step[x]) for x in xs]),
            np.array([np.std(by_step[x]) for x in xs]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="data/checkpoint_sweep.csv",
                   help="one path, or several comma-separated to merge arms")
    p.add_argument("--out", default="docs/journey/figures/checkpoint_sweep.png")
    args = p.parse_args()

    rows = _read(args.csv)
    n_eval = rows[0]["n"]
    n_train = len({r["train_seed"] for r in rows})
    present = {r["arm"] for r in rows}
    arms = [a for a in ARMS if a[0] in present]
    unknown = present - {a[0] for a in ARMS}
    if unknown:
        raise SystemExit(f"no colour assigned for arm(s) {sorted(unknown)} -- add them "
                         f"to ARMS rather than letting a hue be cycled")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=140)
    panels = [("reach_rate", "reach rate", "Deployed-policy reach rate"),
              ("return_mean", "mean episode return", "Deployed-policy return")]
    for ax, (field, ylabel, title) in zip(axes, panels):
        for arm, color, _label in arms:
            xs, mean, std = _series(rows, arm, field)
            ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.16,
                            lw=0, zorder=2)
            ax.plot(xs, mean, color=color, lw=2.0, marker="o", ms=5.5, mec="white",
                    mew=1.0, zorder=3)
        if field == "reach_rate":
            ax.axhline(DEEPC_REACH, color=_BASELINE, lw=1.4, ls="--", zorder=1)
            ax.annotate("DeePC / clone (0.385)", xy=(1.0, DEEPC_REACH),
                        xycoords=("axes fraction", "data"), xytext=(-4, 5),
                        textcoords="offset points", ha="right", fontsize=7.5,
                        color=_BASELINE)
            ax.set_ylim(-0.03, 1.05)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xscale("log")
        # Every checkpoint is plotted, but only the octave-ish ones get a label --
        # 16 labels on a log axis collide into a smear.
        all_steps = sorted({r["steps"] for r in rows})
        ax.set_xticks([n for n in (5_000, 10_000, 25_000, 50_000, 100_000,
                                   200_000, 400_000) if n in all_steps])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(title, fontsize=10.5, color=_INK, fontweight="bold", pad=8)
        ax.set_xlabel("environment steps", fontsize=9, color=_INK)
        ax.set_ylabel(ylabel, fontsize=9, color=_INK)
        ax.grid(True, color=_GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)
        ax.tick_params(labelsize=8, colors=_INK)

    # Direct labels on the left panel, so arm identity is never colour-alone.
    for arm, _color, label in arms:
        xs, mean, _ = _series(rows, arm, "reach_rate")
        axes[0].annotate(label.split(" (")[0], xy=(xs[0], mean[0]), xytext=(6, -10),
                         textcoords="offset points", fontsize=7.5, color=_INK)
    handles = [plt.Line2D([], [], color=c, lw=2.0, marker="o", ms=5.5, mec="white",
                          mew=1.0, label=lab) for _, c, lab in arms]
    fig.legend(handles=handles, loc="lower center", ncol=len(arms), frameon=False, fontsize=9,
               labelcolor=_INK, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(
        f"Sample efficiency, deployed policy — {n_eval} eval seeds, "
        f"bands = ±1 std over {n_train} training seeds",
        fontsize=11, color=_INK, y=1.0,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

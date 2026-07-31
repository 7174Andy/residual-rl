#!/usr/bin/env python
"""Plot robustness degradation curves: reach rate vs plant perturbation level.

Three panels, one per perturbation family (state noise / actuator gain / control
latency), three arms per panel. The RL arms carry a mean +/- std band across their 5
training seeds; the clone has a single model so it draws a line only.

Input is the CSV `robustness_sweep.py` writes (one row per arm x training seed x
perturbation x level):

    uv run python scripts/plot_robustness.py --csv data/robustness.csv

Each panel's nominal (unperturbed) point is the shared `perturbation=nominal` row,
placed at that family's neutral x (sigma 0, gain 1.0, k 0), so all three curves
start from the same anchor.
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

# dataviz-skill palette, validated:
#   node scripts/validate_palette.js "#898781,#3987e5,#c1701c" --mode light
#   -> lightness/CVD/normal-vision/contrast all PASS; the lone FAIL is the gray's
#      chroma floor, deliberate for the baseline arm (same choice, same reason, as
#      scripts/plot_reach_rates.py). Secondary encoding: legend + direct labels.
_CLONE = "#898781"   # baseline: frozen DeePC surrogate
_RESIDUAL = "#3987e5"
_VANILLA = "#c1701c"
_INK = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"

ARMS = [("clone", _CLONE, "clone (DeePC surrogate)"),
        ("residual", _RESIDUAL, "clone + residual (frac 2.0)"),
        ("vanilla", _VANILLA, "vanilla TD3")]

# family -> (axis label, x of the nominal anchor, panel title)
FAMILIES = [
    ("state_noise", "process noise  $\\sigma_{pos}$  (units/step)", 0.0,
     "Unmeasured state disturbance"),
    ("actuator_gain", "actuator gain on $v$  (1.0 = calibrated)", 1.0,
     "Actuator calibration error"),
    ("latency", "control delay  $k$  (steps)", 0.0, "Control latency"),
]


def _read(path: str) -> list[dict]:
    with open(path) as f:
        return [{
            "arm": r["arm"], "train_seed": int(r["train_seed"]),
            "perturbation": r["perturbation"], "level": float(r["level"]),
            "reach_rate": float(r["reach_rate"]), "n": int(r["n"]),
        } for r in csv.DictReader(f)]


def _series(rows, arm: str, family: str, nominal_x: float):
    """(levels, mean, std) for one arm in one panel, nominal spliced in at its x."""
    by_level = defaultdict(list)
    for r in rows:
        if r["arm"] != arm:
            continue
        if r["perturbation"] == "nominal":
            by_level[nominal_x].append(r["reach_rate"])
        elif r["perturbation"] == family:
            by_level[r["level"]].append(r["reach_rate"])
    xs = sorted(by_level)
    mean = np.array([np.mean(by_level[x]) for x in xs])
    std = np.array([np.std(by_level[x]) for x in xs])
    return np.array(xs), mean, std


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="data/robustness.csv")
    p.add_argument("--out", default="docs/journey/figures/robustness.png")
    args = p.parse_args()

    rows = _read(args.csv)
    n_eval = rows[0]["n"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=140)
    for ax, (family, xlabel, nominal_x, title) in zip(axes, FAMILIES):
        for arm, color, _label in ARMS:
            xs, mean, std = _series(rows, arm, family, nominal_x)
            if not len(xs):
                continue
            if std.any():  # 5 training seeds -> spread band; clone has none
                ax.fill_between(xs, mean - std, mean + std, color=color,
                                alpha=0.16, lw=0, zorder=2)
            ax.plot(xs, mean, color=color, lw=2.0, marker="o", ms=5.5,
                    mec="white", mew=1.0, zorder=3)
        ax.axvline(nominal_x, color=_MUTED, lw=0.9, ls=":", alpha=0.7, zorder=1)
        ax.set_title(title, fontsize=10.5, color=_INK, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, fontsize=9, color=_INK)
        ax.set_ylim(-0.03, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, color=_GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)
        ax.tick_params(labelsize=8, colors=_INK)

    axes[0].set_ylabel("reach rate", fontsize=9, color=_INK)
    # Direct labels at each curve's right end in the latency panel (secondary
    # encoding, so identity is never colour-alone), plus the legend below.
    ax = axes[-1]
    for arm, _color, label in ARMS:
        xs, mean, _ = _series(rows, arm, "latency", 0.0)
        if len(xs):
            ax.annotate(label.split(" (")[0], xy=(xs[-1], mean[-1]),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=7.5, color=_INK, va="center")
    handles = [plt.Line2D([], [], color=c, lw=2.0, marker="o", ms=5.5,
                          mec="white", mew=1.0, label=lab) for _, c, lab in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=_INK, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(
        f"Robustness to plant perturbation — {n_eval} eval seeds, "
        f"bands = ±1 std over 5 training seeds",
        fontsize=11, color=_INK, y=1.0,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

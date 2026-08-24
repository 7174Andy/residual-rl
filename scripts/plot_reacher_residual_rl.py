"""The 200k vs 400k figure for the Reacher RL arms (journey 13).

A: training return over the full 400k runs (rolling mean 100), line at 200k.
B: reach rate, 200k vs 400k checkpoints of both RL arms + frozen baselines.
C: best->final drift, same rows.

Panels B/C read the eval CSVs written by `scripts/eval_reacher_residual.py`;
CSVs are gitignored repo-wide, so rerun that eval (~13 min) on a fresh clone.
Panel A reads the SB3 monitor CSVs from `data/` -- rerun
`scripts/train_reacher_residual.py --steps 400000` and
`scripts/train_reacher_vanilla.py --steps 400000` to recreate them.

    uv run python scripts/plot_reacher_residual_rl.py
"""
from __future__ import annotations

import argparse
import csv
import io
import os
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rl.stats import wilson_ci  # noqa: E402

BLUE, BLUE_LT = "#2a78d6", "#9dc1ea"
ORANGE, ORANGE_LT = "#eb6834", "#f5b39a"
INK, INK2, MUTED, CRITICAL = "#0b0b0b", "#52514e", "#b8b7b2", "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})


def monitor_curve(path: str, window: int = 100):
    with open(path) as f:
        body = "".join(ln for ln in f if not ln.startswith("#"))
    rows = list(csv.DictReader(io.StringIO(body)))
    r = np.array([float(x["r"]) for x in rows])
    steps = np.cumsum([int(x["l"]) for x in rows])
    k = np.ones(window) / window
    return steps[window - 1:], np.convolve(r, k, mode="valid")


def eval_rows(path: str) -> dict[str, list[dict]]:
    rows = defaultdict(list)
    with open(path) as f:
        next(f)  # producing-command comment
        for x in csv.DictReader(f):
            rows[x["controller"]].append(x)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-200k",
                   default="docs/reference/reacher_residual_200k_rerun.csv")
    p.add_argument("--eval-400k",
                   default="docs/reference/reacher_residual_400k.csv")
    p.add_argument("--residual-monitor",
                   default="data/reacher_residual_400k.monitor.csv")
    p.add_argument("--vanilla-monitor",
                   default="data/reacher_vanilla_400k.monitor.csv")
    p.add_argument("--out", default="docs/reference/reacher_residual_rl.png")
    args = p.parse_args()

    fig, ax = plt.subplots(1, 3, figsize=(13.0, 4.1))

    for lab, path, c in (
        ("clone + residual (SAC)", args.residual_monitor, BLUE),
        ("vanilla RL (SAC)", args.vanilla_monitor, ORANGE),
    ):
        s, r = monitor_curve(path)
        ax[0].plot(s / 1e3, r, color=c, zorder=3, label=lab)
    ax[0].axvline(200, color=INK2, lw=1.0, ls="--", zorder=2)
    ax[0].text(205, 0.98, "200k checkpoint",
               transform=ax[0].get_xaxis_transform(), fontsize=8, color=INK2,
               ha="left", va="top")
    ax[0].set_xlabel("training steps (k)")
    ax[0].set_ylabel("episode return (rolling mean, 100 ep)")
    ax[0].set_title("A · Training return over the full 400k", loc="left",
                    color=INK)
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")

    r200, r400 = eval_rows(args.eval_200k), eval_rows(args.eval_400k)
    n = len(r400["clone"])
    rows = [
        ("Select-DPC", r400["Select-DPC"], MUTED),
        ("clone", r400["clone"], MUTED),
        ("residual · 200k", r200["clone + residual"], BLUE_LT),
        ("residual · 400k", r400["clone + residual"], BLUE),
        ("vanilla · 200k", r200["vanilla RL"], ORANGE_LT),
        ("vanilla · 400k", r400["vanilla RL"], ORANGE),
    ]
    y = np.arange(len(rows))

    for i, (_lab, r, c) in enumerate(rows):
        k = sum(int(x["reached"]) for x in r)
        lo, hi = wilson_ci(k, n)
        rate = 100 * k / n
        ax[1].barh(i, rate, 0.55, color=c, zorder=3)
        ax[1].errorbar(rate, i, xerr=[[rate - 100 * lo], [100 * hi - rate]],
                       fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3,
                       zorder=4)
        ax[1].annotate(f"{k}/{n}", (100 * hi + 2, i), ha="left", va="center",
                       fontsize=8, color=INK2, zorder=5)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([lab for lab, _, _ in rows], fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlim(0, 118)
    ax[1].set_xlabel("reach rate (%)  ·  Wilson 95% CI")
    ax[1].set_title("B · Reach rate, 120 frozen scenarios", loc="left",
                    color=INK)

    for i, (_lab, r, c) in enumerate(rows):
        b = np.median([float(x["best"]) for x in r])
        f = np.median([float(x["final"]) for x in r])
        ax[2].barh(i, f / max(b, 1e-9), 0.55, color=c, zorder=3)
        ax[2].annotate(f"{f/max(b,1e-9):.1f}x   ({b*1e3:.1f} → {f*1e3:.1f} mm)",
                       (f / max(b, 1e-9) + 0.05, i), ha="left", va="center",
                       fontsize=8, color=INK2, zorder=5)
    ax[2].axvline(1.0, color=CRITICAL, lw=1.2, ls="--", zorder=4)
    ax[2].set_yticks(y)
    ax[2].set_yticklabels([lab for lab, _, _ in rows], fontsize=8)
    ax[2].invert_yaxis()
    ax[2].set_xlim(0, 3.6)
    ax[2].set_xlabel("best → final ratio (1.0 = holds position)")
    ax[2].set_title("C · The drift — does it HOLD the goal?", loc="left",
                    color=INK)

    for a in ax:
        a.grid(True, zorder=0)
        a.set_axisbelow(True)
    fig.suptitle("Reacher residual RL — 200k vs 400k steps, 120 held-out "
                 "scenarios, all rows evaluated under identical code",
                 x=0.005, ha="left", color=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

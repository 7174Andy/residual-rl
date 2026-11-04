"""Panda box RL arms: training return and deployed reach rate vs training step.

    uv run python scripts/plot/plot_panda_rl_curves.py

Left: the behaviour policy's episode return from each run's SB3 Monitor CSV,
rolling-mean over `--window` episodes, plotted against cumulative env steps (not
episodes -- arms whose episodes end early would otherwise look slower). The mean
is trailing over the episodes seen so far until `--window` have accumulated, so
the curve starts at the 10th episode (`rl.curves`), not ~15k steps in: a
100-episode window first fills after the warm arm has already collapsed and hid
the dip. Each clone arm's curve starts at step 0 with its clone's deterministic
mean return on the 78 box scenarios -- the same policy and scenarios the reach
rate scores at step 0 -- and that value counts as episode 0 of the trailing
window. Caveat: the residual's step-0 behaviour policy is far worse than its
clone (-28.8 vs 47.3 when sampled; SAC's zero-init leaves sigma ~1), so part of
its first drop is exploration noise. The warm arms lose nothing to it (54.9 vs 54.0).
Right: the DETERMINISTIC policy's reach on the 78 frozen box scenarios at every
50k checkpoint (`data/panda_rl_sweep.csv`, written by `train_panda_rl.py --eval`),
with the clone arms' step-0 score from `data/panda_results.csv`. The lines are
dashed because nothing is measured between checkpoints: the warm arm's
82% -> 9% is not a gradual decline (every training episode from step 150 to
10k missed).
"""
from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rl.curves import return_curve  # noqa: E402

# (arm, label, color, marker, step-0 method in panda_results.csv or None)
ARMS = [
    ("residual", "Residual (Eq. 18)", "#2a78d6", "o", "panda_clone_box_r2"),
    ("warm", "Warm start (Eq. 17)", "#eb6834", "s", "panda_clone_box_r2_squash"),
    ("vanilla", "Vanilla (env obs)", "#1baf7a", "D", None),
    ("warm_anneal", "Warm start + entropy anneal from 100k", "#eda100", "^",
     "panda_clone_box_r2_squash"),
]
ANNEAL_FROM_K = 100
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})


def reach_by_method(path: str) -> dict[str, int]:
    out: dict[str, int] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["method"]] = out.get(row["method"], 0) + (row["reached"] == "True")
    return out


def return_by_method(path: str) -> dict[str, float]:
    tot: dict[str, list[float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            tot.setdefault(row["method"], []).append(float(row["ret"]))
    return {m: float(np.mean(v)) for m, v in tot.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--sweep", default="data/panda_rl_sweep.csv")
    p.add_argument("--results", default="data/panda_results.csv")
    p.add_argument("--expert", type=int, default=69, help="Select-DPC on the box 78")
    p.add_argument("--out", default="docs/reference/panda_rl_curves.png")
    args = p.parse_args()

    sweep, results = reach_by_method(args.sweep), reach_by_method(args.results)
    base_ret = return_by_method(args.results)
    ks = range(50, 401, 50)
    fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)

    for arm, label, color, marker, base in ARMS:
        x, _ep, y = return_curve(f"data/panda_{arm}_400k.monitor.csv", args.window,
                                 step0=base_ret.get(base) if base else None)
        ax_r.plot(x / 1e3, y, color=color, label=label)

        xs = [k for k in ks if f"panda_{arm}_{k}k" in sweep]
        ys = [100 * sweep[f"panda_{arm}_{k}k"] / 78 for k in xs]
        if base and base in results:
            xs, ys = [0, *xs], [100 * results[base] / 78, *ys]
        ax_p.plot(xs, ys, color=color, marker=marker, markersize=8, ls="--", lw=1.4,
                  markeredgecolor="#fcfcfb", markeredgewidth=2, label=label)

    for ax in (ax_r, ax_p):
        ax.axvline(ANNEAL_FROM_K, color=MUTED, lw=1, ls=":", zorder=0)
    ax_r.text(ANNEAL_FROM_K + 4, ax_r.get_ylim()[0] + 4, "anneal starts",
              color=INK2, fontsize=8)
    for ax_val, name in ((args.expert, "Expert (Select-DPC)"),
                         (results.get("panda_clone_box_r2", 0), "Clone r2")):
        y = 100 * ax_val / 78
        ax_p.axhline(y, color=MUTED, lw=1, ls="--", zorder=0)
        ax_p.text(402, y, f" {name} {y:.0f}%", color=INK2, va="center", fontsize=8)

    ax_r.set(xlabel="Training step (k)", ylabel=f"Episode return (trailing {args.window}-ep mean)",
             title="Training return (behaviour policy)")
    ax_p.set(xlabel="Training step (k)", ylabel="Reached (% of 78 box scenarios)",
             title="Deployed reach rate, measured every 50k (seed 0)", ylim=(0, 105), xlim=(-10, 400))
    for ax in (ax_r, ax_p):
        ax.grid(True, axis="y")
        ax.title.set_ha("left"), ax.title.set_x(0)
    handles, labels = ax_p.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 0.93, 0.90))
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

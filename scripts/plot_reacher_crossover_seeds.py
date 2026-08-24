"""Multi-seed reacher crossover: residual - vanilla reach rate, seeds 0-4.

Journey 13 measured the sample-efficiency crossover on ONE training run per arm
and flagged the ~150k crossing as single-seed. This aggregates five seeds' sweep
CSVs (each from `scripts/sweep_reacher_checkpoints.py` on that seed's checkpoint
dirs) into one figure: per-seed difference curves, their mean, and each seed's
crossing point. Verdict: the crossover exists on every seed; its location is
seed-luck (first-touch 75k-225k).

    uv run python scripts/plot_reacher_crossover_seeds.py \
        docs/reference/reacher_crossover.csv data/reacher_ckpt_seeds/reacher_crossover_s*.csv

Writes the figure and a combined CSV (seed column prepended) next to --out.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, INK2, FAINT, MUTED2 = ("#2a78d6", "#eb6834", "#52514e",
                                     "#9db9d9", "#8a8a86")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+",
                   help="per-seed sweep CSVs, in seed order (seed = position)")
    p.add_argument("--out", default="docs/reference/reacher_crossover_seeds.png")
    args = p.parse_args()

    rows_by_seed: dict[int, list[dict]] = {}
    reached: dict[int, dict] = defaultdict(lambda: defaultdict(dict))
    n = None
    for seed, path in enumerate(args.csvs):
        with open(path) as f:
            next(f)  # producing-command comment
            rows_by_seed[seed] = list(csv.DictReader(f))
        for row in rows_by_seed[seed]:
            reached[seed][row["arm"]][int(row["steps"])] = int(row["reached"])
            n = int(row["n"])

    seeds = sorted(reached)
    steps = np.array(sorted(reached[0]["residual"]))
    xs = steps / 1e3
    D = np.array([[100 * (reached[s]["residual"][st] - reached[s]["vanilla"][st]) / n
                   for st in steps] for s in seeds])

    # first checkpoint where vanilla has caught the residual (diff <= 0)
    cross = [xs[np.argmax(D[i] <= 0)] for i in range(len(seeds))]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, s in enumerate(seeds):
        ax.plot(xs, D[i], color=FAINT, lw=1.1, zorder=2)
        ax.annotate(f"s{s}", (xs[-1] + 4, D[i, -1]), fontsize=7, color=MUTED2,
                    va="center")
    ax.plot(xs, D.mean(axis=0), color=BLUE, lw=2.6, marker="o", ms=4, zorder=4,
            label=f"mean of {len(seeds)} seeds")
    ax.axhline(0, color=INK2, lw=1.0)
    ax.scatter(cross, [0] * len(cross), marker="|", s=120, color=INK2, zorder=5,
               label="per-seed first catch-up")
    ax.text(0.02, 0.95, "residual ahead", transform=ax.transAxes, fontsize=8,
            color=BLUE)
    ax.text(0.98, 0.04, "vanilla ahead", transform=ax.transAxes, fontsize=8,
            color=ORANGE, ha="right")
    ax.set_xlabel("training steps (k)")
    ax.set_ylabel("reach-rate difference (pp)")
    ax.set_title(f"Reacher crossover, {len(seeds)} training seeds x {n} frozen "
                 "scenarios, greedy", loc="left", fontsize=9, color=INK2)
    ax.grid(True, color="#e8e7e3", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")

    out_csv = args.out.replace(".png", ".csv")
    with open(out_csv, "w") as fh:
        fh.write("# scripts/plot_reacher_crossover_seeds.py\n")
        fh.write("seed,arm,steps,reached,n,best_med_mm,final_med_mm\n")
        for seed in seeds:
            for r in rows_by_seed[seed]:
                fh.write(f"{seed},{r['arm']},{r['steps']},{r['reached']},"
                         f"{r['n']},{r['best_med_mm']},{r['final_med_mm']}\n")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()

"""DAWN (arXiv:2602.10539) on the Reacher warm-start and residual arms: deployed
reach over training, per component (soft SAC, seed 0, 100k steps).

    uv run python scripts/plot/plot_dawn.py

Runs (`data/repro/sweep_<stem>.csv`, 2k checkpoints, 120 frozen scenarios, greedy):
baseline = `q_warm` / `q_res`; warmup = `--warmup-steps 20000` (the prior acting
alone, sigma 0); LayerNorm = `--critic-layernorm`; DAWN = both; wide = DAWN with
`--warmup-sigma 0.5` (warm) / `0.25` (residual, = 0.5 torque at residual_frac 2).
The 20k warmup steps are NOT on the x-axis: shift those curves right by 20k to
charge them to the sample budget.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N = 120
PANELS = (  # title, prior reach at step 0, stem suffixes keyed by variant
    ("Warm-start RL", 79, "the clone", {"baseline": "q_warm", "warmup": "dawn_w_warmup",
                                        "ln": "dawn_w_ln", "dawn": "dawn_w",
                                        "wide": "dawn_w_wide"}),
    ("Residual RL", 82, "the base clone", {"baseline": "q_res", "warmup": "dawn_r_warmup",
                                           "ln": "dawn_r_ln", "dawn": "dawn_r",
                                           "wide": "dawn_r_wide"}),
)
VARIANTS = (  # key, label, colour (categorical order), width
    ("baseline", "no fix (baseline)", "#2a78d6", 2.4),
    ("warmup", "warmup data only", "#eb6834", 1.6),
    ("ln", "LayerNorm critic only", "#1baf7a", 1.6),
    ("dawn", "DAWN (warmup + LayerNorm)", "#4a3aa7", 1.6),
    ("wide", "DAWN, wide warmup", "#e87ba4", 1.6),
)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9.5,
})


def reach(path: str, prior: int) -> list[tuple[int, int]]:
    k: dict[int, int] = defaultdict(int)
    with open(path) as f:
        for r in csv.DictReader(f):
            k[int(r["steps"])] += int(r["reached"])
    return sorted({0: prior, **k}.items())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="data/repro")
    p.add_argument("--out", default="docs/journey/figures/dawn_reach.png")
    args = p.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    for ax, (title, prior, prior_name, stems) in zip(axes, PANELS):
        for key, label, color, lw in VARIANTS:
            pts = reach(f"{args.dir}/sweep_{stems[key]}.csv", prior)
            xs, ys = zip(*pts)
            worst = min(v for s, v in pts if s > 0)
            mean = sum(v for s, v in pts if s >= 10_000) / sum(1 for s, _ in pts if s >= 10_000)
            ax.plot(xs, ys, color=color, lw=lw, zorder=3 if key == "baseline" else 2,
                    label=f"{label}: worst {worst}, {ys[-1]} @100k, mean {mean:.0f}")
        ax.axhline(prior, color=MUTED, lw=1, ls="--", zorder=0)
        ax.text(99_000, prior - 7, f"{prior_name}: {prior}", color=INK2, fontsize=8.5, ha="right",
                zorder=4, bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=1.5))
        ax.set(xlim=(0, 100_000), ylim=(0, N + 4), xlabel="RL training steps")
        ax.set_xticks([0, 25_000, 50_000, 75_000, 100_000], ["0", "25k", "50k", "75k", "100k"])
        ax.grid(axis="y", color=MUTED, alpha=0.4, lw=0.6)
        ax.set_title(title, loc="left", fontsize=11)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8.5)
    axes[0].set_ylabel(f"scenarios reached (of {N}), greedy")
    fig.suptitle("DAWN components on Reacher: deployed reach over training (seed 0; "
                 "mean = average reach over 10k–100k)", x=0.01, ha="left", fontsize=11.5,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

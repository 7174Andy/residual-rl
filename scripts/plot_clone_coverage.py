"""The coverage ablation: clone-vs-expert disagreement, on- vs off-policy.

Journey 14's figure. For each clone, the median ||u_clone - u_expert|| is
measured at states the EXPERT drives to (the BC training regime) and at states
the CLONE drives to (the deployment regime) — 15 held-out seeds each, expert
labels from prime_buffer+act (`measure_clone_disagreement.py`). The reacher
panel restates journey 13's measurement for the cross-system comparison; the
two systems' action units differ, so the panels share the concept, not a scale.

    uv run python scripts/plot_clone_coverage.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, INK, INK2, MUTED = ("#2a78d6", "#eb6834", "#0b0b0b", "#52514e",
                                  "#b8b7b2")
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6,
})

# journey 13's measurement (reacher, pre-DAgger BC clone) — action units differ
# from the unicycle's, so it gets its own panel.
REACHER = ("expert-rollout data\n(pre-DAgger BC)", 0.1025, 0.2815)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hybrid", default="data/disagree_hybrid.npz")
    p.add_argument("--onpolicyonly", default="data/disagree_onpolicyonly.npz")
    p.add_argument("--noonpolicy", default="data/disagree_noonpolicy.npz")
    p.add_argument("--out", default="docs/reference/clone_coverage.png")
    args = p.parse_args()

    def med(path):
        with np.load(path) as z:
            return (float(np.median(z["on_expert"])),
                    float(np.median(z["on_clone"])))

    uni = [
        ("expert-rollout\ndata only", *med(args.onpolicyonly)),
        ("hybrid\n(shipped clone)", *med(args.hybrid)),
        ("synthetic+degen.\nonly", *med(args.noonpolicy)),
    ]

    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.0),
                           gridspec_kw={"width_ratios": [1, 2.6]})

    def panel(a, rows, title, unit):
        x = np.arange(len(rows))
        w = 0.36
        a.bar(x - w / 2 - 0.01, [r[1] for r in rows], w, color=BLUE,
              label="at expert-visited states", zorder=3)
        a.bar(x + w / 2 + 0.01, [r[2] for r in rows], w, color=ORANGE,
              label="at clone-visited states", zorder=3)
        for i, (_n, e, c) in enumerate(rows):
            a.annotate(f"{c / e:.2f}x", (i, max(e, c)),
                       textcoords="offset points", xytext=(0, 5),
                       ha="center", fontsize=9, color=INK,
                       fontweight="bold")
        a.set_xticks(x, [r[0] for r in rows], fontsize=8)
        a.set_ylabel(f"median ||u_clone − u_expert|| ({unit})")
        a.set_title(title, loc="left", color=INK)
        a.grid(True, axis="y", zorder=0)
        a.set_axisbelow(True)
        a.set_ylim(0, max(max(r[1], r[2]) for r in rows) * 1.32)

    panel(ax[0], [REACHER], "A · Reacher (journey 13)", "torque units")
    panel(ax[1], uni, "B · Unicycle, dataset ablation", "action units")
    ax[1].legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("Same clone architecture, same expert protocol — the "
                 "off-policy error blow-up follows the DATA, not the system",
                 x=0.005, ha="left", color=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

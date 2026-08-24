"""Figures for the Reacher DeePC result: gate, reaching, and the failure split.

Numbers are transcribed from the runs recorded in journey 11 rather than
recomputed, because the closed-loop rows cost ~15 min of QP each and the point of
this script is to render them, not re-measure them. Every value here has its
producing command in the CSV alongside.

    uv run python scripts/plot_reacher_results.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, MUTED, CRITICAL, GOOD = "#0b0b0b", "#52514e", "#b8b7b2", "#d03b3b", "#0ca30c"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})

# --- measured data -----------------------------------------------------------
RADII = [0.0, 0.25, 0.5, 1.0, 2.0]
GATE = {                       # skill, per system, vs distance from the anchor
    "Reacher (30 anchors)": [0.94, 0.91, 0.84, -0.02, -6.06],
    "Panda (K=4)": [0.93, 0.88, 0.72, 0.14, -9.93],
}
GATE_COS = {
    "Reacher (30 anchors)": [0.97, 0.96, 0.93, 0.60, -0.41],
    "Panda (K=4)": [0.98, 0.96, 0.90, 0.85, -0.03],
}
# Reacher closed loop, 20 episodes, 50 steps, 10 mm tolerance.
LOOP = [
    #  label                     reached  achievable  path/net
    ("DeePC 30 anchors\n(margin 0.10)", 14, 16, 1.6),
    ("DeePC 108 anchors\n(margin 0.10)", 14, 16, 1.4),
    ("DeePC 30 anchors\n(margin 0.02)", 11, 20, 1.6),
    ("random torque", 1, 20, 8.2),
]
# Per-episode final distance under margin 0.02, all 20 goals reachable.
FINALS = [6.5, 8.8, 8.7, 12.3, 83.0, 9.5, 7.6, 10.0, 9.9, 26.2,
          97.6, 9.1, 7.0, 81.3, 9.2, 11.0, 21.2, 19.8, 9.3, 23.7]
TOL = 10.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="docs/reference/reacher_results.png")
    args = p.parse_args()

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.9))

    # A: the gate, both systems -- the headline cross-system agreement
    for i, (k, v) in enumerate(GATE.items()):
        ax[0].plot(RADII, v, color=SERIES[i], marker="o", ms=5, zorder=3, label=k)
    ax[0].axhline(0, color=CRITICAL, lw=1.2, ls="--", zorder=1)
    ax[0].text(2.0, 0.3, "useless below", ha="right", color=CRITICAL, fontsize=8)
    ax[0].axvspan(0, 0.5, color="#e8f3e8", zorder=0)
    ax[0].text(0.25, -8.5, "usable", ha="center", fontsize=8, color=GOOD)
    ax[0].set_xlabel("distance from anchor (rad)")
    ax[0].set_ylabel("prediction skill")
    ax[0].set_title("A · Same validity radius, both arms", loc="left", color=INK)
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")

    # B: direction quality -- what actually decides whether it steers
    for i, (k, v) in enumerate(GATE_COS.items()):
        ax[1].plot(RADII, v, color=SERIES[i], marker="o", ms=5, zorder=3, label=k)
    ax[1].axhline(0, color=CRITICAL, lw=1.2, ls="--", zorder=1)
    ax[1].text(2.0, 0.06, "steers backwards below", ha="right", color=CRITICAL,
               fontsize=8)
    ax[1].axvspan(0, 0.5, color="#e8f3e8", zorder=0)
    ax[1].set_ylim(-0.6, 1.1)
    ax[1].set_xlabel("distance from anchor (rad)")
    ax[1].set_ylabel("cos(predicted, true tip motion)")
    ax[1].set_title("B · Direction of the prediction", loc="left", color=INK)

    # C: closed loop, scored against ACHIEVABLE goals
    y = np.arange(len(LOOP))
    rates = [100 * r / a for _, r, a, _ in LOOP]
    cols = [SERIES[0], SERIES[0], SERIES[2], SERIES[1]]
    ax[2].barh(y, rates, 0.62, color=cols, zorder=3)
    for i, (_lab, r, a, eff) in enumerate(LOOP):
        ax[2].annotate(f"{r}/{a}   path/net {eff}", xy=(rates[i], i), xytext=(5, 0),
                       textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax[2].set_yticks(y)
    ax[2].set_yticklabels([lab for lab, _, _, _ in LOOP], fontsize=8)
    ax[2].invert_yaxis()
    ax[2].set_xlim(0, 118)
    ax[2].set_xlabel("reach rate among ACHIEVABLE goals (%)")
    ax[2].set_title("C · Closed loop, 20 episodes", loc="left", color=INK)
    ax[2].grid(True, axis="x", zorder=0)

    # D: the failure split -- why the binary metric understates it
    f = np.array(sorted(FINALS))
    colors = [GOOD if v < TOL else (SERIES[3] if v < 30 else CRITICAL) for v in f]
    ax[3].bar(np.arange(len(f)), f, 0.8, color=colors, zorder=3)
    ax[3].axhline(TOL, color=CRITICAL, lw=1.2, ls="--", zorder=4)
    ax[3].text(19, TOL * 1.15, "10 mm tolerance", ha="right", color=CRITICAL,
               fontsize=8)
    ax[3].set_yscale("log")
    ax[3].set_xlabel("episode (sorted)")
    ax[3].set_ylabel("final distance (mm)")
    n_near = int(((f >= TOL) & (f < 30)).sum())
    ax[3].set_title(f"D · Failures: {n_near} near-misses, "
                    f"{int((f >= 30).sum())} real", loc="left", color=INK)

    for a in ax:
        a.set_axisbelow(True)
        if a is not ax[2]:
            a.grid(True, axis="y", zorder=0)

    fig.suptitle("Reacher-v5 local-library DeePC — 2 DoF, pure torque, "
                 "30 anchors, 20 episodes", x=0.005, ha="left", color=INK2,
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")

    csv = args.out.replace(".png", ".csv")
    with open(csv, "w") as fh:
        fh.write("# scripts/run_reacher_deepc.py --grid 6 5 --episodes 20\n")
        fh.write("radius,reacher_skill,panda_skill,reacher_cos,panda_cos\n")
        for i, r in enumerate(RADII):
            fh.write(f"{r},{GATE['Reacher (30 anchors)'][i]},{GATE['Panda (K=4)'][i]},"
                     f"{GATE_COS['Reacher (30 anchors)'][i]},"
                     f"{GATE_COS['Panda (K=4)'][i]}\n")
        fh.write("\nconfig,reached,achievable,path_per_net\n")
        for lab, r, a, e in LOOP:
            fh.write(f"{lab.replace(chr(10), ' ')},{r},{a},{e}\n")
        fh.write("\nfinal_distance_mm_margin002\n")
        for v in FINALS:
            fh.write(f"{v}\n")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()

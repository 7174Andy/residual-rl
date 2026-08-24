"""The evidence that the Panda's problem is DATA COVERAGE, not the controller.

Four independent measurements, each from a different script, assembled into one
figure. Every value is transcribed from a run recorded in journey 11 or in this
session's logs; the producing command sits beside each in the CSV.

The argument in one line: the Panda's libraries are excellent within ~0.5 rad of
their data and anti-informative beyond ~2 rad, and the arm operates at ~2 rad,
because covering its configuration space at 0.5 rad costs ~10^5 trajectories.

  A  VALIDITY vs DISTANCE. Both arms have the same ~0.5 rad usable radius. What
     differs is where each one OPERATES -- Reacher inside it, the Panda far
     outside. This is the whole argument in one panel.
  B  WHAT COVERAGE COSTS. `r_K` decays as a clean power law with no elbow, so the
     anchor count needed to reach a given radius is an extrapolation, not a
     guess.
  C  CONTROL vs DISTANCE. The same Panda controller, varying only how far the
     start is from an anchor: 3/3 with optimal path efficiency at the anchor,
     collapsing to thrashing at 2 rad. Nothing about the controller changed.
  D  AT THE OPERATING DISTANCE, EVERYTHING TIES. Three Panda controllers -- one of
     them Select-DPC -- are indistinguishable from a random walk. The identical
     `core/selectdpc.py` gets 16/20 on Reacher.

    uv run python scripts/plot_panda_coverage_argument.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, CRITICAL, GOOD = "#0b0b0b", "#52514e", "#b8b7b2", "#d03b3b", "#0ca30c"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})

# --- A: scripts/verify_libraries.py / scripts/run_reacher_deepc.py -----------
RAD_P = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
SKILL_PANDA = [0.93, 0.88, 0.72, 0.14, -9.93, -23.64]
RAD_R = [0.0, 0.25, 0.5, 1.0, 2.0]
SKILL_REACH = [0.94, 0.91, 0.84, -0.02, -6.06]
OP_PANDA = 1.98      # median nearest-data distance, 65-trajectory uniform bank
OP_REACH = 0.52      # anchor spacing on the 12x9 grid

# --- B: scripts/anchor_coverage.py, farthest-point on IK configurations ------
FPS_K = np.array([1, 2, 4, 8, 12, 16, 20, 37, 65])
FPS_R = np.array([3.21, 2.99, 2.60, 1.99, 1.73, 1.48, 1.35, 1.00, 0.72])
FIT = (-0.319, 1.333)          # log r = a log K + b, R^2 = 0.94, d = -1/a = 3.13

# --- C: scripts/test_valid_region_control.py --------------------------------
START_D = [0.0, 0.5, 2.0]
REACHED = [3, 2, 1]
EFF_C = [1.0, 1.3, 53.8]

# --- D: scripts/run_select_dpc.py (Panda) and the Reacher runs --------------
BARS = [
    ("Panda\nSelect-DPC", 0, 10, 7.9, SERIES[2]),
    ("Panda\nK=65 fixed", 0, 10, 13.9, SERIES[1]),
    ("Panda\nrandom", 0, 10, 7.5, SERIES[3]),
    ("Reacher\nSelect-DPC", 16, 20, 1.4, SERIES[0]),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="docs/reference/panda_coverage_argument.png")
    args = p.parse_args()
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.1))

    # A -----------------------------------------------------------------
    ax[0].axhline(0, color=CRITICAL, lw=1.2, ls="--", zorder=1)
    ax[0].axvspan(0, 0.5, color="#e8f3e8", zorder=0)
    ax[0].plot(RAD_P, SKILL_PANDA, color=SERIES[1], marker="o", ms=5, zorder=3,
               label="Panda (7-DoF)")
    ax[0].plot(RAD_R, SKILL_REACH, color=SERIES[0], marker="o", ms=5, zorder=3,
               label="Reacher (2-DoF)")
    ax[0].axvline(OP_PANDA, color=SERIES[1], lw=1.6, ls=":", zorder=2)
    ax[0].axvline(OP_REACH, color=SERIES[0], lw=1.6, ls=":", zorder=2)
    ax[0].annotate("Panda operates HERE\n(nearest data 1.98 rad)",
                   xy=(OP_PANDA, -14), xytext=(2.15, -14), fontsize=8,
                   color=SERIES[1])
    ax[0].annotate("Reacher operates here", xy=(OP_REACH, -5.5), xytext=(0.72, -5.0),
                   fontsize=8, color=SERIES[0],
                   arrowprops=dict(arrowstyle="->", color=SERIES[0], lw=1))
    ax[0].text(0.25, 2.5, "usable", ha="center", fontsize=8, color=GOOD)
    ax[0].set_xlabel("distance from the nearest data (rad)")
    ax[0].set_ylabel("prediction skill")
    ax[0].set_title("A · Same validity radius,\ndifferent operating point",
                    loc="left", color=INK)
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")

    # B -----------------------------------------------------------------
    k_ext = np.logspace(0, 5.5, 200)
    ax[1].plot(k_ext, np.exp(FIT[1]) * k_ext ** FIT[0], color=MUTED, lw=1.2,
               ls=":", zorder=1, label="power-law fit ($d$=3.13)")
    ax[1].plot(FPS_K, FPS_R, color=SERIES[1], marker="o", ms=5, zorder=3,
               label="measured (farthest-point)")
    ax[1].axhline(0.5, color=CRITICAL, lw=1.2, ls="--", zorder=2)
    ax[1].text(1.2, 0.55, "usable radius", color=CRITICAL, fontsize=8)
    for k, lab in ((571, "~570\n(IK manifold,\n$d$=3.1)"),
                   (165000, "~165,000\n(uniform starts,\n$d$=5.7)")):
        ax[1].plot([k], [0.5], marker="o", ms=10, mfc="none", mec=CRITICAL, mew=2,
                   zorder=4)
        ax[1].annotate(lab, xy=(k, 0.5), xytext=(0, -46), textcoords="offset points",
                       ha="center", fontsize=8, color=CRITICAL)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("number of anchors / trajectories $K$")
    ax[1].set_ylabel("worst-case distance $r_K$ (rad)")
    ax[1].set_title("B · What coverage costs\n(no elbow: it is a power law)",
                    loc="left", color=INK)
    ax[1].legend(frameon=False, fontsize=8, loc="upper right")

    # C -----------------------------------------------------------------
    x = np.arange(len(START_D))
    ax[2].bar(x, [100 * r / 3 for r in REACHED], 0.55, color=SERIES[1], zorder=3)
    for i, (r, e) in enumerate(zip(REACHED, EFF_C)):
        ax[2].annotate(f"{r}/3\npath/net {e}", xy=(i, 100 * r / 3), xytext=(0, 5),
                       textcoords="offset points", ha="center", fontsize=8,
                       color=INK2)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([f"{d}" for d in START_D])
    ax[2].set_ylim(0, 128)
    ax[2].set_xlabel("start distance from an anchor (rad)")
    ax[2].set_ylabel("reach rate (%)")
    ax[2].set_title("C · Same controller,\nonly the distance changes",
                    loc="left", color=INK)

    # D -----------------------------------------------------------------
    y = np.arange(len(BARS))
    ax[3].barh(y, [100 * k / n for _, k, n, _, _ in BARS], 0.6,
               color=[c for *_, c in BARS], zorder=3)
    for i, (_, k, n, e, c) in enumerate(BARS):
        if k == 0:
            ax[3].plot([0], [i], marker="|", ms=16, mew=3, color=c, zorder=4)
        ax[3].annotate(f"{k}/{n}   path/net {e}", xy=(100 * k / n, i), xytext=(9, 0),
                       textcoords="offset points", va="center", fontsize=8,
                       color=CRITICAL if k == 0 else INK2)
    ax[3].set_yticks(y)
    ax[3].set_yticklabels([b[0] for b in BARS], fontsize=8)
    ax[3].invert_yaxis()
    ax[3].set_xlim(0, 135)
    ax[3].set_xlabel("reach rate (%)")
    ax[3].set_title("D · Identical code.\nOnly the data differs.",
                    loc="left", color=INK)

    for a in ax:
        a.grid(True, zorder=0)
        a.set_axisbelow(True)
    fig.suptitle("The 7-DoF Panda's problem is data coverage, not the controller "
                 "— four independent measurements",
                 x=0.005, ha="left", color=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")

    csv = args.out.replace(".png", ".csv")
    with open(csv, "w") as fh:
        fh.write("# A: scripts/verify_libraries.py ; scripts/run_reacher_deepc.py\n")
        fh.write("radius_rad,panda_skill,reacher_skill\n")
        for i, r in enumerate(RAD_P):
            rs = SKILL_REACH[i] if i < len(SKILL_REACH) else ""
            fh.write(f"{r},{SKILL_PANDA[i]},{rs}\n")
        fh.write("\n# B: scripts/anchor_coverage.py --k-max 65\nK,r_K_rad\n")
        for k, r in zip(FPS_K, FPS_R):
            fh.write(f"{k},{r}\n")
        fh.write("\n# C: scripts/test_valid_region_control.py\n")
        fh.write("start_dist_rad,reached_of_3,path_per_net\n")
        for dd, r, e in zip(START_D, REACHED, EFF_C):
            fh.write(f"{dd},{r},{e}\n")
        fh.write("\n# D: scripts/run_select_dpc.py ; scripts/run_select_dpc_reacher.py\n")
        fh.write("controller,reached,n,path_per_net\n")
        for lab, k, n, e, _ in BARS:
            fh.write(f"{lab.replace(chr(10), ' ')},{k},{n},{e}\n")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()

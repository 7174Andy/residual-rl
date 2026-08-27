"""Journey 15's figures: the 20k task bank, and what it bought.

Three figures from the one Phase-1 lever that worked — goal-directed collection
feeding Select-DPC instead of random excitation feeding fixed anchors:

  panda_taskbank_result.png   reach across every arm, per-scenario detail, and
                              the paired Select-DPC vs IK-oracle comparison
  panda_taskbank_coverage.png the coverage power law that predicted the 20k
                              bank's nearest-sample distance, and the price per step
  panda_taskbank_horizon.png  the 8 misses split into budget vs controller

Per-scenario rows come from `data/ck_taskbank20k_sdpc.jsonl`, the eval's own
resume checkpoint — it carries all THREE arms (Select-DPC, a DLS-IK oracle and
a random-walk control) over the identical frozen 78, which the results CSV does
not. Gate and horizon numbers are transcribed from `data/expert_phase1_log.md`;
they were measured once, at ~13 wall-hours per reach eval, and are not
re-derivable from anything cheaper in this tree.

    uv run python scripts/plot_panda_taskbank.py
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rl.stats import mcnemar_pvalue, wilson_ci  # noqa: E402

BLUE, ORANGE, INK, INK2, MUTED = ("#2a78d6", "#eb6834", "#0b0b0b", "#52514e",
                                  "#b8b7b2")
GREEN, RED = "#2e8b57", "#c1443c"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6,
})

TOL_MM = 50.0        # goal_tolerance = 0.05 m, the frozen protocol
VALID_RAD = 0.5      # journey 11's measured library validity radius

# Arms scored on the frozen 78. Sources: this run's checkpoint (select, oracle,
# random) and data/panda_results.csv row counts (the deepc_* arms), with the
# 300-step horizon supplement from data/expert_phase1_log.md.
ARMS = [
    ("random walk", 0, 78, MUTED),
    ("DeePC v2\n(file-default λ)", 11, 78, MUTED),
    ("DLS-IK oracle\n(rate-limited)", 42, 78, INK2),
    ("DeePC v2 re-pin\n(swept λ)", 46, 78, INK2),
    ("DeePC v2\n(published)", 47, 78, INK2),
    ("Select-DPC\n20k task bank", 70, 78, BLUE),
    ("…same, 300-step\nhorizon", 76, 78, ORANGE),
]

# The 8 misses, re-run at STEPS=300 with the identical frozen controller
# (data/expert_phase1_log.md, "horizon experiment"). best_mm is at the frozen
# 150-step budget; reach_step is None where it never reached.
HORIZON = [
    # scenario, best@150 mm, reach step @300, tail slope mm/step
    (53, 66.0, 159, -1.7),
    (13, 173.0, 169, -7.2),
    (46, 478.0, 182, -17.6),
    (19, 198.0, 183, -2.0),
    (59, 95.0, 194, -4.3),
    (0, 408.0, 276, -13.6),
    (16, 201.0, None, -0.40),
    (17, 113.0, None, +0.35),
]

# The three banks tried, by ANCHOR count (= independent placements, which is the
# quantity the coverage law is in — not raw sample count, and not trajectory
# length: the old bank is 65 anchors x 1500 steps, the new ones 150 steps each).
#
# Distances are the median over the 78 frozen episode starts of the distance to
# the nearest COLLECTED SAMPLE, all three measured on one protocol by
# scripts/plot_bank_informativeness.py, so they are directly comparable and
# reproducible. Do NOT mix in journey 11's 1.98 rad here: that is a nearest-
# ANCHOR distance, a different quantity that happens to have the same units.
# `skill` is the open-loop prediction skill of the SELECTED columns against a
# hold-last baseline (>0 = informative), from data/expert_phase1_log.md.
BANKS = [
    # label, n_anchors, nearest-sample median rad, skill, reach
    ("random OU", 65, 1.43, -1.83, "0/10"),
    ("task servo 1k", 1000, 0.87, -0.29, "not run"),
    ("task servo 20k", 20000, 0.54, +0.40, "70/78"),
]


def load_arms(path: str) -> dict:
    """Per-scenario results keyed by arm, from the eval's resume checkpoint."""
    out: dict[str, dict[int, dict]] = {}
    for line in open(path):
        r = json.loads(line)
        out.setdefault(r["row"], {})[int(r["i"])] = r["result"]
    return {k: [v[i] for i in sorted(v)] for k, v in out.items()}


def fig_result(arms: dict, out: str) -> None:
    sel, orc, rnd = arms["select"], arms["oracle"], arms["random"]
    fig = plt.figure(figsize=(12.4, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.42,
                          wspace=0.30)

    # --- A: reach across arms, with Wilson intervals ---------------------------
    a = fig.add_subplot(gs[0, :])
    x = np.arange(len(ARMS))
    vals = [100.0 * k / n for _l, k, n, _c in ARMS]
    los, his = zip(*[wilson_ci(k, n) for _l, k, n, _c in ARMS])
    err = np.vstack([np.array(vals) - 100 * np.array(los),
                     100 * np.array(his) - np.array(vals)])
    a.bar(x, vals, 0.62, color=[c for *_r, c in ARMS], zorder=3)
    a.errorbar(x, vals, yerr=err, fmt="none", ecolor=INK2, elinewidth=1.1,
               capsize=3, zorder=4)
    for i, (_l, k, n, _c) in enumerate(ARMS):
        a.annotate(f"{k}/{n}", (i, 100 * his[i]), textcoords="offset points",
                   xytext=(0, 5), ha="center", fontsize=9, fontweight="bold",
                   color=INK)
    a.set_xticks(x, [lab for lab, *_r in ARMS], fontsize=8)
    a.set_ylabel("reach rate (%)")
    a.set_ylim(0, 108)
    a.grid(True, axis="y", zorder=0)
    a.set_axisbelow(True)
    a.set_title("A · Every arm on the frozen 78 scenarios "
                "(goal_tolerance 50 mm, 150 steps unless noted); "
                "bars are 95% Wilson intervals", loc="left", color=INK)

    # --- B: sorted per-scenario final distance ---------------------------------
    b = fig.add_subplot(gs[1, 0])
    for rows, lab, col in ((rnd, "random walk", MUTED),
                           (orc, "DLS-IK oracle", INK2),
                           (sel, "Select-DPC 20k", BLUE)):
        d = np.sort([1000.0 * r["final"] for r in rows])
        b.plot(np.arange(len(d)), d, lw=1.8, color=col, label=lab)
    b.axhline(TOL_MM, color=RED, lw=1.2, ls="--")
    b.annotate("50 mm tolerance", (1, TOL_MM), textcoords="offset points",
               xytext=(2, 4), fontsize=8, color=RED)
    b.set_yscale("log")
    b.set_xlabel("scenario, sorted by final distance")
    b.set_ylabel("closest approach (mm, log)")
    b.legend(frameon=False, fontsize=8, loc="upper left")
    b.grid(True, zorder=0)
    b.set_axisbelow(True)
    b.set_title("B · Where each arm ends up", loc="left", color=INK)

    # --- C: paired outcomes vs the oracle --------------------------------------
    #
    # A "who got closer" scatter is meaningless here: an episode stops at first
    # reach, so every success parks just under the 50 mm line and the two arms
    # tie on ~40 scenarios by construction. The paired OUTCOME table is the
    # comparison that carries information, and it is what McNemar consumes.
    c = fig.add_subplot(gs[1, 1])
    s_ok = np.array([bool(r["reached"]) for r in sel])
    o_ok = np.array([bool(r["reached"]) for r in orc])
    cells = [("both\nreached", int((s_ok & o_ok).sum()), INK2),
             ("Select-DPC\nonly", int((s_ok & ~o_ok).sum()), BLUE),
             ("oracle\nonly", int((~s_ok & o_ok).sum()), ORANGE),
             ("neither", int((~s_ok & ~o_ok).sum()), MUTED)]
    cx = np.arange(len(cells))
    c.bar(cx, [v for _l, v, _c in cells], 0.62,
          color=[col for *_r, col in cells], zorder=3)
    for i, (_l, v, _c) in enumerate(cells):
        c.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 4),
                   ha="center", fontsize=9, fontweight="bold", color=INK)
    b_disc = int((s_ok & ~o_ok).sum())
    c_disc = int((~s_ok & o_ok).sum())
    pval = mcnemar_pvalue(b_disc, c_disc)
    c.set_xticks(cx, [lab for lab, *_r in cells], fontsize=8)
    c.set_ylabel("scenarios")
    c.set_ylim(0, max(v for _l, v, _c in cells) * 1.25)
    c.grid(True, axis="y", zorder=0)
    c.set_axisbelow(True)
    c.set_title(f"C · Paired vs the IK oracle\nMcNemar {b_disc}–{c_disc}, "
                f"p = {pval:.2g}", loc="left", color=INK)

    # --- D: steps to reach ------------------------------------------------------
    d = fig.add_subplot(gs[1, 2])
    steps = [r["steps"] for r in sel if r["reached"]]
    d.hist(steps, bins=np.arange(0, 160, 10), color=BLUE, zorder=3)
    d.axvline(150, color=RED, lw=1.2, ls="--")
    d.annotate("150-step budget", (150, 0), textcoords="offset points",
               xytext=(4, 6), fontsize=8, color=RED, rotation=90, ha="left")
    d.set_xlabel("steps to reach")
    d.set_ylabel("scenarios")
    d.grid(True, axis="y", zorder=0)
    d.set_axisbelow(True)
    d.set_title(f"D · {len(steps)} successes,\nmedian {int(np.median(steps))} steps",
                loc="left", color=INK)

    fig.suptitle("Select-DPC on a 20,000-trajectory goal-directed bank: "
                 "70/78, above every fixed-anchor arm and above the IK oracle "
                 "under the same rate limit",
                 x=0.005, ha="left", color=INK2, fontsize=10.5)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.09)
    fig.savefig(out, dpi=160)
    print("wrote", out)


def fig_coverage(arms: dict, out: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.9))

    # --- A: the coverage power law ---------------------------------------------
    a = ax[0]
    K = np.array([b[1] for b in BANKS], dtype=float)
    dist = np.array([b[2] for b in BANKS])
    grid = np.logspace(np.log10(40), np.log10(60000), 100)
    # d ~ K^(-1/d_eff), d_eff = 5.7 measured three independent ways
    # (docs/reference/panda_coverage_argument.png). Anchored on the 1k point.
    a.plot(grid, 0.87 * (grid / 1000.0) ** (-1.0 / 5.7), color=MUTED, lw=1.4,
           ls="--", label="$K^{-1/5.7}$ coverage law")
    a.scatter(K, dist, s=70, color=[MUTED, INK2, BLUE], zorder=4)
    for (_lab, k, dd, *_r) in BANKS:
        a.annotate(f"{dd:.2f} rad", (k, dd), textcoords="offset points",
                   xytext=(7, 5), fontsize=8.5, fontweight="bold", color=INK)
    a.annotate("65 anchors\n× 1500 steps", (K[0], dist[0]),
               textcoords="offset points", xytext=(9, -22), fontsize=7.5,
               color=INK2)
    a.annotate("20,000 anchors\n× 150 steps", (K[2], dist[2]),
               textcoords="offset points", xytext=(-6, 16), ha="right",
               fontsize=7.5, color=INK2)
    a.axhline(VALID_RAD, color=RED, lw=1.2, ls="--")
    a.annotate("0.5 rad validity radius", (45, VALID_RAD),
               textcoords="offset points", xytext=(0, 5), fontsize=8, color=RED)
    a.set_xscale("log")
    a.set_xlabel("anchors collected (independent placements)")
    a.set_ylabel("median distance to nearest\ncollected sample (rad)")
    a.legend(frameon=False, fontsize=8, loc="lower left")
    a.grid(True, zorder=0)
    a.set_axisbelow(True)
    a.set_title("A · Coverage tracks PLACEMENTS, not samples —\n"
                "and 20k lands on the validity radius",
                loc="left", color=INK)

    # --- B: what the win costs --------------------------------------------------
    c = ax[1]
    ms = np.array([r["mean_step_ms"] for r in arms["select"]
                   if np.isfinite(r["mean_step_ms"])])
    c.hist(ms, bins=18, color=BLUE, zorder=3)
    c.axvline(np.median(ms), color=RED, lw=1.3, ls="--")
    c.annotate(f"median {np.median(ms):.0f} ms per step\n"
               f"= {np.median(ms) / 20.0:.0f}× the 20 ms\ncontrol period",
               (0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
               fontsize=8, color=RED, fontweight="bold")
    c.set_xlabel("mean solve time per control step (ms)")
    c.set_ylabel("scenarios")
    c.grid(True, axis="y", zorder=0)
    c.set_axisbelow(True)
    c.set_title("B · The price of the win\n→ an offline labeller, not a controller",
                loc="left", color=INK)

    fig.suptitle("Coverage of a 5.7-dimensional configuration set is bought by "
                 "independent placements, and it is not cheap to run",
                 x=0.005, ha="left", color=INK2, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=160)
    print("wrote", out)


def fig_horizon(out: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.1),
                           gridspec_kw={"width_ratios": [1.35, 1]})

    rows = sorted(HORIZON, key=lambda r: (r[2] is None, r[2] or 0))
    y = np.arange(len(rows))

    # --- A: when each miss actually reached ------------------------------------
    a = ax[0]
    for i, (_sid, _best, step, slope) in enumerate(rows):
        stuck = step is None
        a.plot([150, step or 300], [i, i], lw=2.0,
               color=RED if stuck else GREEN, alpha=0.85, zorder=3)
        a.scatter([150], [i], s=30, color=INK2, zorder=4)
        if stuck:
            a.scatter([300], [i], s=60, marker="x", color=RED, zorder=4)
            a.annotate(f"still missing ({slope:+.2f} mm/step)", (300, i),
                       textcoords="offset points", xytext=(-8, 6), ha="right",
                       fontsize=8, color=RED)
        else:
            a.scatter([step], [i], s=44, color=GREEN, zorder=4)
            a.annotate(f"reached @{step}", (step, i), textcoords="offset points",
                       xytext=(7, -3), fontsize=8, color=INK2)
    a.axvline(150, color=INK2, lw=1.2, ls="--")
    a.annotate("frozen 150-step budget", (150, len(rows) - 0.4),
               textcoords="offset points", xytext=(-6, 0), ha="right",
               fontsize=8.5, color=INK2)
    a.set_yticks(y, [f"scenario {r[0]}" for r in rows], fontsize=8.5)
    a.set_ylim(-0.7, len(rows) - 0.05)   # headroom so the title clears row 0
    a.set_xlabel("control step")
    a.set_xlim(120, 330)
    a.grid(True, axis="x", zorder=0)
    a.set_axisbelow(True)
    a.set_title("A · 6 of the 8 misses were budget failures, not controller "
                "failures", loc="left", color=INK)

    # --- B: tail slope separates the two kinds ---------------------------------
    b = ax[1]
    slopes = [r[3] for r in rows]
    cols = [RED if r[2] is None else GREEN for r in rows]
    b.barh(y, slopes, 0.6, color=cols, zorder=3)
    b.axvline(0, color=INK2, lw=1.0)
    b.set_yticks(y, [str(r[0]) for r in rows], fontsize=8.5)
    b.set_xlabel("tail slope of distance-to-goal (mm/step)")
    b.set_ylabel("scenario")
    b.grid(True, axis="x", zorder=0)
    b.set_axisbelow(True)
    b.set_title("B · Still closing (green) vs\nstalled or drifting out (red)",
                loc="left", color=INK)

    fig.suptitle("Extended-horizon reach is 76/78 (97.4%); the official number "
                 "stays 70/78 because 150 steps is part of the frozen protocol",
                 x=0.005, ha="left", color=INK2, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, dpi=160)
    print("wrote", out)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="data/ck_taskbank20k_sdpc.jsonl")
    p.add_argument("--outdir", default="docs/reference")
    args = p.parse_args()

    arms = load_arms(args.checkpoint)
    os.makedirs(args.outdir, exist_ok=True)
    fig_result(arms, os.path.join(args.outdir, "panda_taskbank_result.png"))
    fig_coverage(arms, os.path.join(args.outdir, "panda_taskbank_coverage.png"))
    fig_horizon(os.path.join(args.outdir, "panda_taskbank_horizon.png"))


if __name__ == "__main__":
    main()

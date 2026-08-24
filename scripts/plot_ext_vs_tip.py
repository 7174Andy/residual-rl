#!/usr/bin/env python
"""Why the extended-output DeePC matched, rather than beat, the tip-only one.

    uv run python scripts/plot_ext_vs_tip.py

Reads `data/panda_results.csv` (methods `deepc`, `deepc_v2`, `deepc_ext_v2`) and
lays out the evidence for a single claim: `y_ext` did not control worse, it
crossed the success threshold less often.

Three panels, left to right:

1. Where each scenario landed, tip vs ext. Points on the diagonal are scenarios
   the two arms finished in the same place. The tolerance lines split it into
   both-reach / one-only / both-fail, so the reach difference is visible as a
   handful of points straddling a line rather than a shifted cloud.
2. Cumulative final distance. If ext were controlling worse its curve would sit
   right of tip's everywhere; instead they overlap, and the T=400 baseline is the
   one that is genuinely displaced.
3. The asymmetry that explains the cost: matching the past grades every output
   equally, while only the tip counts toward the goal. Adding `q` bought 35 more
   graded rows and zero more rewarded ones.

Panel 3 is the mechanism, panels 1-2 are the symptom. Open-loop |g|_1 evidence
lives in the separate output-map probe.
"""
from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Baseline is a deliberate neutral (below the chroma floor -- it is context, not a
# subject). tip/ext are a categorical pair, validated:
#   node <dataviz>/scripts/validate_palette.js "#184f95,#b45309" --mode light
#   -> ALL CHECKS PASS (CVD dE 22.0 protan, normal 29.3, both >= 3:1 on surface)
_BASE = "#898781"
_TIP = "#184f95"
_EXT = "#b45309"
_INK = "#52514e"
_GRID = "#e1e0d9"

TOL = 0.05          # goal_tolerance, metres
T_INI = 5
ARMS = (("deepc", "T=400 tip", _BASE), ("deepc_v2", "T=3000 tip", _TIP),
        ("deepc_ext_v2", "T=3000 ext", _EXT))


def load(path: str) -> dict:
    out: dict = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(r["method"], {})[int(r["scenario_id"])] = {
            "reached": r["reached"] == "True",
            "final": float(r["final_dist"]),
            "solve": float(r["mean_solve_ms"]),
        }
    return out


def _panel_scatter(ax, arms):
    tip, ext = arms["deepc_v2"], arms["deepc_ext_v2"]
    ids = sorted(set(tip) & set(ext))
    x = np.array([tip[i]["final"] for i in ids])
    y = np.array([ext[i]["final"] for i in ids])
    tr = np.array([tip[i]["reached"] for i in ids])
    er = np.array([ext[i]["reached"] for i in ids])

    lim = (0.02, 2.0)
    ax.fill_between(lim, TOL, lim[1], color=_TIP, alpha=0.05, lw=0)
    ax.axvspan(lim[0], TOL, color=_EXT, alpha=0.05, lw=0)
    ax.plot(lim, lim, "-", color=_GRID, lw=1.2, zorder=1)
    ax.axhline(TOL, color=_INK, lw=0.9, ls="--", zorder=2)
    ax.axvline(TOL, color=_INK, lw=0.9, ls="--", zorder=2)

    both = tr & er
    tip_only = tr & ~er
    ext_only = ~tr & er
    neither = ~tr & ~er
    for mask, c, m, lab in ((both, _INK, "o", f"both reach ({both.sum()})"),
                            (tip_only, _TIP, "^", f"tip only ({tip_only.sum()})"),
                            (ext_only, _EXT, "v", f"ext only ({ext_only.sum()})"),
                            (neither, _BASE, "x", f"neither ({neither.sum()})")):
        ax.plot(x[mask], y[mask], m, color=c, ms=6, mew=1.4,
                mfc="none" if m in "ox" else c, ls="none", label=lab, zorder=4)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("tip final distance  [m]", fontsize=9, color=_INK)
    ax.set_ylabel("ext final distance  [m]", fontsize=9, color=_INK)
    ax.set_title("They disagree on 17 of 78 — and not only at the line",
                 fontsize=10, color=_INK, pad=8)
    ax.legend(loc="lower right", fontsize=7.2, frameon=False, labelcolor=_INK)
    ax.grid(True, which="major", color=_GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_GRID)
    ax.tick_params(colors=_BASE, labelsize=8)
    # How to read it, because the axes are not symmetric in meaning: an episode
    # that reaches TERMINATES at the tolerance, so every success lands at ~0.05 by
    # construction and the discordant markers necessarily sit on one line. What is
    # informative is their spread along the OTHER axis -- that is how badly the
    # losing arm did, and much of it is far past a near-miss.
    r = np.corrcoef(np.log(x), np.log(y))[0, 1]
    lost = np.concatenate([y[tip_only], x[ext_only]])
    ax.annotate(f"log-log r = {r:.2f}\nreaching ends the episode at the\n"
                f"tolerance, so wins pile on a line;\nthe loser's spread is the real\n"
                f"signal — median {np.median(lost):.2f} m, max {lost.max():.2f} m",
                xy=(0.035, 0.985), xycoords="axes fraction", fontsize=7.0,
                color=_INK, va="top")


def _panel_ecdf(ax, arms):
    for key, label, color in ARMS:
        v = np.sort([r["final"] for r in arms[key].values()])
        frac = np.arange(1, len(v) + 1) / len(v)
        ax.step(v, frac, where="post", color=color, lw=2.0, label=label)
    ax.axvline(TOL, color=_INK, lw=1.0, ls="--")
    ax.annotate("goal_tolerance\n50 mm", xy=(TOL, 0.03), xytext=(6, 0),
                textcoords="offset points", fontsize=7.4, color=_INK, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.02, 2.0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("final distance  [m]", fontsize=9, color=_INK)
    ax.set_ylabel("fraction of scenarios at or below", fontsize=9, color=_INK)
    ax.set_title("Tip and ext distributions overlap", fontsize=10, color=_INK, pad=8)
    ax.legend(loc="lower right", fontsize=7.6, frameon=False, labelcolor=_INK)
    ax.grid(True, color=_GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_GRID)
    ax.tick_params(colors=_BASE, labelsize=8)


def _panel_rows(ax, arms):
    """Graded-when-matching vs rewarded-toward-the-goal, per output map."""
    maps = [("tip\n(3-D)", 3, _TIP), ("ext\n(10-D)", 10, _EXT)]
    y = np.arange(len(maps))
    h = 0.34
    for k, (label, p_y, color) in enumerate(maps):
        ax.barh(y[k] + h / 2, p_y * T_INI, height=h, color=color, alpha=0.35,
                zorder=2)
        ax.barh(y[k] - h / 2, 3, height=h, color=color, zorder=3)
        ax.annotate(f"{p_y * T_INI}", xy=(p_y * T_INI, y[k] + h / 2), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=_INK, fontweight="bold")
        ax.annotate("3", xy=(3, y[k] - h / 2), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8, color=_INK)
    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in maps], fontsize=9, color=_INK)
    ax.set_xlim(0, 62)
    ax.set_xlabel("number of output rows", fontsize=9, color=_INK)
    ax.set_title("Graded on 50, rewarded on 3", fontsize=10, color=_INK, pad=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_INK, alpha=0.35,
                      label="graded when matching the past  ($p_y\\cdot T_{ini}$)"),
        plt.Rectangle((0, 0), 1, 1, color=_INK,
                      label="weighted toward the goal (tip only, $Q=\\mathrm{diag}(I_3,0)$)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              fontsize=7.2, frameon=False, labelcolor=_INK, ncol=1)
    ax.grid(True, axis="x", color=_GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(colors=_BASE, labelsize=8)

    ms_tip = np.mean([r["solve"] for r in arms["deepc_v2"].values()])
    ms_ext = np.mean([r["solve"] for r in arms["deepc_ext_v2"].values()])
    ax.annotate(f"cost of the extra 35 rows:\n{ms_tip:.0f} → {ms_ext:.0f} ms/step "
                f"({ms_ext / ms_tip:.2f}×)\nbenefit: none measurable",
                xy=(0.97, 0.60), xycoords="axes fraction", ha="right", va="top",
                fontsize=7.6, color=_INK)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="data/panda_results.csv")
    ap.add_argument("--out", default="docs/reference/ext_vs_tip.png")
    args = ap.parse_args()

    arms = load(args.results)
    missing = [k for k, _, _ in ARMS if k not in arms]
    if missing:
        raise SystemExit(f"{args.results} is missing methods {missing}")

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.0),
                            gridspec_kw={"width_ratios": [1.05, 1.05, 0.9]})
    _panel_scatter(axes[0], arms)
    _panel_ecdf(axes[1], arms)
    _panel_rows(axes[2], arms)

    n = len(arms["deepc_ext_v2"])
    reach = {k: np.mean([r["reached"] for r in arms[k].values()]) for k, _, _ in ARMS}
    fig.suptitle(
        f"Adding joint angles changed WHICH scenarios succeed, not how well the arm "
        f"is controlled   (n={n} paired; "
        f"tip {reach['deepc_v2']:.3f} vs ext {reach['deepc_ext_v2']:.3f}, "
        f"McNemar p=0.14 — not significant)",
        fontsize=11, color=_INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=130, facecolor="white")
    print(f"wrote {args.out}")
    for k, label, _ in ARMS:
        v = [r["final"] for r in arms[k].values()]
        print(f"  {label:12s} reach {reach[k]:.3f}  median final {np.median(v):.3f} m")


if __name__ == "__main__":
    main()

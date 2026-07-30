#!/usr/bin/env python
"""Plot reach-rate comparison: DeePC vs clone vs clone+TD3 residual (200k, 400k).

Default input is the committed benchmark-results CSV
(`docs/journey/figures/reach_rates.csv`, columns ``label,k,n``) transcribed from
the 78-seed canonical sweep in docs/journey/08-residual-rl.md, so the figure
regenerates without re-running the (QP-bound, ~minutes-per-seed) benchmark:

    uv run python scripts/plot_reach_rates.py

Wilson 95% CIs are computed directly from (k, n) via the same
`two_wheel_robot.rl.clone_eval.wilson_ci` that `eval_residual.py` uses, so the
error bars match the CIs reported in the doc exactly.
"""
from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from two_wheel_robot.rl.clone_eval import wilson_ci  # noqa: E402

# dataviz-skill palette: baseline bars read as a neutral (below the chroma floor
# on purpose -- they're not the focus), residual bars are an ordinal one-hue ramp
# keyed to training budget (validated: node scripts/validate_palette.js
# "#3987e5,#184f95" --mode light --ordinal -> ALL CHECKS PASS).
_GRAY = "#898781"
_BLUE_200K = "#3987e5"
_BLUE_400K = "#184f95"
_INK = "#52514e"
_GRID = "#e1e0d9"


def _read_rows(path: str) -> list[dict]:
    with open(path) as f:
        return [{"label": r["label"], "k": int(r["k"]), "n": int(r["n"])}
                for r in csv.DictReader(f)]


def _color_for(label: str) -> str:
    if "400k" in label:
        return _BLUE_400K
    if "200k" in label:
        return _BLUE_200K
    return _GRAY


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="docs/journey/figures/reach_rates.csv")
    p.add_argument("--out", default="docs/journey/figures/reach_rates.png")
    args = p.parse_args()

    rows = _read_rows(args.input)
    labels = [r["label"] for r in rows]
    rates = [r["k"] / r["n"] for r in rows]
    cis = [wilson_ci(r["k"], r["n"]) for r in rows]
    lo_err = [rate - ci[0] for rate, ci in zip(rates, cis)]
    hi_err = [ci[1] - rate for rate, ci in zip(rates, cis)]
    colors = [_color_for(lbl) for lbl in labels]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = range(len(labels))
    ax.bar(x, rates, color=colors, width=0.6, zorder=3)
    ax.errorbar(x, rates, yerr=[lo_err, hi_err], fmt="none",
                ecolor=_INK, elinewidth=1.4, capsize=4, zorder=4)

    for xi, rate, row in zip(x, rates, rows):
        ax.annotate(f"{row['k']}/{row['n']} = {rate:.1%}", (xi, rate),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="#0b0b0b")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("reach rate (78-seed sweep)")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title("Reach rate — DeePC vs clone vs clone + TD3 residual")
    ax.grid(axis="y", color=_GRID, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  ({len(rows)} bars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

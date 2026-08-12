#!/usr/bin/env python
"""Plot training return vs TIMESTEPS for vanilla TD3 vs the clone+residual, 5 seeds each.

The x axis is environment steps, not episodes: sample efficiency is a claim about
steps consumed, and the two arms have different episode lengths, so an episode axis
would silently rescale one against the other.

Reads the SB3 Monitor CSVs the seed sweep wrote:

    uv run python scripts/plot_learning_curves.py --glob-van 'data/seedsweep/van_s*_mon.monitor.csv'

Left panel is the full budget, right panel zooms the first 100k steps where the two
curves actually differ. Bands are +/-1 std of the per-seed rolling-mean curves,
resampled onto a shared step grid (runs end at slightly different episode counts).
"""
from __future__ import annotations

import argparse
import csv
import glob
import io

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

# dataviz-skill palette, validated:
#   node scripts/validate_palette.js "#c1701c,#3987e5" --mode light -> ALL CHECKS PASS
#   (normal-vision dE 29.5, protan 27.2, both >= 3:1 on surface)
# Adding the SAC arm's violet keeps every check passing (worst adjacent CVD dE 15.9,
# all >= 3:1); see scripts/plot_checkpoint_sweep.py for the full validator run.
_VANILLA = "#c1701c"
_RESIDUAL = "#3987e5"
_RESIDUAL_SAC = "#4a3aa7"
_VANILLA_SAC = "#e87ba4"
_INK = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"


def _monitor_curve(path: str, window: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """(cumulative steps, trailing rolling-mean return) from one SB3 Monitor CSV."""
    with open(path) as f:
        body = "".join(ln for ln in f if not ln.startswith("#"))
    rows = list(csv.DictReader(io.StringIO(body)))
    r = np.asarray([float(x["r"]) for x in rows])
    steps = np.cumsum([int(x["l"]) for x in rows])
    w = min(window, r.size)
    kernel = np.ones(w) / w
    smooth = np.convolve(r, kernel, mode="valid")
    return steps[w - 1:], smooth


def _band(paths: list[str], grid: np.ndarray):
    """Per-seed curves resampled onto `grid`; returns (mean, std, n)."""
    curves = []
    for p in sorted(paths):
        s, v = _monitor_curve(p)
        curves.append(np.interp(grid, s, v, left=np.nan, right=v[-1]))
    arr = np.vstack(curves)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0), arr.shape[0]


def _steps_to(paths: list[str], threshold: float) -> list[float]:
    """Per-seed env steps at which the rolling-mean return first clears `threshold`."""
    out = []
    for p in sorted(paths):
        s, v = _monitor_curve(p)
        hit = np.flatnonzero(v > threshold)
        out.append(float(s[hit[0]]) if hit.size else float("nan"))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob-van", default="data/seedsweep/van_s*_mon.monitor.csv")
    p.add_argument("--glob-res", default="data/seedsweep/res_f2_s*_mon.monitor.csv")
    p.add_argument("--glob-sac", default=None,
                   help="optional third arm: SAC residual monitor CSVs")
    p.add_argument("--glob-sac-van", default=None,
                   help="optional fourth arm: SAC vanilla monitor CSVs")
    p.add_argument("--threshold", type=float, default=-6000.0)
    p.add_argument("--out", default="docs/journey/figures/learning_curves.png")
    args = p.parse_args()

    van, res = sorted(glob.glob(args.glob_van)), sorted(glob.glob(args.glob_res))
    if not van or not res:
        raise SystemExit(f"no monitor CSVs matched ({len(van)} vanilla, {len(res)} residual)")

    series = [("vanilla TD3", van, _VANILLA),
              ("clone + residual TD3 (frac 2.0)", res, _RESIDUAL)]
    if args.glob_sac:
        sac = sorted(glob.glob(args.glob_sac))
        if not sac:
            raise SystemExit(f"no monitor CSVs matched {args.glob_sac}")
        series.append(("clone + residual SAC (frac 2.0)", sac, _RESIDUAL_SAC))
    if args.glob_sac_van:
        sac_van = sorted(glob.glob(args.glob_sac_van))
        if not sac_van:
            raise SystemExit(f"no monitor CSVs matched {args.glob_sac_van}")
        series.append(("vanilla SAC", sac_van, _VANILLA_SAC))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=140)
    for ax, xmax, title in ((axes[0], 400_000, "Full budget"),
                            (axes[1], 100_000, "First 100k steps (detail)")):
        grid = np.linspace(1_000, xmax, 400)
        for label, paths, color in series:
            mean, std, _n = _band(paths, grid)
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.16, lw=0,
                            zorder=2)
            ax.plot(grid, mean, color=color, lw=2.0, zorder=3)
        ax.axhline(args.threshold, color=_MUTED, lw=0.9, ls=":", zorder=1)
        ax.annotate(f"{args.threshold:,.0f}", xy=(xmax, args.threshold), xytext=(-2, 4),
                    textcoords="offset points", ha="right", fontsize=7.5, color=_MUTED)
        ax.set_title(title, fontsize=10.5, color=_INK, fontweight="bold", pad=8)
        ax.set_xlabel("environment steps", fontsize=9, color=_INK)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _p: f"{v / 1000:.0f}k")
        )
        ax.grid(True, color=_GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)
        ax.tick_params(labelsize=8, colors=_INK)
    axes[0].set_ylabel("episode return (100-ep rolling mean)", fontsize=9, color=_INK)
    axes[1].set_ylim(-20_000, -3_000)  # the full-range panel's tail hides this detail

    handles = [plt.Line2D([], [], color=c, lw=2.0, label=lab) for lab, _, c in series]
    fig.legend(handles=handles, loc="lower center", ncol=len(series), frameon=False, fontsize=9,
               labelcolor=_INK, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Training return vs environment steps — bands = ±1 std over 5 training seeds",
                 fontsize=11, color=_INK, y=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")

    for label, paths, _c in series:
        hits = _steps_to(paths, args.threshold)
        arr = np.array(hits, dtype=float)
        print(f"  {label:<28} steps to return > {args.threshold:,.0f}: "
              f"{np.nanmean(arr):,.0f} +/- {np.nanstd(arr):,.0f}   per seed "
              f"{[f'{h:,.0f}' if h == h else 'never' for h in hits]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

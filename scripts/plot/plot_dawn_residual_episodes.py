"""Reacher residual RL: the baseline (dips) vs DAWN, training return and reach rate over RL training steps.

    uv run python scripts/plot/plot_dawn_residual_episodes.py

Reads the monitor CSVs and per-scenario sweeps under `data/repro/` (gitignored):
`q_res` (no fix), `dawn_r_warmup` (20k base-clone warmup transitions) and `dawn_r`
(warmup + LayerNorm critic), all residual_frac 2, soft SAC, seed 0, each with
`--suffix` appended (default `_200k`, the 200k-step reruns; `--suffix ""` plots
the original 100k runs). Both panels
start at step 0 from the base clone acting alone (deterministic, 120 frozen
scenarios); for the return that value is episode 0 of the trailing window
(`rl.curves`). DAWN's warmup transitions are collected before step 0 and are
not on this axis.
"""
from __future__ import annotations

import argparse
import collections
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rl.curves import return_curve  # noqa: E402

R = "data/repro"
# Base clone alone (a_res = 0), deterministic, 120 frozen scenarios; measured 2026-10-22
# with ResidualSelectEnv(data/dagger_clone_r3.pt, residual_frac=2.0).
CLONE_RET, CLONE_REACH = 8.02, 82
ARMS = [("residual RL, no fix (the dip)", "q_res", "#c92a2a", 2.6),
        ("+ DAWN warmup data (20k base-clone transitions)", "dawn_r_warmup", "#1c7ed6", 2.6),
        ("+ full DAWN (warmup + LayerNorm critic)", "dawn_r", "#7b2cbf", 1.4)]


def reach(run: str):
    c = collections.Counter()
    with open(f"{R}/sweep_{run}.csv") as f:
        for r in csv.DictReader(f):
            c[int(r["steps"])] += r["reached"] in ("1", "True", "true")
    s = sorted(c)
    return np.array([0, *s]) / 1000, np.array([CLONE_REACH, *[c[k] for k in s]])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suffix", default="_200k")
    p.add_argument("--out", default="docs/journey/figures/residual_dawn_episodes.png")
    args = p.parse_args()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # dy: "lowest" label offset; ey: end-value label offset (the three end values sit within 3 of each other)
    for (lab, run, col, lw), dy, ey in zip(ARMS, (-14, -2, 10), (-14, 6, -4)):
        steps, _ep, ret = return_curve(f"{R}/{run}{args.suffix}.monitor.csv", step0=CLONE_RET)
        ax[0].plot(steps / 1000, ret, color=col, lw=lw, label=lab)
        x, y = reach(run + args.suffix)
        ax[1].plot(x, y, color=col, lw=lw, marker="o", ms=3, label=lab)
        i = int(np.argmin(y))
        ax[1].annotate(f"lowest {y[i]}", (x[i], y[i]), (8, dy), textcoords="offset points",
                       fontsize=8, color=col)
        ax[1].annotate(f"{y[-1]}", (x[-1], y[-1]), (5, ey), textcoords="offset points",
                       fontsize=8.5, color=col, va="center")
    xmax = x[-1]
    ax[1].axhline(CLONE_REACH, color="k", ls=":", lw=1)
    ax[1].text(0.75 * xmax, CLONE_REACH + 2, "base clone: 82", fontsize=8)
    ax[0].axhline(CLONE_RET, color="k", ls=":", lw=1)
    ax[0].text(0.75 * xmax, CLONE_RET + 0.8, "base clone: 8.0", fontsize=8)
    for a in ax:
        a.set_xlim(0, xmax)
        a.set_xlabel("RL training steps (k)")
        a.grid(alpha=0.3)
    ax[0].set(ylabel="training episode return (trailing 100-ep mean)",
              title="Training return (exploring policy)")
    ax[1].set(ylabel="scenarios reached (of 120, deterministic policy)", ylim=(0, 121),
              title="Reach rate (checkpoint every 2k steps)")
    ax[1].legend(fontsize=8.5, loc="lower right")
    fig.suptitle("Reacher residual RL (residual_frac 2, soft SAC, seed 0): the baseline vs DAWN. "
                 "Both curves start at\nstep 0 from the base clone acting alone. DAWN's 20k warmup "
                 "transitions happen before step 0 and are not on this axis.", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

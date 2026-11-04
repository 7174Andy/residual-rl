"""Reacher deployed reach rate vs training steps, seed 0 -- journey 16's figure.

    uv run python scripts/plot/plot_reacher_warmstart_reach.py

Checkpoint reach comes from the existing sweeps (no re-evaluation):
`docs/reference/reacher_crossover.csv` (residual frac 2.0, vanilla) and
`data/bcinit_ckpt_reach.csv` (warm actor). Step 0 is each arm's clone -- the
policy the arm starts as -- scored on the same 120 frozen scenarios:
`dagger_clone_r3.pt` 82/120 (journey 13) under the residual, the refit squash
clone 79/120 (journey 16) under the warm actor. Without it the early collapse
(both arms ~37% at 25k, below their clones) is invisible. Vanilla has no prior,
so no step-0 point.
"""
from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N = 120
STEP0 = {"residual": 82, "bcinit": 79}


def read(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crossover", default="docs/reference/reacher_crossover.csv")
    p.add_argument("--bcinit", default="data/bcinit_ckpt_reach.csv")
    p.add_argument("--out", default="docs/journey/figures/warmstart_reach.png")
    args = p.parse_args()

    rows = read(args.crossover) + read(args.bcinit)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for arm, label, color in (("residual", "residual RL on clone", "C2"),
                              ("vanilla", "vanilla RL from scratch", "C1"),
                              ("bcinit", "warm actor (bcinit)", "C0")):
        pts = sorted({int(r["steps"]): int(r["reached"]) for r in rows
                      if r["arm"] == arm}.items())
        if arm in STEP0:
            pts = [(0, STEP0[arm]), *pts]
        xs, ys = zip(*pts)
        ax.plot(xs, [100 * y / N for y in ys], "o-", ms=4, color=color, label=label)
    ax.set(title="Reacher deployed reach rate vs training steps, seed 0",
           xlabel="training steps", ylabel=f"reach rate (%) — {N} frozen scenarios, greedy")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

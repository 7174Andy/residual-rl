"""Reacher training return, warm actor vs residual vs vanilla, seed 0 (journey 16).

    uv run python scripts/plot/plot_reacher_warmstart_returns.py

Left: the full 400k budget (8000 episodes); right: the first 2000 episodes. The
curves are `rl.curves` trailing means, so they start at episode 10 rather than
100. The two clone-based arms start at step 0 with their clone's deterministic
mean return on the 120 frozen scenarios (measured 2026-10-22) -- what the reach
rate scores at step 0 -- counted as episode 0 of the window. Without it the warm
arm's curve starts at -8 and its collapse from the clone is invisible. Caveat:
the residual's step-0 behaviour policy explores at sigma ~1 (SAC's zero-init) and
scores -7.5 sampled, so its first drop is mostly exploration noise; the warm
arm's sigma 0.1 costs little (5.2 sampled vs 6.7).
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rl.curves import return_curve  # noqa: E402

# (label, monitor CSV, color, step-0 deterministic return of the arm's clone or None)
ARMS = [
    ("warm actor (bcinit, 400k)", "data/reacher_bcinit_400k.monitor.csv", "C0", 6.69),     # dagger_clone_r3_squash
    ("residual RL on clone (400k)", "data/reacher_residual_400k.monitor.csv", "C2", 8.02),  # dagger_clone_r3, a_res=0
    ("vanilla RL from scratch (400k)", "data/reacher_vanilla_400k.monitor.csv", "C1", None),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="docs/journey/figures/warmstart_returns.png")
    args = p.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, xmax, title in ((axes[0], 8000, "Full budget (400k steps)"),
                            (axes[1], 2000, "First 2000 episodes (100k steps)")):
        for label, path, color, step0 in ARMS:
            _s, ep, r = return_curve(path, step0=step0)
            ax.plot(ep, r, color=color, lw=1.5, label=label)
        ax.set(xlim=(0, xmax), xlabel="training episode", title=title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("episode return (trailing 100-ep mean)")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Reacher: warm actor vs residual RL vs vanilla RL — training return, seed 0")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

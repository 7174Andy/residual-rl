"""What the clone's training set actually looks like in its own feature space.

    uv run python scripts/plot/plot_clone_dataset_coverage.py

30,000 rows sounds like a cloud. It is 200 threads: each episode is 150 states
sampled densely along one smooth path, and the gap between threads is 26x the
gap along one. Panels, left to right:

A  the first two principal components of the standardized 99-D features, one
   line per episode. Training episodes grey, held-out episodes red -- the red
   threads run through regions no grey thread visits.
B  the same fact as two histograms: step-to-step distance within an episode
   against distance to the nearest state in ANY other episode.
C  what fraction of states have a training neighbour close enough for the
   expert's decision to transfer (cosine >= 0.64 needs ~0.93 standardized
   units, measured for journey 18 by a since-removed script).

Reads data/pcd_r0.npz + data/pcd_r1.npz; nothing is cached, so it re-runs from
a clone as long as those collections exist.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

BLUE, ORANGE, MUTED2, CRITICAL = "#2a78d6", "#eb6834", "#8a8a86", "#d03b3b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"
GREEN = "#2f7d32"
EP = 150
USEFUL_RAD = 0.93      # distance at which expert increments agree at cos +0.64


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+",
                   default=["data/pcd_r0.npz", "data/pcd_r1.npz"])
    p.add_argument("--n-query", type=int, default=5000)
    p.add_argument("--out", default="docs/reference/panda_clone_dataset_coverage.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
    })

    X = np.vstack([np.load(f)["features"] for f in args.data]).astype(np.float32)
    ep = np.arange(len(X)) // EP
    n_ep = int(ep.max()) + 1
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-8] = 1.0
    A = (X - mu) / sd
    held = np.isin(ep, np.r_[90:100, 190:200])
    print(f"{len(X)} rows, {n_ep} episodes, {X.shape[1]} features")

    # --- PCA (SVD on the standardized matrix) ------------------------------
    Ac = A - A.mean(0)
    _, S, Vt = np.linalg.svd(Ac, full_matrices=False)
    P = Ac @ Vt[:2].T
    var = (S[:2] ** 2) / (S ** 2).sum()

    # --- nearest CROSS-episode neighbour -----------------------------------
    rng = np.random.default_rng(0)
    qi = rng.choice(len(A), min(args.n_query, len(A)), replace=False)
    An = (A ** 2).sum(1)
    cross = np.empty(len(qi), dtype=np.float32)
    for s in range(0, len(qi), 256):
        e = min(s + 256, len(qi))
        blk = A[qi[s:e]]
        d2 = An[None, :] - 2.0 * (blk @ A.T) + (blk ** 2).sum(1)[:, None]
        d2[ep[qi[s:e]][:, None] == ep[None, :]] = np.inf
        cross[s:e] = np.sqrt(np.maximum(d2.min(axis=1), 0))

    within = np.concatenate([np.linalg.norm(np.diff(A[ep == e], axis=0), axis=1)
                             for e in range(n_ep)])

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14.6, 4.6))

    # --- A: threads, not a cloud -------------------------------------------
    # 200 overlapping threads at 34% explained variance read as a blob, so a
    # legible subset is drawn; the structure, not the density, is the point.
    tr_ids = [e for e in range(n_ep) if not held[ep == e][0]]
    he_ids = [e for e in range(n_ep) if held[ep == e][0]]
    show_tr = rng.choice(tr_ids, 25, replace=False)
    show_he = rng.choice(he_ids, 5, replace=False)
    for e in show_tr:
        m = ep == e
        a.plot(P[m, 0], P[m, 1], color=MUTED2, lw=0.9, alpha=0.85, zorder=1)
    for e in show_he:
        m = ep == e
        a.plot(P[m, 0], P[m, 1], color=CRITICAL, lw=1.3, alpha=0.95, zorder=3)
    a.plot([], [], color=MUTED2, lw=1.2, label="25 of 180 training episodes")
    a.plot([], [], color=CRITICAL, lw=1.4, label="5 of 20 held-out")
    a.set_xlabel(f"PC1 ({100 * var[0]:.0f}% of variance)")
    a.set_ylabel(f"PC2 ({100 * var[1]:.0f}%)")
    a.set_title("A · 30,000 rows is 200 threads,\nnot a cloud", loc="left",
                fontsize=10.5)
    a.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # --- B: the gap between threads ----------------------------------------
    bins = np.logspace(-2, 1.1, 60)
    b.hist(within, bins=bins, color=GREEN, alpha=0.75,
           label=f"along one episode (median {np.median(within):.2f})")
    b.hist(cross, bins=bins, color=CRITICAL, alpha=0.75,
           label=f"to the nearest OTHER episode (median {np.median(cross):.2f})")
    b.set_xscale("log")
    b.axvline(USEFUL_RAD, color=INK2, ls="--", lw=1.3)
    b.text(USEFUL_RAD * 1.15, b.get_ylim()[1] * 0.55,
           "close enough for the\nexpert's decision\nto transfer",
           fontsize=8.5, color=INK2, va="top")
    b.set_xlabel("distance in standardized feature space")
    b.set_ylabel("states")
    b.set_title(f"B · The gap between threads is "
                f"{np.median(cross) / np.median(within):.0f}x\nthe gap along one",
                loc="left", fontsize=10.5)
    b.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # --- C: how much of the space is actually covered ----------------------
    xs = np.logspace(-1, 1.1, 200)
    frac = [(cross <= x).mean() * 100 for x in xs]
    c.plot(xs, frac, color=BLUE)
    c.set_xscale("log")
    c.axvline(USEFUL_RAD, color=CRITICAL, ls="--", lw=1.3)
    hit = (cross <= USEFUL_RAD).mean() * 100
    c.plot([USEFUL_RAD], [hit], "o", color=CRITICAL, ms=7, zorder=4)
    c.annotate(f"only {hit:.1f}% of states have a\nneighbour this close",
               xy=(USEFUL_RAD, hit), xytext=(1.25, 26), fontsize=9,
               color=CRITICAL,
               arrowprops=dict(arrowstyle="->", color=CRITICAL, lw=1.1))
    c.set_xlabel("radius in standardized feature space")
    c.set_ylabel("% of states with a neighbour inside the radius")
    c.set_ylim(0, 100)
    c.set_title("C · Almost nothing is near enough\nto be informative",
                loc="left", fontsize=10.5)
    c.grid(axis="y")

    fig.suptitle("The clone's training set: dense along 200 trajectories, "
                 "empty everywhere else", x=0.005, ha="left", fontsize=11.5,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    print(f"  within-episode median {np.median(within):.3f}, "
          f"cross-episode median {np.median(cross):.3f}, "
          f"{hit:.1f}% inside {USEFUL_RAD}")


if __name__ == "__main__":
    main()

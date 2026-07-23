#!/usr/bin/env python
"""Plot the TD3 residual's training return curve (mean episode return vs episode).

Default input is the committed curve CSV captured from the shipped seed-0 run
(`docs/journey/figures/residual_return.csv`, columns ``episode,ep_rew_mean``), so
the figure regenerates from committed data with no training re-run:

    uv run python scripts/plot_training_return.py

Pass ``--monitor <path>`` to derive the curve fresh from a Stable-Baselines3
``Monitor`` CSV (raw per-episode returns) produced by a new training run via
``scripts/train_residual.py --monitor-out <path>`` — the raw returns are smoothed
with a trailing rolling mean to match ``ep_rew_mean``.
"""
from __future__ import annotations

import argparse
import csv
import io

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _from_curve_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a committed ``episode,ep_rew_mean`` curve CSV."""
    ep, rew = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ep.append(int(row["episode"]))
            rew.append(float(row["ep_rew_mean"]))
    return np.asarray(ep), np.asarray(rew)


def _from_monitor_csv(path: str, window: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Read an SB3 Monitor CSV (raw returns ``r``) and return a rolling-mean curve.

    SB3 prepends a ``#{...}`` metadata line before the ``r,l,t`` header, so it is
    skipped. The trailing rolling mean over ``window`` episodes mirrors SB3's
    ``ep_rew_mean``.
    """
    with open(path) as f:
        body = "".join(ln for ln in f if not ln.startswith("#"))
    returns = np.asarray([float(r["r"]) for r in csv.DictReader(io.StringIO(body))])
    if returns.size == 0:
        raise ValueError(f"{path} has no episodes")
    w = min(window, returns.size)
    kernel = np.ones(w) / w
    roll = np.convolve(returns, kernel, mode="valid")
    episodes = np.arange(w, returns.size + 1)  # x = end-of-window episode index
    return episodes, roll


def _parse_compare(specs: list[str]) -> list[tuple[str, str]]:
    """Parse ``LABEL:PATH`` strings into (label, path) pairs."""
    pairs = []
    for spec in specs:
        label, _, path = spec.partition(":")
        if not path:
            raise ValueError(f"--compare entry {spec!r} must be LABEL:PATH")
        pairs.append((label, path))
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="docs/journey/figures/residual_return.csv",
                   help="committed curve CSV (episode,ep_rew_mean)")
    p.add_argument("--monitor", default=None,
                   help="SB3 Monitor CSV of raw returns; overrides --input")
    p.add_argument("--out", default="docs/journey/figures/residual_return.png")
    p.add_argument("--window", type=int, default=100,
                   help="rolling-mean window when reading --monitor")
    p.add_argument("--save-curve", default=None,
                   help="also write the (possibly --monitor-derived) curve to this "
                        "CSV path (episode,ep_rew_mean), for committing")
    p.add_argument("--compare", nargs="+", default=None, metavar="LABEL:PATH",
                   help="overlay multiple committed curve CSVs instead of a single "
                        "curve, e.g. --compare 200k:docs/.../residual_return.csv "
                        "400k:docs/.../residual_return_400k.csv")
    args = p.parse_args()

    if args.compare:
        pairs = _parse_compare(args.compare)
        seq_colors = ["#3987e5", "#184f95", "#1baf7a", "#eb6834"]
        curves = [(label, color, *_from_curve_csv(path))
                  for (label, path), color in zip(pairs, seq_colors)]

        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        # Draw the longest-running curve first so a shorter curve that overlaps
        # it almost exactly (same seed/hyperparameters, just fewer steps) still
        # shows on top instead of being fully occluded.
        draw_order = sorted(range(len(curves)), key=lambda i: -curves[i][2].size)
        for rank, i in enumerate(draw_order):
            label, color, ep, rew = curves[i]
            ax.plot(ep, rew, color=color, lw=1.6, label=label, zorder=3 + rank)
            best = int(np.argmax(rew))
            ax.annotate(f"best ≈ {rew[best]:,.0f}", (ep[best], rew[best]),
                        textcoords="offset points", xytext=(6, -14), fontsize=8,
                        color="#0b0b0b", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.85))
            ax.scatter([ep[best]], [rew[best]], color="#e34948", zorder=9, s=28)
        # legend follows the order --compare was given, not draw order
        handles, labels_ = ax.get_legend_handles_labels()
        order = [labels_.index(label) for label, *_ in curves]
        ax.legend([handles[i] for i in order], [labels_[i] for i in order],
                   frameon=False, loc="lower right")
        ax.set_xlabel("training episode")
        ax.set_ylabel("mean episode return  (SB3 ep_rew_mean)")
        ax.set_title("TD3 residual — training return, 200k vs 400k steps")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out, dpi=130)
        print(f"wrote {args.out}  ({len(pairs)} curves)")
        return 0

    if args.monitor:
        ep, rew = _from_monitor_csv(args.monitor, window=args.window)
    else:
        ep, rew = _from_curve_csv(args.input)

    if args.save_curve:
        with open(args.save_curve, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "ep_rew_mean"])
            writer.writerows(zip(ep.tolist(), rew.tolist()))
        print(f"wrote {args.save_curve}  ({ep.size} rows)")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(ep, rew, color="#1f77b4", lw=1.6)
    best = int(np.argmax(rew))
    ax.scatter([ep[best]], [rew[best]], color="#d62728", zorder=5, s=28)
    ax.annotate(f"best ≈ {rew[best]:,.0f}", (ep[best], rew[best]),
                textcoords="offset points", xytext=(6, -12), fontsize=9,
                color="#d62728")
    ax.set_xlabel("training episode")
    ax.set_ylabel("mean episode return  (SB3 ep_rew_mean)")
    ax.set_title("TD3 residual — training return (seed 0, 200k steps)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  ({ep.size} points, episodes {int(ep[0])}..{int(ep[-1])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

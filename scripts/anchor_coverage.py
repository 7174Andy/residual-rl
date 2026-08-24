"""Coverage-driven anchor selection: is there cluster structure, and what is r_K?

Replaces the `k-medoids -> PCA -> pick K` workflow with
`task configurations -> coverage anchors -> r_K curve`. Three parts:

  [1] SILHOUETTE vs K. If it stays flat and low (~0.1-0.2) at every K, the
      configurations form a continuous manifold and no K is "the natural" one --
      which makes any clustering-quality criterion for choosing K meaningless and
      forces the choice onto coverage instead.

  [2] FARTHEST-POINT SAMPLING. `a_new = argmax_q min_i ||q - a_i||`. Optimizes
      worst-case coverage (k-center) rather than k-medoids' mean distance. On a
      manifold with no clusters this is the objective that actually matters: what
      decides success is whether ANY configuration sits outside every library's
      valid radius, not whether the average one is close.

  [3] THE r_K CURVE, `r_K = max_q min_i ||q - a_i||`, for K = 1..k_max. Where it
      flattens is the candidate range for the expensive DeePC experiments. Read
      against the MEASURED usable radius (~0.5-1.0 rad, from
      `scripts/verify_libraries.py`): the useful K is where `r_K` drops below it,
      not where the curve looks visually elbowed.

    uv run python scripts/anchor_coverage.py
    uv run python scripts/anchor_coverage.py --anchors data/panda_anchors_k4_ik.npz --k-max 30
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from panda.anchors import (  # noqa: E402
    _pairwise, coverage, farthest_point, kmedoids, silhouette,
)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, CRITICAL = "#0b0b0b", "#52514e", "#b8b7b2", "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anchors", default="data/panda_anchors_k4_ik.npz",
                   help="npz carrying Q (the task configurations) and weights")
    p.add_argument("--k-max", type=int, default=20, help="largest K for the r_K curve")
    p.add_argument("--silhouette-k", type=int, nargs="+",
                   default=[2, 4, 6, 8, 10, 12, 16])
    p.add_argument("--usable-radius", type=float, default=1.0,
                   help="measured library validity radius, rad (verify_libraries.py)")
    p.add_argument("--out", default="docs/reference/panda_coverage.png")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with np.load(args.anchors) as z:
        Q = z["Q"]
        w = z["weights"] if z["weights"].size else None
        ik = str(z["ik"]) if "ik" in z.files else "unknown"
    rng = np.random.default_rng(args.seed)
    D = _pairwise(Q, w)
    print(f"{len(Q)} task configurations from {args.anchors} (ik={ik})")

    # [1] Is there cluster structure at ANY K?
    print("\n[1] silhouette vs K -- flat and low means a continuum, not clusters")
    print(f"  {'K':>4}{'silhouette':>13}{'k-medoids r_K':>16}{'mean nn':>10}")
    sil = []
    for k in args.silhouette_k:
        med, labels, _ = kmedoids(D, k, rng, n_init=10)
        s = silhouette(D, labels)
        cov = coverage(Q, Q[med], w)
        sil.append(s)
        print(f"  {k:>4}{s:>13.3f}{cov['r']:>16.2f}{cov['mean_nn']:>10.2f}")
    lo_s, hi_s = min(sil), max(sil)
    print(f"  -> silhouette spans {lo_s:.3f}-{hi_s:.3f} across K"
          + ("  CONTINUUM: no K is natural; choose K by coverage."
             if hi_s < 0.35 else "  some structure present."))

    # [2]/[3] Farthest-point sampling, nested, so one pass gives every K.
    fps = farthest_point(Q, args.k_max, w=w)
    print(f"\n[3] farthest-point coverage, K = 1..{args.k_max}")
    print(f"  {'K':>4}{'r_K (worst)':>14}{'mean nn':>10}{'vs k-medoids r_K':>19}")
    kmed_r = {}
    for k in args.silhouette_k:
        med, _, _ = kmedoids(D, k, rng, n_init=10)
        kmed_r[k] = coverage(Q, Q[med], w)["r"]
    for k in range(1, args.k_max + 1):
        extra = f"{kmed_r[k]:>19.2f}" if k in kmed_r else " " * 19
        flag = "  <- below usable radius" if fps["r"][k - 1] <= args.usable_radius else ""
        print(f"  {k:>4}{fps['r'][k - 1]:>14.2f}{fps['mean_nn'][k - 1]:>10.2f}{extra}{flag}")

    below = np.flatnonzero(fps["r"] <= args.usable_radius)
    k_star = int(below[0] + 1) if below.size else None
    print(f"\n  usable radius = {args.usable_radius} rad ->"
          + (f" r_K first drops below it at K = {k_star}" if k_star
             else f" never reached by K = {args.k_max}"))

    # --- figure
    K = np.arange(1, args.k_max + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(K, fps["r"], color=SERIES[0], marker="o", ms=4, zorder=3,
            label="farthest-point (k-center)")
    a1.plot(K, fps["mean_nn"], color=SERIES[2], marker="o", ms=4, zorder=3,
            label="mean nearest-anchor")
    ks = sorted(kmed_r)
    a1.plot(ks, [kmed_r[k] for k in ks], color=SERIES[1], marker="s", ms=5,
            ls="--", zorder=3, label="k-medoids worst-case")
    a1.axhline(args.usable_radius, color=CRITICAL, lw=1.2, ls="--", zorder=1)
    a1.text(args.k_max * 0.42, args.usable_radius * 1.06,
            f"measured usable radius ({args.usable_radius} rad)",
            color=CRITICAL, fontsize=8)
    if k_star:
        a1.plot([k_star], [fps["r"][k_star - 1]], marker="o", ms=12, mfc="none",
                mec=CRITICAL, mew=2, zorder=4)
        a1.annotate(f"K = {k_star}", xy=(k_star, fps["r"][k_star - 1]),
                    xytext=(6, 10), textcoords="offset points",
                    color=CRITICAL, fontsize=9, weight="bold")
    a1.set_xlabel("number of anchors K")
    a1.set_ylabel("nearest-anchor distance (rad)")
    a1.set_title("A · Coverage — $r_K = \\max_q \\min_i \\|q - a_i\\|$",
                 loc="left", color=INK)
    a1.legend(frameon=False, fontsize=8)
    a1.grid(True, axis="y", zorder=0)
    a1.set_axisbelow(True)

    a2.plot(args.silhouette_k, sil, color=SERIES[3], marker="o", ms=5, zorder=3)
    a2.axhspan(0.0, 0.35, color="#e8e7e3", zorder=0)
    a2.text(args.silhouette_k[-1], 0.33, "no meaningful cluster structure",
            ha="right", va="top", fontsize=8, color=INK2)
    a2.set_ylim(0, max(0.5, max(sil) * 1.25))
    a2.set_xlabel("number of clusters K")
    a2.set_ylabel("mean silhouette (full 7-D)")
    a2.set_title("B · Is there cluster structure at any K?", loc="left", color=INK)
    a2.grid(True, axis="y", zorder=0)
    a2.set_axisbelow(True)

    fig.suptitle(f"Anchor coverage — {len(Q)} task configurations, ik={ik}",
                 x=0.005, ha="left", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"\nwrote {args.out}")

    out_npz = args.anchors.replace(".npz", "_fps.npz")
    np.savez(out_npz, Q=Q, fps_idx=fps["idx"], fps_anchors=Q[fps["idx"]],
             r=fps["r"], mean_nn=fps["mean_nn"],
             weights=np.array([] if w is None else w))
    print(f"wrote {out_npz}  (nested FPS anchors: first K rows are the K-anchor set)")


if __name__ == "__main__":
    main()

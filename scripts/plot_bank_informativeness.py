"""Journey 15's data-distribution figure: same controller, three banks.

Regenerates `docs/reference/panda_bank_informativeness.png`, which previously
existed only as an orphan PNG from a throwaway script.

Layout:

  A1-A3  one panel PER COLLECTION DESIGN — where that design puts its samples
         in tip x-y (normalized per design, so the panels compare SPREAD, not
         data volume), with its anchor points and the 78 goals on top.
  B      joint-space distance from each of the 78 episode starts to its nearest
         collected sample, as a CDF against the 0.5 rad validity radius.
  C      median prediction skill of the columns Select-DPC actually selects,
         against the reach each bank produced.

Why A is faceted rather than overlaid: all three banks blanket the same tip
volume, so drawing them on shared axes — as scatter OR as contours — buries the
two smaller banks under the 20k one and answers no question. Separate panels
show each design's own structure, and the ANCHOR COUNT in each title is the
quantity the coverage argument actually turns on: 65 anchors against 20,000.

`anchors[i]` is trajectory `i`'s start configuration (verified: it equals
`q_i[0]` in all three payloads), so an anchor's tip is `tip_i[0]` and needs no
forward-kinematics call.

    uv run python scripts/plot_bank_informativeness.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

INK, INK2, MUTED, RED = "#0b0b0b", "#52514e", "#b8b7b2", "#c1443c"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6,
})

VALID_RAD = 0.5

# Okabe-Ito, colour-vision safe, paired with distinct line styles so panel B
# does not depend on hue alone.
BANKS = [
    ("random OU (old)", "data/panda_uniform_libs.npz", "#d55e00", "dotted"),
    ("task servo 1k", "data/panda_taskbank_v1.npz", "#009e73", "dashed"),
    ("task servo 20k", "data/panda_taskbank_20k_sdpc.npz", "#0072b2", "solid"),
]

# Median skill of SELECTED columns, per bank, with the reach each one produced.
# Measured by scripts/measure_selection_distance.py during the Phase-1 gate
# investigation and recorded in data/expert_phase1_log.md; re-deriving them here
# would mean re-running that gate for numbers that are already pinned, so they
# are transcribed with provenance instead.
SKILL = [(-1.83, "reach 0/10"), (-0.29, "reach n/a (gate)"), (+0.40, "reach 70/78")]
REACHER_SKILL = 0.72        # journey 12's working reacher expert, for scale


def bank_arrays(path: str, max_traj: int | None = None):
    """`(q, tip, anchor_tips)` for a collection payload.

    `anchor_tips` is `tip_i[0]` per trajectory — the tip at each anchor, i.e.
    one point per independent placement in the collection.
    """
    with np.load(path) as z:
        n = int(z["anchors"].shape[0])
        if max_traj is not None:
            n = min(n, max_traj)
        tips = [z[f"tip_{i}"] for i in range(n)]
        q = np.vstack([z[f"q_{i}"] for i in range(n)])
    anchor_tips = np.array([t[0] for t in tips])
    return q, np.vstack(tips), anchor_tips


def nearest_distances(starts: np.ndarray, q: np.ndarray,
                      chunk: int = 200_000) -> np.ndarray:
    """Joint-space distance from each start to its nearest collected sample.

    Chunked over the bank: the 20k bank holds 3M samples, and a full
    (78, 3M) distance matrix would be 1.9 GB.

    Distances come from the expansion `|a-b|^2 = |a|^2 - 2a·b + |b|^2` rather
    than from broadcasting `starts[:, None] - block[None, :]`. The broadcast
    form materializes an (78, chunk, 7) temporary — 873 MB at chunk=200k — while
    this one peaks at the (78, chunk) result itself, 125 MB.
    """
    s2 = (starts ** 2).sum(1)[:, None]
    best = np.full(starts.shape[0], np.inf)
    for s in range(0, q.shape[0], chunk):
        block = q[s:s + chunk]
        d2 = s2 - 2.0 * (starts @ block.T) + (block ** 2).sum(1)[None, :]
        best = np.minimum(best, np.sqrt(np.maximum(d2.min(axis=1), 0.0)))
    return best


def tip_histogram(tip, extent, bins=46):
    """Share of a design's samples per tip-space cell, in percent.

    NORMALIZED per design, deliberately. Raw counts confound two things: how a
    design spreads its data, and how much data it happened to collect. Plotting
    counts makes the 20k bank uniformly darker purely because it holds 30x more
    samples, which reads as "tip coverage improved with more anchors" — the exact
    conclusion this panel exists to rule out. The old bank is the disproof: it
    used MORE steps per trajectory (1500 against 150) and still failed.

    Normalizing leaves only the shape of the distribution, which is the question
    panel A actually asks. The count is in each panel's title, and the coverage
    argument lives in panel B.
    """
    H, xe, ye = np.histogram2d(tip[:, 0], tip[:, 1], bins=bins, range=extent)
    return 100.0 * H / max(H.sum(), 1.0), xe, ye


def tip_panel(ax, hist, anchor_tips, goals, color, extent, norm):
    """Sample density for ONE collection design, with its anchors and the goals.

    `norm` is passed in and SHARED across the three panels on purpose. Letting
    each panel auto-scale to its own maximum would make the same shade mean a
    different count in each one, which is exactly the ambiguity a reader would
    (reasonably) resolve by comparing shades across panels and get wrong.

    The scale is logarithmic: the random-OU bank puts 1500 steps around each of
    only 65 anchors, so its per-cell shares span orders of magnitude and a linear
    scale would show a few hot cells and nothing else.

    Note the "same tip volume" reading is carried by the coloured AREA, not by
    the shade, so sharing the norm does not distort it.
    """
    cmap = LinearSegmentedColormap.from_list("bank", ["#f2f1ee", color])
    H, xe, ye = hist
    mesh = ax.pcolormesh(xe, ye, np.ma.masked_where(H.T == 0, H.T), cmap=cmap,
                         norm=norm, zorder=2)
    # 20,000 anchor markers would ink over the density entirely, so dense banks
    # get a deterministic subsample and the legend says exactly what is drawn.
    # The true count is in the panel title either way.
    n_anchor = len(anchor_tips)
    cap = 1200
    if n_anchor > cap:
        idx = np.random.default_rng(0).choice(n_anchor, cap, replace=False)
        shown, note = anchor_tips[idx], f"{cap:,} of {n_anchor:,} shown"
    else:
        shown, note = anchor_tips, f"{n_anchor:,}"
    size = 26 if n_anchor < 200 else 11
    ax.scatter(shown[:, 0], shown[:, 1], s=size, facecolor="none",
               edgecolor=INK, linewidths=0.5, alpha=0.75, zorder=4,
               label=f"anchors ({note})")
    ax.scatter(goals[:, 0], goals[:, 1], marker="*", s=54, color=RED,
               edgecolor="white", linewidths=0.35, zorder=5,
               label="the 78 goals")
    ax.set_xlim(*extent[0])
    ax.set_ylim(*extent[1])
    ax.set_aspect("equal")
    ax.set_xlabel("tip x (m)")
    return mesh


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", default="data/panda_scenarios_v1.npz")
    p.add_argument("--max-traj", type=int, default=None,
                   help="cap trajectories per bank (for a fast smoke run)")
    p.add_argument("--out", default="docs/reference/panda_bank_informativeness.png")
    args = p.parse_args()

    with np.load(args.scenarios) as z:
        starts, goals = np.asarray(z["qpos"]), np.asarray(z["goal"])

    fig = plt.figure(figsize=(13.2, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.50,
                          wspace=0.22)
    top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    b = fig.add_subplot(gs[1, :2])
    c = fig.add_subplot(gs[1, 2])

    extent = [(-0.92, 0.92), (-0.92, 0.92)]

    # Pass 1 gathers the small per-bank summaries (a 46x46 histogram and the
    # anchor tips) so the colour scale can be shared across panels; the big
    # (samples x 7) arrays are released as soon as each bank is summarised, which
    # is what keeps peak memory at one bank rather than three.
    summaries = []
    for label, path, color, ls in BANKS:
        q, tip, anchor_tips = bank_arrays(path, args.max_traj)
        n_traj = anchor_tips.shape[0]
        n_steps = q.shape[0] // max(n_traj, 1)
        print(f"{label}: {q.shape[0]:,} samples from {n_traj:,} anchors "
              f"({n_steps} steps each)", flush=True)
        summaries.append({
            "hist": tip_histogram(tip, extent), "anchor_tips": anchor_tips,
            "n_traj": n_traj, "n_steps": n_steps, "n_samples": q.shape[0],
        })

        d = nearest_distances(starts, q)
        b.plot(np.sort(d), np.arange(1, d.size + 1) / d.size, lw=2.2,
               color=color, ls=ls,
               label=f"{label} — median {np.median(d):.2f} rad, "
                     f"{100 * (d < VALID_RAD).mean():.0f}% inside the radius")
        print(f"  median nearest {np.median(d):.2f} rad, "
              f"inside {VALID_RAD} rad: {100 * (d < VALID_RAD).mean():.0f}%",
              flush=True)
        del q, tip

    # Pass 2 draws the tip panels on ONE shared log colour scale, so a given
    # shade means the same SHARE of a design's samples in every panel. Both
    # halves matter: normalized (tip_histogram) removes the total-count
    # confound, shared removes the per-panel-autoscale ambiguity.
    vmax = max(s["hist"][0].max() for s in summaries)
    vmin = min(h[h > 0].min() for h in (s["hist"][0] for s in summaries))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    for i, ((label, _path, color, _ls), s) in enumerate(zip(BANKS, summaries)):
        mesh = tip_panel(top[i], s["hist"], s["anchor_tips"], goals, color,
                         extent, norm)
        top[i].set_title(
            f"A{i + 1} · {label}\n{s['n_traj']:,} anchors × {s['n_steps']} "
            f"steps = {s['n_samples']:,} samples",
            loc="left", color=INK, fontsize=9.5)
        if i == 0:
            top[i].set_ylabel("tip y (m)")
        top[i].legend(fontsize=7.2, loc="lower left", framealpha=0.93,
                      facecolor="#fcfcfb", edgecolor=MUTED)
        # Ticks only. The bars share one scale, so labelling each of the three
        # would repeat the same sentence and collide with the next panel's axis;
        # the row caption below states the unit once.
        cb = fig.colorbar(mesh, ax=top[i], fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6.5)

    fig.text(0.5, top[0].get_position().y0 - 0.085,
             "colour = share of THAT design's samples per tip-space cell "
             "(%, log, shared across A1–A3) — normalized so the panels compare "
             "spread, not how much data each design happened to collect",
             ha="center", fontsize=8.2, color=INK2)

    b.axvline(VALID_RAD, color=RED, lw=1.3, ls="--")
    b.annotate("0.5 rad validity radius", (VALID_RAD, 0.60),
               textcoords="offset points", xytext=(-6, 0), ha="right",
               fontsize=8, color=RED)
    b.set_xscale("log")
    b.set_xlabel("joint-space distance to nearest collected sample (rad)")
    b.set_ylabel("fraction of the 78\nepisode starts")
    b.set_ylim(0, 1.02)
    b.legend(frameon=False, fontsize=8, loc="upper left")
    b.grid(True, zorder=0)
    b.set_axisbelow(True)
    b.set_title("B · Coverage of the episode starts, in JOINT space — the panel "
                "that predicted the outcome", loc="left", color=INK)

    xs = np.arange(len(BANKS))
    c.bar(xs, [s for s, _r in SKILL], 0.55,
          color=[col for _l, _p, col, _ls in BANKS], zorder=3)
    c.axhline(0, color=INK2, lw=0.9)
    c.axhline(REACHER_SKILL, color=MUTED, lw=1.2, ls=":")
    c.annotate("reacher expert (works, 80%)", (len(BANKS) - 0.5, REACHER_SKILL),
               textcoords="offset points", xytext=(0, 4), ha="right",
               fontsize=8, color=INK2)
    for i, (s, reach) in enumerate(SKILL):
        c.annotate(reach, (i, s), ha="center", va="bottom" if s > 0 else "top",
                   xytext=(0, 5 if s > 0 else -5), textcoords="offset points",
                   fontsize=8, color=INK, fontweight="bold")
    c.set_xticks(xs, [lab.replace(" ", "\n") for lab, *_r in BANKS], fontsize=8)
    c.set_ylim(-2.1, 1.05)
    c.set_ylabel("median skill of SELECTED columns")
    c.grid(True, axis="y", zorder=0)
    c.set_axisbelow(True)
    c.set_title("C · Informativeness → outcome", loc="left", color=INK)

    fig.suptitle("Panda Select-DPC — the data distribution decides everything: "
                 "same controller, three collection designs",
                 x=0.005, ha="left", color=INK2, fontsize=10.5)
    fig.subplots_adjust(left=0.06, right=0.965, top=0.90, bottom=0.07,
                        wspace=0.30)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

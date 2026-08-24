"""Figures backing the Panda linear-region, state-space and anchor measurements.

Every number plotted is recomputed here from the model, not transcribed -- these
are the same computations `scripts/measure_linear_region.py` and
`scripts/select_panda_anchors.py` print, rendered.

    uv run python scripts/plot_panda_analysis.py --figure all
    uv run python scripts/plot_panda_analysis.py --figure linear --out-dir docs/reference

Palette is the validated 7-slot categorical set (all six checks PASS; the
contrast WARN is discharged by direct-labelling every series).
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_linear_region as mlr  # noqa: E402

from panda.model import frame_skip, load_model, safe_box, tip_id  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})


def _grid(ax):
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)


def fig_linear(out: str, n_samples: int = 150) -> None:
    """Panel A: error vs horizon excursion. Panel B: error vs anchor offset."""
    model, data = load_model()
    fs, tip = frame_skip(model), tip_id(model)
    lo, hi = safe_box(model)
    q_a = np.asarray(model.key_qpos[0], np.float64).copy()
    q_a[0] = -0.6
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- A: the anchor's own linearization error as the horizon excursion grows.
    probes = [0.02, 0.05, 0.08, 0.10, 0.12, 0.16, 0.20]
    errs = []
    for p in probes:
        rng = np.random.default_rng(0)
        Us = rng.uniform(-p, p, (n_samples, mlr.N_HORIZON, model.nu))
        Y, free = mlr.responses(model, data, q_a, Us, fs, tip, lo, hi)
        errs.append(mlr.horizon_err(Us, Y, free, mlr.fit_slope(Us, Y, free)) * 1e3)
    exc = [p * mlr.N_HORIZON for p in probes]
    a1.axhline(25, color=CRITICAL, lw=1.2, ls="--", zorder=1)
    a1.text(0.30, 26.5, "25 mm  (½ goal tolerance)", color=CRITICAL, fontsize=8)
    a1.plot(exc, errs, color=SERIES[0], marker="o", ms=5, zorder=3)
    op = 0.20 * mlr.N_HORIZON
    a1.plot([op], [errs[-1]], marker="o", ms=11, mfc="none", mec=CRITICAL, mew=2, zorder=4)
    a1.annotate(f"env DELTA_MAX=0.2\n{errs[-1]:.0f} mm at the anchor",
                xy=(op, errs[-1]), xytext=(op - 1.55, errs[-1] - 2),
                color=CRITICAL, fontsize=8, ha="left")
    a1.set_xlabel("horizon excursion  N · δ  (rad)")
    a1.set_ylabel("linearization error at the anchor (mm)")
    a1.set_title("A · Horizon excursion sets the error", loc="left", color=INK)
    _grid(a1)

    # --- B: transfer error as the test point moves away, per joint.
    rng = np.random.default_rng(0)
    probe = 0.05
    Us = rng.uniform(-probe, probe, (n_samples, mlr.N_HORIZON, model.nu))
    Y_a, free_a = mlr.responses(model, data, q_a, Us, fs, tip, lo, hi)
    G = mlr.fit_slope(Us, Y_a, free_a)
    offsets = np.linspace(0.2, 1.4, 7)
    ends = []
    a2.axhline(25, color=CRITICAL, lw=1.2, ls="--", zorder=1)
    for j in range(model.nq):
        xs, ys = [], []
        for o in offsets:
            q_t = q_a.copy()
            q_t[j] += o
            if np.any(q_t < lo) or np.any(q_t > hi):
                q_t[j] = q_a[j] - o
                if np.any(q_t < lo) or np.any(q_t > hi):
                    continue
            try:
                Y_t, free_t = mlr.responses(model, data, q_t, Us, fs, tip, lo, hi)
            except ValueError:
                continue
            xs.append(o)
            ys.append(mlr.horizon_err(Us, Y_t, free_t, G) * 1e3)
        a2.plot(xs, ys, color=SERIES[j], zorder=3)
        if xs:  # direct label -- discharges the palette's contrast WARN
            ends.append((xs[-1], ys[-1], j))

    # Stagger labels whose series end within `gap` of each other, so the direct
    # labels stay legible where curves converge (q1/q3 collide otherwise).
    span = max(y for _, y, _ in ends) - min(y for _, y, _ in ends)
    gap, last = 0.045 * span, -1e9
    for x_e, y_e, j in sorted(ends, key=lambda t: t[1]):
        y_lab = max(y_e, last + gap)
        last = y_lab
        a2.annotate(f"q{j + 1}", xy=(x_e, y_lab), xytext=(5, 0),
                    textcoords="offset points", color=SERIES[j], fontsize=8,
                    weight="bold", va="center")
    a2.set_xlabel("offset from the anchor along one joint (rad)")
    a2.set_ylabel("transfer error at horizon 12 (mm)")
    a2.set_title("B · Anchor distance barely matters", loc="left", color=INK)
    a2.set_xlim(0.15, 1.62)
    _grid(a2)

    fig.suptitle("Panda linear region — probe δ=0.05 in B, N=12, 50 Hz",
                 x=0.005, ha="left", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def _linearize(mode: str):
    m, d = load_model()
    if mode != "servo":
        m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[:] = 0.0
        m.actuator_gainprm[:, 0] = 1.0
        m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
        m.actuator_biasprm[:] = 0.0
        m.actuator_ctrlrange[:] = m.actuator_forcerange
    if mode == "gravcomp":
        m.opt.gravity[:] = 0.0
    fs, tip, nv = frame_skip(m), tip_id(m), m.nv
    q = np.asarray(m.key_qpos[0], np.float64).copy()
    q[0] = -0.6
    d.qpos[:] = q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    d.ctrl[:] = q if mode == "servo" else (0.0 if mode == "gravcomp" else d.qfrc_bias)
    mujoco.mj_forward(m, d)
    A1 = np.zeros((2 * nv, 2 * nv))
    B1 = np.zeros((2 * nv, m.nu))
    mujoco.mjd_transitionFD(m, d, 1e-7, 1, A1, B1, None, None)
    A = np.linalg.matrix_power(A1, fs)
    J = np.zeros((3, nv))
    mujoco.mj_jacSite(m, d, J, None, tip)
    return m, A, J


def fig_statespace(out: str) -> None:
    """Panel A-C: eigenvalues per actuation mode. Panel D: observability floor."""
    modes = [("servo", "PD servo (current)"), ("torque", "pure torque"),
             ("gravcomp", "grav-comp torque")]
    fig = plt.figure(figsize=(12, 3.8))
    sig = {}
    for i, (mode, label) in enumerate(modes):
        m, A, J = _linearize(mode)
        ax = fig.add_subplot(1, 4, i + 1)
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(th), np.sin(th), color=MUTED, lw=1.2, zorder=1)
        ev = np.linalg.eigvals(A)
        outside = np.abs(ev) > 1 + 1e-6
        ax.scatter(ev[~outside].real, ev[~outside].imag, s=42, c=SERIES[0],
                   edgecolors="#fcfcfb", linewidths=1.2, zorder=3)
        if outside.any():
            ax.scatter(ev[outside].real, ev[outside].imag, s=64, c=CRITICAL,
                       marker="X", edgecolors="#fcfcfb", linewidths=1.2, zorder=4)
        ax.set_title(f"{'ABC'[i]} · {label}\nmax |λ| = {np.abs(ev).max():.3f}   "
                     f"{outside.sum()} unstable", loc="left", fontsize=9,
                     color=CRITICAL if outside.any() else INK)
        ax.set_aspect("equal")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.axhline(0, color="#e8e7e3", lw=0.6, zorder=0)
        ax.axvline(0, color="#e8e7e3", lw=0.6, zorder=0)
        ax.set_xlabel("Re λ")
        if i == 0:
            ax.set_ylabel("Im λ")

        S = np.diag(2.0 / (m.jnt_range[:, 1] - m.jnt_range[:, 0]))
        Cy = np.hstack([J, np.zeros((3, 7))])
        Ce = np.vstack([Cy, np.hstack([S, np.zeros((7, 7))])])
        for name, C in [("tip", Cy), ("y_ext", Ce)]:
            obs = np.vstack([C @ np.linalg.matrix_power(A, kk) for kk in range(14)])
            sig[(mode, name)] = np.linalg.svd(obs, compute_uv=False)[-1]

    ax = fig.add_subplot(1, 4, 4)
    x = np.arange(3)
    for s, (name, col) in enumerate([("tip", SERIES[1]), ("y_ext", SERIES[2])]):
        vals = [max(sig[(mo, name)], 1e-22) for mo, _ in modes]
        ax.bar(x + (s - 0.5) * 0.38, vals, 0.34, color=col, zorder=3,
               label=f"y = {name}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["PD", "torque", "grav-comp"])
    ax.set_ylabel("σ_min of observability matrix")
    ax.set_title("D · Can the output see the state?", loc="left", fontsize=9, color=INK)
    ax.axhline(1e-8, color=CRITICAL, lw=1.2, ls="--", zorder=1)
    ax.text(2.48, 3e-8, "unobservable\nbelow", color=CRITICAL, fontsize=7,
            ha="right", va="bottom")
    # Below the axes: on a log bar chart every bar spans the full height, so any
    # in-axes legend placement collides with a mark.
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_ylim(1e-22, 1e-1)
    _grid(ax)

    fig.suptitle("Panda state-space at q₁=−0.6 — x=(q,q̇)∈ℝ¹⁴, 50 Hz discrete",
                 x=0.005, ha="left", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def fig_anchors(out: str, path: str) -> None:
    """A: PCA of the clustered configurations. B: scree. C: task space. D: §5."""
    from panda.anchors import _pairwise, pca, silhouette

    with np.load(path) as z:
        anchors, Q, P, labels = z["anchors"], z["Q"], z["P"], z["labels"]
        w = z["weights"] if z["weights"].size else None
    k = len(anchors)
    pc = pca(Q, w)
    sil = silhouette(_pairwise(Q, w), labels)
    ev = pc["explained"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    (a1, a2), (a3, a4) = ax

    # --- A: the clustering, drawn in the space it was actually computed in.
    S = pc["scores"]
    A = pc["project"](anchors)
    for j in range(k):
        sel = labels == j
        a1.scatter(S[sel, 0], S[sel, 1], s=13, c=SERIES[j], alpha=0.45,
                   linewidths=0, zorder=3, label=f"region {j}  (n={sel.sum()})")
    for j in range(k):  # medoids on top, ringed so they read against the cloud
        a1.scatter(A[j, 0], A[j, 1], s=150, c=SERIES[j], marker="o",
                   edgecolors="#fcfcfb", linewidths=2.2, zorder=5)
        a1.annotate(f"a{j}", xy=(A[j, 0], A[j, 1]), xytext=(0, 0),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=7, color="#fcfcfb", weight="bold", zorder=6)
    a1.set_xlabel(f"PC1  ({ev[0] * 100:.0f}% of variance)")
    a1.set_ylabel(f"PC2  ({ev[1] * 100:.0f}%)")
    a1.set_title(f"A · Clusters in configuration space (PCA)\n"
                 f"silhouette = {sil:.3f} in full 7-D", loc="left", color=INK)
    a1.legend(frameon=False, fontsize=8, loc="best")
    _grid(a1)

    # --- B: how much of the 7-D structure panel A can possibly show.
    x = np.arange(1, 8)
    a2.bar(x, ev, 0.62, color=SERIES[0], zorder=3, label="per component")
    a2.plot(x, np.cumsum(ev), color=SERIES[1], marker="o", ms=5, zorder=4,
            label="cumulative")
    a2.axhline(1.0, color=MUTED, lw=0.8, ls=":", zorder=1)
    a2.annotate(f"{np.cumsum(ev)[1] * 100:.0f}% in PC1+PC2\n(what panel A shows)",
                xy=(2, np.cumsum(ev)[1]), xytext=(2.35, np.cumsum(ev)[1] - 0.22),
                fontsize=8, color=SERIES[1],
                arrowprops=dict(arrowstyle="-", color=SERIES[1], lw=1))
    a2.set_xticks(x)
    a2.set_xlabel("principal component")
    a2.set_ylabel("fraction of variance")
    a2.set_ylim(0, 1.08)
    a2.set_title("B · Scree — how much panel A can show", loc="left", color=INK)
    a2.legend(frameon=False, fontsize=8, loc="center right")
    _grid(a2)

    # --- C: the same points in task space. Contrast with A is the finding.
    r = np.linalg.norm(P[:, :2], axis=1)
    for j in range(k):
        sel = labels == j
        a3.scatter(r[sel], P[sel, 2], s=13, c=SERIES[j], alpha=0.45,
                   linewidths=0, zorder=3)
    a3.set_xlabel("horizontal reach  ‖(x, y)‖  (m)")
    a3.set_ylabel("height z (m)")
    a3.set_title("C · The same points in task space — no separation",
                 loc="left", color=INK)
    _grid(a3)

    # --- D: plan section 5.
    xj = np.arange(7)
    spread = anchors.std(axis=0)
    a4.bar(xj, spread, 0.62, color=[SERIES[j] for j in range(7)], zorder=3)
    for j in range(7):
        a4.annotate(f"{spread[j]:.2f}", xy=(j, spread[j]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color=INK2)
    a4.set_xticks(xj)
    a4.set_xticklabels([f"q{j + 1}" for j in range(7)])
    a4.set_ylabel("std across anchors (rad)")
    v = anchors.var(axis=0)
    eff = v.sum() ** 2 / (v**2).sum()
    a4.set_title(f"D · Plan §5 — {eff:.2f}/7 effective joints, "
                 f"{100 * v[0] / v.sum():.0f}% in q₁", loc="left", color=INK)
    _grid(a4)

    fig.suptitle(f"Anchor selection, K={k} — k-medoids over task-relevant "
                 f"configurations (PCA in the clustering metric, unstandardized)",
                 x=0.005, ha="left", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figure", choices=["linear", "statespace", "anchors", "all"],
                   default="all")
    p.add_argument("--anchors", default="data/panda_anchors_k4.npz")
    p.add_argument("--out-dir", default="docs/reference")
    p.add_argument("--n-samples", type=int, default=150)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    d = args.out_dir
    if args.figure in ("statespace", "all"):
        fig_statespace(os.path.join(d, "panda_statespace.png"))
    if args.figure in ("anchors", "all"):
        fig_anchors(os.path.join(d, "panda_anchors.png"), args.anchors)
    if args.figure in ("linear", "all"):
        fig_linear(os.path.join(d, "panda_linear_region.png"), args.n_samples)


if __name__ == "__main__":
    main()

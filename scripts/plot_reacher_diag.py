"""Why the residual parks further out than vanilla, in four panels (journey 13).

Reads the per-step dump from `scripts/diag_reacher_residual.py`:

    uv run python scripts/diag_reacher_residual.py          # writes the npz
    uv run python scripts/plot_reacher_diag.py

A: commanded torque against distance-to-goal. The clone appears twice -- on its
   own trajectory, and queried at the residual's states -- in the SAME distance
   bins, so distance is controlled and the gap between those two curves is pure
   state-distribution shift: 3-8x more torque at the states the residual creates.
B: how often each arm commands effectively nothing in the terminal phase. On a
   damped planar arm with no gravity, resting IS how you hold a position. The
   clone can rest (16.1% on its own) but does not at the residual's states (0.8%).
C: the cancellation the residual would need (-u_base/rho) against what it emits.
   On the diagonal would be a perfect hold; the spread is the parking error.
D: the cost, paired per episode -- final distance, residual against vanilla.
   Doubles as a cross-check: these medians must reproduce journey 13's 400k
   table (2.4 mm against 1.7 mm). The clip rate, which the far/near split shows
   living entirely in the FAR field (54.7% against 0.4%), is printed by
   `diag_reacher_residual.py` rather than plotted.

Single seed, single checkpoint pair. Not evidence for a mechanism claim on its
own; see journey 13's record of retracted diagnoses.
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, CRITICAL = "#0b0b0b", "#52514e", "#b8b7b2", "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})

# Bin edges in mm. Dense below the 10 mm tolerance, because that is where the
# whole question lives; the far field only needs enough resolution to show trend.
EDGES = np.array([0, 1, 2, 3.5, 5, 7.5, 10, 20, 40, 80, 160, 400]) * 1e-3


def limp(x, thr=0.01):
    """Fraction of steps commanding effectively nothing (all components < thr)."""
    return float(np.mean(np.all(np.abs(x) < thr, axis=1)))


def binned(dist, u, edges=EDGES):
    """mean |u|^2 per distance bin, with bin centres; empty bins dropped."""
    idx = np.digitize(dist, edges) - 1
    xs, ys = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 20:
            xs.append(np.sqrt(edges[b] * max(edges[b + 1], 1e-9)) * 1e3
                      if edges[b] > 0 else edges[b + 1] * 0.5e3)
            ys.append(float(np.mean(np.sum(u[m] ** 2, axis=1))))
    return np.array(xs), np.array(ys)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", default="data/reacher_diag_steps.npz")
    p.add_argument("--out", default="data/reacher_residual_diag.png")
    args = p.parse_args()

    z = np.load(args.dump)
    tol, frac = float(z["tol"]), float(z["frac"])
    rd, vd = z["res_dist_pre"], z["van_dist_pre"]
    rn, vn = rd < tol, vd < tol
    ub, ua, ar = z["res_u_base"], z["res_u_applied"], z["res_a_res"]
    va = z["van_u_applied"]

    fig, ax = plt.subplots(1, 4, figsize=(15.4, 4.0))

    # --- A: torque vs distance to goal --------------------------------------
    # The clone appears TWICE and that is the point: the same frozen network,
    # queried on its own trajectory and queried at the residual's states. Same
    # distance bins, so distance is controlled for; what differs is the state
    # distribution, and the gap between the two green curves is the finding.
    a = ax[0]
    ca, cd = z["clo_u_applied"], z["clo_dist_pre"]
    series = ((va, vd, BLUE, "-", "o", "vanilla RL"),
              (ca, cd, GREEN, "-", "o", "clone, on its own trajectory"),
              (ub, rd, GREEN, ":", "s", "clone, at the RESIDUAL's states"),
              (ua, rd, ORANGE, "-", "o", "clone + residual (applied)"))
    for u, d, c, ls, mk, lbl in series:
        x, y = binned(d, u)
        a.plot(x, y, ls, marker=mk, color=c, ms=4, mfc="none" if ls == ":" else c,
               label=lbl)
    a.axvspan(EDGES[0] * 1e3, tol * 1e3, color="#e8e7e3", zorder=0)
    a.text(1.05, 0.92, "inside\ntolerance", transform=a.transData, fontsize=7.5,
           color=MUTED, va="top")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("distance to goal when the action was chosen (mm)")
    a.set_ylabel(r"commanded torque, mean $|u|^2$")
    a.set_title("A  the base gets loud where the residual goes",
                loc="left", color=INK)
    a.legend(frameon=False, fontsize=7.5, loc="lower right")
    a.grid(True, which="both", alpha=0.5)

    # --- B: resting fraction in the terminal phase --------------------------
    b = ax[1]
    cn = cd < tol
    vals = [limp(va[vn]), limp(ca[cn]), limp(ub[rn]), limp(ua[rn])]
    names = ["vanilla\nRL", "clone\nalone", "clone at\nres. states",
             "clone +\nresidual"]
    bars = b.bar(names, [v * 100 for v in vals],
                 color=[BLUE, GREEN, GREEN, ORANGE], width=0.62)
    bars[2].set_hatch("///")
    bars[2].set_edgecolor("#fcfcfb")
    for r, v in zip(bars, vals):
        b.text(r.get_x() + r.get_width() / 2, v * 100 + 2.5, f"{v:.1%}",
               ha="center", fontsize=9.5, color=INK)
    b.set_ylim(0, 100)
    b.set_ylabel("terminal steps commanding ~zero torque (%)")
    b.set_title("B  the base can rest -- just not there", loc="left", color=INK)
    b.grid(True, axis="y", alpha=0.5)

    # --- C: the cancellation it needs vs the one it emits --------------------
    c = ax[2]
    need = (-ub[rn] / frac).ravel()
    got = ar[rn].ravel()
    lim = float(np.percentile(np.abs(np.concatenate([need, got])), 99.5))
    c.plot([-lim, lim], [-lim, lim], "-", color=INK2, lw=1.0, zorder=3,
           label="perfect hold")
    c.scatter(need, got, s=3, alpha=0.12, color=ORANGE, linewidths=0, zorder=2)
    c.axhline(0, color=MUTED, lw=0.8)
    c.axvline(0, color=MUTED, lw=0.8)
    c.set_xlim(-lim, lim)
    c.set_ylim(-lim, lim)
    c.set_xlabel(r"$a_{res}$ that would command zero  ($-u_{base}/\rho$)")
    c.set_ylabel(r"$a_{res}$ the policy emitted")
    c.set_title("C  it over-cancels by ~40%", loc="left", color=INK)
    c.legend(frameon=False, fontsize=8, loc="upper left")
    rms = float(np.sqrt(np.mean((got - need) ** 2)))
    c.text(0.97, 0.04, f"RMS miss {rms:.3f}\nslope "
                       f"{np.polyfit(need, got, 1)[0]:.2f}", transform=c.transAxes,
           ha="right", fontsize=8, color=INK2)
    c.grid(True, alpha=0.5)

    # --- D: what it costs, paired per episode -------------------------------
    # The consequence panel. Same dump, so it doubles as a cross-check: these
    # medians must reproduce journey 13's 400k table (2.4 mm vs 1.7 mm final).
    d = ax[3]
    def per_episode(scen, dist):
        """Final distance per scenario -- the last step's post-step distance."""
        out = []
        for k in np.unique(scen):
            out.append(dist[scen == k][-1])
        return np.array(out)
    rf = per_episode(z["res_scen"], z["res_dist_post"]) * 1e3
    vf = per_episode(z["van_scen"], z["van_dist_post"]) * 1e3
    lo, hi = 0.15, max(rf.max(), vf.max()) * 1.3
    d.plot([lo, hi], [lo, hi], "-", color=INK2, lw=1.0, zorder=3, label="tie")
    d.axhline(tol * 1e3, color=CRITICAL, lw=0.9, ls=":", zorder=2)
    d.axvline(tol * 1e3, color=CRITICAL, lw=0.9, ls=":", zorder=2)
    d.text(lo * 1.1, tol * 1e3 * 0.82, "10 mm tolerance", fontsize=7.5,
           color=CRITICAL, va="top")
    d.scatter(vf, rf, s=16, alpha=0.6, color=ORANGE, linewidths=0, zorder=4)
    d.set_xscale("log")
    d.set_yscale("log")
    d.set_xlim(lo, hi)
    d.set_ylim(lo, hi)
    d.set_xlabel("vanilla RL, final distance (mm)")
    d.set_ylabel("clone + residual, final distance (mm)")
    d.set_title("D  what the fighting costs", loc="left", color=INK)
    worse = int((rf > vf).sum())
    d.text(0.97, 0.06, f"residual further out on {worse}/{len(rf)}\n"
                       f"median {np.median(rf):.1f} vs {np.median(vf):.1f} mm",
           transform=d.transAxes, ha="right", fontsize=8, color=INK2)
    d.legend(frameon=False, fontsize=8, loc="upper left")
    d.grid(True, which="both", alpha=0.5)

    fig.suptitle("Why the residual parks further from the goal than vanilla — "
                 f"120 scenarios, 400k checkpoints, $\\rho$ = {frac}",
                 fontsize=11.5, y=0.99, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    print(f"  A: terminal mean|u|^2 -- vanilla "
          f"{np.mean(np.sum(va[vn] ** 2, axis=1)):.3f}, base "
          f"{np.mean(np.sum(ub[rn] ** 2, axis=1)):.3f}, residual "
          f"{np.mean(np.sum(ua[rn] ** 2, axis=1)):.3f}")
    print(f"  B: resting -- vanilla {vals[0]:.1%}, base {vals[1]:.1%}, "
          f"residual {vals[2]:.1%}")
    print(f"  C: RMS miss {rms:.4f} on a target of median magnitude "
          f"{np.median(np.abs(need)):.4f}")


if __name__ == "__main__":
    main()

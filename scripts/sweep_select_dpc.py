"""Sweep Select-DPC's `n_max` (Algorithm 1 iterations) and plot reach rate.

`n_max` caps how many times the select -> solve -> re-select loop runs per control
step. Every Reacher run so far used `n_max = 3` and measured `iters/step ~ 2.97`,
so the loop was ALWAYS being cut off rather than converging -- meaning every
action measured came from an under-converged sequential linearization.

`n_max = 1` is the informative corner: it selects once and solves once, so it
isolates how much of the fixed-anchor -> Select-DPC gain comes from SELECTING the
data versus from ITERATING on the selection.

Cost is linear in `n_max` (one QP solve per iteration), so this is also the
accuracy/compute trade-off curve. Wall time per control step is reported for that
reason -- reach rate alone would hide it.

Not swept here: `N_cols` (`--n-cols`, the paper's other design parameter, held at
300). The paper sweeps that axis in its Figure 3.

    uv run python scripts/sweep_select_dpc.py
    uv run python scripts/sweep_select_dpc.py --n-max 1 2 3 5 8 --episodes 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reacher.deepc_setup import (  # noqa: E402
    anchor_grid, TIP_WEIGHT, collect_anchor, make_controller, y_ref_for,
)
from reacher.model import (  # noqa: E402
    NQ_ARM, fingertip, frame_skip, load_model, sample_config, sample_goal,
    set_state, step_torque,
)
from reacher.selectdpc import SelectDPC, trajectory_bank  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, CRITICAL = "#0b0b0b", "#52514e", "#b8b7b2", "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})


def run(model, data, ctrl, eps, args):
    """All episodes under one controller. Returns aggregate metrics."""
    fs = frame_skip(model)
    hits, fins, effs, its, t_solve, n_solve = 0, [], [], [], 0.0, 0
    for q0, goal in eps:
        set_state(model, data, q0, goal)
        t0 = fingertip(data)
        need = float(np.linalg.norm(goal - t0))
        ctrl.reset(np.concatenate([q0, t0]), u_initial=np.zeros(NQ_ARM))
        yref = y_ref_for(goal)
        best, path, prev = need, 0.0, t0.copy()
        for _ in range(args.steps):
            y = np.concatenate([np.asarray(data.qpos[:NQ_ARM]), fingertip(data)])
            t0_ = time.perf_counter()
            u = ctrl.act(y, yref)
            t_solve += time.perf_counter() - t0_
            n_solve += 1
            its.append(getattr(ctrl, "last_iters", 1))
            step_torque(model, data, u, fs)
            path += float(np.linalg.norm(fingertip(data) - prev))
            prev = fingertip(data)
            best = min(best, float(np.linalg.norm(fingertip(data) - goal)))
            if best < args.tol:
                break
        hits += best < args.tol
        fins.append(best)
        effs.append(path / (need - best) if need - best > 1e-4 else np.nan)
    return {"reached": hits, "n": len(eps), "final": float(np.median(fins)) * 1e3,
            "eff": float(np.nanmedian(effs)), "iters": float(np.mean(its)),
            "ms": 1e3 * t_solve / max(n_solve, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, nargs=2, default=[6, 5])
    p.add_argument("--T", type=int, default=1200)
    p.add_argument("--n-max", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    p.add_argument("--n-cols", type=int, default=300,
                   help="the paper's N_cols: columns selected per solve")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--T-ini", type=int, default=5)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--lambda-g", type=float, default=5e-3)
    p.add_argument("--out", default="docs/reference/reacher_nmax_sweep.png")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, data = load_model()
    rng = np.random.default_rng(args.seed)
    anchors = anchor_grid(model, *args.grid)
    print(f"collecting {len(anchors)} libraries ...")
    payload = {"anchors": anchors}
    for i, a in enumerate(anchors):
        rec = collect_anchor(model, data, a, args.T, rng)
        payload[f"u_{i}"], payload[f"q_{i}"], payload[f"tip_{i}"] = (
            rec["u"], rec["q"], rec["tip"])
    bank = trajectory_bank(payload, args.T_ini, args.N, stride=args.stride)
    print(f"  bank {bank['Up'].shape[1]} columns, N_cols={args.n_cols}")

    rng = np.random.default_rng(args.seed + 99)
    eps = [(sample_config(model, data, rng)[0], sample_goal(rng))
           for _ in range(args.episodes)]

    print(f"\n{'controller':>16}{'reached':>10}{'median final':>15}"
          f"{'path/net':>10}{'iters':>8}{'ms/step':>10}")
    fixed, _ = make_controller(payload, T_ini=args.T_ini, N=args.N,
                               lambda_g=args.lambda_g)
    base = run(model, data, fixed, eps, args)
    print(f"{'fixed anchors':>16}{base['reached']:>6}/{base['n']:<3}"
          f"{base['final']:>13.1f} mm{base['eff']:>10.1f}{base['iters']:>8.2f}"
          f"{base['ms']:>10.1f}", flush=True)

    rows = []
    for nm in args.n_max:
        ctrl = SelectDPC(
            bank, anchor_headings=np.zeros(1),
            Q=np.diag([0.0] * NQ_ARM + [TIP_WEIGHT] * 2), R=1e-3 * np.eye(NQ_ARM),
            T_ini=args.T_ini, N=args.N, lambda_g=args.lambda_g, lambda_y=7.5e3,
            u_bounds=(-np.ones(NQ_ARM), np.ones(NQ_ARM)), solver="SCS",
            n_cols=args.n_cols, n_max=nm)
        r = run(model, data, ctrl, eps, args)
        r["n_max"] = nm
        rows.append(r)
        print(f"{'Select n_max=' + str(nm):>16}{r['reached']:>6}/{r['n']:<3}"
              f"{r['final']:>13.1f} mm{r['eff']:>10.1f}{r['iters']:>8.2f}"
              f"{r['ms']:>10.1f}", flush=True)

    nm = [r["n_max"] for r in rows]
    rate = [100 * r["reached"] / r["n"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.0))

    ax[0].axhline(100 * base["reached"] / base["n"], color=SERIES[1], lw=1.6,
                  ls="--", zorder=2, label=f"fixed anchors ({base['reached']}/{base['n']})")
    ax[0].plot(nm, rate, color=SERIES[2], marker="o", ms=7, zorder=3,
               label="Select-DPC")
    for r, y in zip(rows, rate):
        ax[0].annotate(f"{r['reached']}/{r['n']}", xy=(r["n_max"], y), xytext=(0, 8),
                       textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax[0].set_xlabel("$n_{max}$ (Algorithm 1 iterations per control step)")
    ax[0].set_ylabel("reach rate (%)")
    ax[0].set_ylim(0, 105)
    ax[0].set_title("A · Does iterating help?", loc="left", color=INK)
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")

    ax[1].plot(nm, [r["iters"] for r in rows], color=SERIES[0], marker="o", ms=6,
               zorder=3, label="iterations actually used")
    ax[1].plot(nm, nm, color=MUTED, lw=1.2, ls=":", zorder=1, label="$n_{max}$ (cap)")
    ax[1].set_xlabel("$n_{max}$")
    ax[1].set_ylabel("mean iterations per step")
    ax[1].set_title("B · Is the cap binding?", loc="left", color=INK)
    ax[1].legend(frameon=False, fontsize=8)

    ax[2].plot([r["ms"] for r in rows], rate, color=SERIES[2], marker="o", ms=7,
               zorder=3)
    for r, y in zip(rows, rate):
        ax[2].annotate(f"$n_{{max}}$={r['n_max']}", xy=(r["ms"], y), xytext=(6, -3),
                       textcoords="offset points", fontsize=8, color=INK2)
    ax[2].scatter([base["ms"]], [100 * base["reached"] / base["n"]], s=70,
                  c=SERIES[1], marker="s", zorder=3)
    ax[2].annotate("fixed", xy=(base["ms"], 100 * base["reached"] / base["n"]),
                   xytext=(6, -3), textcoords="offset points", fontsize=8,
                   color=SERIES[1])
    ax[2].set_xlabel("solve time per control step (ms)")
    ax[2].set_ylabel("reach rate (%)")
    ax[2].set_ylim(0, 105)
    ax[2].set_title("C · Accuracy vs compute", loc="left", color=INK)

    for a in ax:
        a.grid(True, axis="y", zorder=0)
        a.set_axisbelow(True)
    fig.suptitle(f"Select-DPC $n_{{max}}$ sweep on Reacher-v5 — "
                 f"$N_{{cols}}$={args.n_cols}, {args.episodes} episodes, "
                 f"{args.steps} steps", x=0.005, ha="left", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

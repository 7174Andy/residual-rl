"""Reacher controller evaluation at scale: 100+ paired scenarios, uncensored.

Fixes two limitations of the `n_max` sweep at once.

**n = 20 was too small.** 16/20 vs 13/20 is three episodes; nothing in that sweep
distinguished `n_max` 1, 2 and 3. Wilson intervals are reported so the reader can
see when two rows genuinely differ rather than inferring it from point estimates.

**Median final distance was CENSORED.** Every previous run stopped the moment
`best < tol`, so a reached episode's "final" is wherever the tip happened to be on
the step it crossed 10 mm -- step granularity, not precision. With ~80% of
episodes reaching, the median was drawn entirely from that just-under-threshold
band, which is why every setting scored 9.0-9.8 mm regardless of controller.
`--no-early-stop` (the default here) runs every episode to `--steps` and reports
the CONVERGED distance, which measures precision.

Three distances are therefore reported, and they answer different questions:

    best     closest approach at any point   -> the reach criterion
    final    distance at the last step       -> does it CONVERGE and hold?
    steps    first step under tol            -> how fast

`best` and `final` diverging means the controller passes through the target and
drifts off -- invisible to a reach-rate metric, and a real failure mode for a
predictive controller with no terminal cost.

    uv run python scripts/eval_reacher_scenarios.py --episodes 120
    uv run python scripts/eval_reacher_scenarios.py --episodes 200 --early-stop
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
from rl.stats import wilson_ci  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, CRITICAL = "#0b0b0b", "#52514e", "#b8b7b2", "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})


def episode(model, data, q0, goal, args, ctrl=None, rand=False):
    fs = frame_skip(model)
    set_state(model, data, q0, goal)
    t0 = fingertip(data)
    need = float(np.linalg.norm(goal - t0))
    if ctrl is not None:
        ctrl.reset(np.concatenate([q0, t0]), u_initial=np.zeros(NQ_ARM))
        yref = y_ref_for(goal)
    rw = np.random.default_rng(int(abs(q0[0]) * 1e6) % 2**31)
    best, path, prev, hit_step = need, 0.0, t0.copy(), None
    for t in range(args.steps):
        if rand:
            u = rw.uniform(-1, 1, NQ_ARM)
        else:
            y = np.concatenate([np.asarray(data.qpos[:NQ_ARM]), fingertip(data)])
            try:
                u = ctrl.act(y, yref)
            except RuntimeError:
                break
        step_torque(model, data, u, fs)
        path += float(np.linalg.norm(fingertip(data) - prev))
        prev = fingertip(data)
        dist = float(np.linalg.norm(fingertip(data) - goal))
        best = min(best, dist)
        if hit_step is None and dist < args.tol:
            hit_step = t + 1
            if args.early_stop:
                break
    return {"reached": best < args.tol, "need": need, "best": best,
            "final": dist, "steps": hit_step,
            "eff": path / (need - best) if need - best > 1e-4 else np.nan}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, nargs=2, default=[6, 5])
    p.add_argument("--T", type=int, default=1200)
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--n-cols", type=int, default=300)
    p.add_argument("--n-max", type=int, nargs="+", default=[1, 3])
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--early-stop", action="store_true",
                   help="stop at first reach (censors the final-distance metric)")
    p.add_argument("--T-ini", type=int, default=5)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--lambda-g", type=float, default=5e-3)
    p.add_argument("--out", default="docs/reference/reacher_scenarios.png")
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

    rng = np.random.default_rng(args.seed + 99)
    eps = [(sample_config(model, data, rng)[0], sample_goal(rng))
           for _ in range(args.episodes)]
    print(f"  {len(eps)} scenarios, {args.steps} steps, "
          f"early_stop={args.early_stop}, tol={args.tol * 1e3:.0f} mm")

    fixed, _ = make_controller(payload, T_ini=args.T_ini, N=args.N,
                               lambda_g=args.lambda_g)
    ctrls = [(f"{len(anchors)} fixed", dict(ctrl=fixed))]
    for nm in args.n_max:
        ctrls.append((f"Select n_max={nm}", dict(ctrl=SelectDPC(
            bank, anchor_headings=np.zeros(1),
            Q=np.diag([0.0] * NQ_ARM + [TIP_WEIGHT] * 2), R=1e-3 * np.eye(NQ_ARM),
            T_ini=args.T_ini, N=args.N, lambda_g=args.lambda_g, lambda_y=7.5e3,
            u_bounds=(-np.ones(NQ_ARM), np.ones(NQ_ARM)), solver="SCS",
            n_cols=args.n_cols, n_max=nm))))
    ctrls.append(("random", dict(rand=True)))

    print(f"\n  {'controller':<16}{'reach rate (95% CI)':>24}{'best':>10}"
          f"{'final':>10}{'steps':>8}{'path/net':>10}{'min':>7}")
    out = {}
    for label, kw in ctrls:
        t0 = time.perf_counter()
        r = [episode(model, data, q0, g, args, **kw) for q0, g in eps]
        out[label] = r
        k = sum(x["reached"] for x in r)
        lo, hi = wilson_ci(k, len(r))
        st = [x["steps"] for x in r if x["steps"] is not None]
        print(f"  {label:<16}{k:>4}/{len(r):<4}"
              f"[{lo * 100:>4.0f}-{hi * 100:>3.0f}%]"
              f"{np.median([x['best'] for x in r]) * 1e3:>9.1f}mm"
              f"{np.median([x['final'] for x in r]) * 1e3:>9.1f}mm"
              f"{np.median(st) if st else float('nan'):>8.0f}"
              f"{np.nanmedian([x['eff'] for x in r]):>10.1f}"
              f"{(time.perf_counter() - t0) / 60:>6.1f}m", flush=True)

    # Paired sign tests against the fixed baseline.
    base = out[f"{len(anchors)} fixed"]
    print("\n  paired vs fixed anchors (best-distance, per scenario):")
    for label, r in out.items():
        if label.startswith(f"{len(anchors)} fixed"):
            continue
        d = np.array([a["best"] - b["best"] for a, b in zip(r, base)])
        print(f"    {label:<16} closer on {int((d < -1e-5).sum()):>3}/{len(d)}, "
              f"median gain {-np.median(d) * 1e3:+6.1f} mm")

    labels = [c[0] for c in ctrls]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
    ks = [sum(x["reached"] for x in out[lb]) for lb in labels]
    cis = [wilson_ci(k, len(eps)) for k in ks]
    y = np.arange(len(labels))
    ax[0].barh(y, [100 * k / len(eps) for k in ks], 0.6,
               color=[SERIES[1]] + [SERIES[2]] * len(args.n_max) + [SERIES[3]],
               zorder=3)
    for i, (k, (lo, hi)) in enumerate(zip(ks, cis)):
        ax[0].plot([lo * 100, hi * 100], [i, i], color=INK, lw=1.6, zorder=4)
        ax[0].annotate(f"{k}/{len(eps)}", xy=(hi * 100, i), xytext=(5, 0),
                       textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labels, fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlim(0, 118)
    ax[0].set_xlabel("reach rate (%) with Wilson 95% CI")
    ax[0].set_title(f"A · {len(eps)} scenarios", loc="left", color=INK)

    for i, lb in enumerate(labels):
        v = np.sort([x["final"] for x in out[lb]]) * 1e3
        ax[1].plot(v, np.linspace(0, 100, len(v)), zorder=3, label=lb,
                   color=([SERIES[1]] + [SERIES[2], SERIES[0]] + [SERIES[3]])[i])
    ax[1].axvline(args.tol * 1e3, color=CRITICAL, lw=1.2, ls="--", zorder=1)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("converged distance at last step (mm)")
    ax[1].set_ylabel("% of scenarios below")
    ax[1].set_title("B · Uncensored final distance", loc="left", color=INK)
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")

    for i, lb in enumerate(labels):
        b = np.array([x["best"] for x in out[lb]]) * 1e3
        f = np.array([x["final"] for x in out[lb]]) * 1e3
        ax[2].scatter(b, f, s=22, alpha=0.6, linewidths=0, zorder=3, label=lb,
                      color=([SERIES[1]] + [SERIES[2], SERIES[0]] + [SERIES[3]])[i])
    lim = 400
    ax[2].plot([0.5, lim], [0.5, lim], color=MUTED, lw=1.2, zorder=1)
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("best distance reached (mm)")
    ax[2].set_ylabel("distance at last step (mm)")
    ax[2].set_title("C · Above the line = drifts off after arriving",
                    loc="left", color=INK)
    ax[2].legend(frameon=False, fontsize=8, loc="upper left")

    for a in ax:
        a.grid(True, zorder=0)
        a.set_axisbelow(True)
    fig.suptitle(f"Reacher controller evaluation — {len(eps)} scenarios, "
                 f"N_cols={args.n_cols}, early_stop={args.early_stop}",
                 x=0.005, ha="left", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

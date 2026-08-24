"""Sweep K over nested farthest-point anchors: coverage, prediction, reaching.

The experiment the anchor plan's sections 8-11 actually ask for, with K chosen by
coverage rather than by clustering quality (silhouette is flat at 0.23-0.28 for
every K, so there is no natural K to find).

Exploits the fact that farthest-point sampling is NESTED: the first K anchors are
a valid K-anchor set for every K, so ONE collection of `max(K)` libraries serves
every row of the sweep. Collecting each K separately would cost 8+16+65 = 89
libraries instead of 65, and the rows would not share data.

Three measurements per K:

  * `r_K`  -- worst-case nearest-anchor distance over the task configurations.
  * PREDICTION -- held-out task configurations, each routed to its nearest anchor
    exactly as the controller would route it, then scored with the same
    skill/cos/RMSE metrics as `scripts/verify_libraries.py`. This is the honest
    version of "prediction error at K": as K grows, configurations sit closer to
    their anchor, so the error should fall.
  * REACHING -- closed-loop episodes with the in-QP rate limit.

    uv run python scripts/sweep_anchor_k.py --K 8 16 65
    uv run python scripts/sweep_anchor_k.py --K 8 16 --episodes 12 --reuse
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from panda.anchors import assign, coverage  # noqa: E402
from panda.model import (  # noqa: E402
    MIN_TIP_Z, frame_skip, load_model, safe_box, sample_config, tip_id,
)
from panda.qdes import (  # noqa: E402
    build_libraries, collect_anchor, make_controller, outputs, predict, step_qdes,
    y_ref_for,
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


def collect_all(model, data, anchors, T, rng, path):
    """Collect one library per anchor, caching to `path` (65 of these is ~45 min)."""
    if os.path.exists(path):
        with np.load(path) as z:
            if int(z["anchors"].shape[0]) >= len(anchors):
                print(f"  reusing {path}")
                return {k: z[k] for k in z.files}
    payload = {"anchors": anchors}
    for i, a in enumerate(anchors):
        rec = collect_anchor(model, data, a, T, rng)
        payload[f"u_{i}"] = rec["u"]
        payload[f"q_{i}"] = rec["q"]
        payload[f"tip_{i}"] = rec["tip"]
        if (i + 1) % 10 == 0 or i + 1 == len(anchors):
            print(f"  collected {i + 1}/{len(anchors)}", flush=True)
    np.savez(path, **payload)
    return payload


def prediction_at_k(model, data, payload, libs, anchors, rng, args, w):
    """Route held-out task configurations to their nearest anchor, then score.

    Routing matters: scoring every configuration against a FIXED radius would
    measure the library, but the controller uses whichever anchor is nearest, so
    the quantity that predicts closed-loop behaviour is the error after routing.
    """
    lo, hi = safe_box(model)
    tip = tip_id(model)
    T_ini, N, nq = args.T_ini, args.N, model.nq
    rmse, skill, cos, dists = [], [], [], []
    for _ in range(args.pred_n):
        q0, _ = sample_config(model, data, rng, lo, hi, tip)
        i = assign(q0, anchors, w)
        dists.append(float(np.linalg.norm(q0 - anchors[i])))
        rec = collect_anchor(model, data, q0, T_ini + N + 1, rng, sigma=args.sigma)
        y = outputs(rec["q"], rec["tip"])
        yh = predict(libs[i], rec["u"][:T_ini], y[:T_ini],
                     rec["u"][T_ini:T_ini + N], args.lambda_g, N, y.shape[1])
        yt = y[T_ini:T_ini + N]
        tp, tt = yh[:, nq:], yt[:, nq:]
        rmse.append(np.sqrt(np.mean(np.sum((tp - tt) ** 2, axis=1))))
        nil = np.tile(y[T_ini - 1, nq:], (N, 1))
        mse_l = np.mean(np.sum((tp - tt) ** 2, axis=1))
        mse_0 = np.mean(np.sum((nil - tt) ** 2, axis=1))
        skill.append(1.0 - mse_l / max(mse_0, 1e-15))
        dp, dt = tp[-1] - y[T_ini - 1, nq:], tt[-1] - y[T_ini - 1, nq:]
        den = np.linalg.norm(dp) * np.linalg.norm(dt)
        cos.append(float(dp @ dt / den) if den > 1e-12 else 0.0)
    return (float(np.median(rmse)), float(np.median(skill)),
            float(np.median(cos)), float(np.median(dists)))


def _goal_near(model, data, q0, s, rng, lo, hi, tip, tries=40):
    for _ in range(tries):
        d = rng.standard_normal(model.nq)
        qg = np.clip(q0 + s * d / np.linalg.norm(d), lo, hi)
        data.qpos[:] = qg
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if data.ncon == 0 and data.site_xpos[tip][2] > MIN_TIP_Z:
            return data.site_xpos[tip].copy()
    return None


def reaching_at_k(model, data, payload, libs, anchors, rng, args, w):
    """Closed-loop reaching from random task starts. Returns the section-9 metrics."""
    lo, hi = safe_box(model)
    tip, fs = tip_id(model), frame_skip(model)
    deepc, info = make_controller(
        {"anchors": anchors, **{k: payload[k] for k in payload if k != "anchors"}},
        model, T_ini=args.T_ini, N=args.N, du_max=args.du_max, weights=w,
    )
    deepc._libraries = libs
    hits, closed, effs, switches = 0, [], [], []
    for _ in range(args.episodes):
        q0, _ = sample_config(model, data, rng, lo, hi, tip)
        goal = _goal_near(model, data, q0, args.goal_dist, rng, lo, hi, tip)
        if goal is None:
            continue
        data.qpos[:] = q0
        data.qvel[:] = 0.0
        data.ctrl[:] = q0
        mujoco.mj_forward(model, data)
        t0 = data.site_xpos[tip].copy()
        need = float(np.linalg.norm(goal - t0))
        deepc.reset(np.concatenate([q0, t0]), u_initial=q0)
        yref = y_ref_for(goal, model.nq)
        best, path, prev, prev_i, sw = need, 0.0, t0.copy(), -1, 0
        for _t in range(args.steps):
            y = np.concatenate([np.asarray(data.qpos), np.asarray(data.site_xpos[tip])])
            try:
                u = deepc.act(y, yref)
            except RuntimeError:
                break
            if deepc.last_library_idx != prev_i and prev_i >= 0:
                sw += 1
            prev_i = deepc.last_library_idx
            step_qdes(model, data, u, lo, hi, fs)
            path += float(np.linalg.norm(data.site_xpos[tip] - prev))
            prev = data.site_xpos[tip].copy()
            best = min(best, float(np.linalg.norm(data.site_xpos[tip] - goal)))
            if best < args.tol:
                break
        net = need - best
        hits += int(best < args.tol)
        closed.append(100.0 * net / need)
        effs.append(path / max(net, 1e-6) if net > 1e-4 else float("nan"))
        switches.append(sw)
    return (hits / max(len(closed), 1), float(np.median(closed)),
            float(np.nanmedian(effs)), float(np.mean(switches)), len(closed))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fps", default="data/panda_anchors_k4_ik_fps.npz")
    p.add_argument("--K", type=int, nargs="+", default=[8, 16, 65])
    p.add_argument("--T", type=int, default=1500)
    p.add_argument("--pred-n", type=int, default=30, help="held-out configs per K")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--goal-dist", type=float, default=0.4)
    p.add_argument("--du-max", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--tol", type=float, default=0.02)
    p.add_argument("--T-ini", type=int, default=5)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--lambda-g", type=float, default=5e-3)
    p.add_argument("--cache", default="data/panda_fps_libs.npz")
    p.add_argument("--out", default="docs/reference/panda_k_sweep.png")
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()

    model, data = load_model()
    with np.load(args.fps) as z:
        Q, fps_anchors = z["Q"], z["fps_anchors"]
        w = z["weights"] if z["weights"].size else None
    k_max = max(args.K)
    if k_max > len(fps_anchors):
        raise SystemExit(f"--fps has only {len(fps_anchors)} anchors; need {k_max}. "
                         f"Re-run anchor_coverage.py --k-max {k_max}")

    print(f"collecting {k_max} nested FPS libraries (T={args.T}) ...")
    rng = np.random.default_rng(args.seed)
    payload = collect_all(model, data, fps_anchors[:k_max], args.T, rng, args.cache)
    print("building Hankels ...", flush=True)
    all_libs = build_libraries({"anchors": fps_anchors[:k_max], **payload},
                               args.T_ini, args.N)

    rows = []
    for K in args.K:
        anchors = fps_anchors[:K]
        libs = all_libs[:K]
        sub = {f"{k}_{i}": payload[f"{k}_{i}"] for i in range(K) for k in ("u", "q", "tip")}
        cov = coverage(Q, anchors, w)
        rng_k = np.random.default_rng(args.seed + 1000)
        rmse, skill, cos, mdist = prediction_at_k(
            model, data, sub, libs, anchors, rng_k, args, w)
        rng_k = np.random.default_rng(args.seed + 2000)
        rate, cl, eff, sw, n = reaching_at_k(
            model, data, {"anchors": anchors, **sub}, libs, anchors, rng_k, args, w)
        rows.append(dict(K=K, r=cov["r"], mean_nn=cov["mean_nn"], rmse=rmse,
                         skill=skill, cos=cos, mdist=mdist, rate=rate,
                         closed=cl, eff=eff, sw=sw, n=n))
        print(f"\nK={K:>3}  r_K={cov['r']:.2f}  median routed dist={mdist:.2f}  "
              f"tip RMSE={rmse * 1e3:.0f} mm  skill={skill:.2f}  cos={cos:.2f}  "
              f"reach={rate * 100:.0f}% ({n} eps)  closed={cl:.0f}%  "
              f"path/net={eff:.1f}  switches/ep={sw:.1f}", flush=True)

    print(f"\n  {'K':>4}{'r_K':>7}{'routed d':>10}{'tip RMSE':>11}{'skill':>8}"
          f"{'cos':>7}{'reach':>8}{'closed':>9}{'path/net':>10}")
    for r in rows:
        print(f"  {r['K']:>4}{r['r']:>7.2f}{r['mdist']:>10.2f}"
              f"{r['rmse'] * 1e3:>9.0f} mm{r['skill']:>8.2f}{r['cos']:>7.2f}"
              f"{r['rate'] * 100:>7.0f}%{r['closed']:>8.0f}%{r['eff']:>10.1f}")

    K = [r["K"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
    ax[0].plot(K, [r["r"] for r in rows], color=SERIES[0], marker="o", ms=6, zorder=3,
               label="worst-case $r_K$")
    ax[0].plot(K, [r["mdist"] for r in rows], color=SERIES[2], marker="o", ms=6,
               zorder=3, label="median routed distance")
    ax[0].axhline(1.0, color=CRITICAL, lw=1.2, ls="--", zorder=1)
    ax[0].text(K[-1], 1.05, "usable radius", ha="right", color=CRITICAL, fontsize=8)
    ax[0].set_xscale("log")
    ax[0].set_xticks(K)
    ax[0].set_xticklabels([str(k) for k in K])
    ax[0].set_xlabel("anchors K")
    ax[0].set_ylabel("distance to nearest anchor (rad)")
    ax[0].set_title("A · Coverage", loc="left", color=INK)
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].plot(K, [r["rmse"] * 1e3 for r in rows], color=SERIES[1], marker="o", ms=6,
               zorder=3)
    ax[1].axhline(args.tol * 1e3, color=CRITICAL, lw=1.2, ls="--", zorder=1)
    ax[1].text(K[-1], args.tol * 1e3 * 1.1, f"reach tol ({args.tol * 1e3:.0f} mm)",
               ha="right", color=CRITICAL, fontsize=8)
    ax[1].set_xscale("log")
    ax[1].set_xticks(K)
    ax[1].set_xticklabels([str(k) for k in K])
    ax[1].set_xlabel("anchors K")
    ax[1].set_ylabel("tip prediction RMSE (mm)")
    ax[1].set_title("B · Prediction, after routing", loc="left", color=INK)

    ax[2].plot(K, [r["rate"] * 100 for r in rows], color=SERIES[3], marker="o", ms=6,
               zorder=3)
    for r in rows:
        ax[2].annotate(f"{r['rate'] * 100:.0f}%", xy=(r["K"], r["rate"] * 100),
                       xytext=(0, 8), textcoords="offset points", ha="center",
                       fontsize=8, color=INK2)
    ax[2].set_xscale("log")
    ax[2].set_xticks(K)
    ax[2].set_xticklabels([str(k) for k in K])
    ax[2].set_ylim(-5, 105)
    ax[2].set_xlabel("anchors K")
    ax[2].set_ylabel("reach success (%)")
    ax[2].set_title(f"C · Closed loop ({args.episodes} eps, "
                    f"{args.goal_dist} rad goals)", loc="left", color=INK)
    for a in ax:
        a.grid(True, axis="y", zorder=0)
        a.set_axisbelow(True)

    fig.suptitle(f"Anchor count sweep — nested FPS anchors, du_max={args.du_max}, "
                 f"T={args.T}", x=0.005, ha="left", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

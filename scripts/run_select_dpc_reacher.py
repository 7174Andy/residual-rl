"""Select-DPC vs fixed anchors on Reacher: paired metrics, figure, and videos.

Runs both controllers on the IDENTICAL 20 episodes, so every comparison is
within-pair. Then records one succeeded and one failed Select-DPC episode, plus
the fixed-anchor controller on the same failed episode for contrast.

Select-DPC is `reacher/selectdpc.py`, faithful to Algorithm 1 + 2 of
arXiv:2503.18845: selection is against the open-loop PREDICTION over the full
length-L trajectory, and it iterates until convergence or `n_max`.

    uv run python scripts/run_select_dpc_reacher.py
    uv run python scripts/run_select_dpc_reacher.py --n-sel 600 --n-max 5
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

from core.video_encoding import encode_video  # noqa: E402
from reacher.deepc_setup import (  # noqa: E402
    anchor_grid, TIP_WEIGHT, collect_anchor, make_controller, y_ref_for,
)
from reacher.model import (  # noqa: E402
    NQ_ARM, fingertip, frame_skip, model_path, sample_config,
    sample_goal, set_state, step_torque,
)
from reacher.selectdpc import SelectDPC, trajectory_bank  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record_reacher_video import _annotate, _ring  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, CRITICAL, GOOD = "#0b0b0b", "#52514e", "#b8b7b2", "#d03b3b", "#0ca30c"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})


def episode(model, data, q0, goal, args, ctrl, renderer=None, cam=None, label=""):
    """One episode. Returns metrics, and frames when a renderer is supplied."""
    fs = frame_skip(model)
    set_state(model, data, q0, goal)
    t0 = fingertip(data)
    need = float(np.linalg.norm(goal - t0))
    ctrl.reset(np.concatenate([q0, t0]), u_initial=np.zeros(NQ_ARM))
    yref = y_ref_for(goal)
    best, path, prev, frames, its = need, 0.0, t0.copy(), [], []
    for t in range(args.steps):
        if renderer is not None:
            renderer.update_scene(data, camera=cam)
            _ring(renderer.scene, goal, args.tol, [0.2, 0.9, 0.2, 0.30])
            cur = float(np.linalg.norm(fingertip(data) - goal))
            frames.append(_annotate(renderer.render(), t, cur, args.tol, label,
                                    cur < args.tol))
        y = np.concatenate([np.asarray(data.qpos[:NQ_ARM]), fingertip(data)])
        try:
            u = ctrl.act(y, yref)
        except RuntimeError:
            break
        its.append(getattr(ctrl, "last_iters", 1))
        step_torque(model, data, u, fs)
        path += float(np.linalg.norm(fingertip(data) - prev))
        prev = fingertip(data)
        best = min(best, float(np.linalg.norm(fingertip(data) - goal)))
        if best < args.tol:
            for _ in range(args.hold if renderer is not None else 0):
                renderer.update_scene(data, camera=cam)
                _ring(renderer.scene, goal, args.tol, [0.2, 0.9, 0.2, 0.55])
                frames.append(_annotate(renderer.render(), t + 1, best, args.tol,
                                        label, True))
            break
    net = need - best
    return {"reached": best < args.tol, "need": need, "final": best,
            "closed": 100.0 * net / need, "steps": t + 1,
            "eff": path / net if net > 1e-5 else float("nan"),
            "iters": float(np.mean(its)) if its else 1.0}, frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, nargs=2, default=[6, 5])
    p.add_argument("--T", type=int, default=1200)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--n-sel", type=int, default=300)
    p.add_argument("--n-max", type=int, default=3)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--hold", type=int, default=10)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--size", type=int, default=720)
    p.add_argument("--T-ini", type=int, default=5)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--lambda-g", type=float, default=5e-3)
    p.add_argument("--out-dir", default="videos/reacher_select")
    p.add_argument("--fig", default="docs/reference/reacher_select_dpc.png")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Larger offscreen buffer: reacher.xml caps it at 640, and a 10 mm tolerance
    # on a 210 mm workspace is illegible below ~700 px.
    xml = open(model_path()).read().replace(
        "<worldbody>",
        f"<visual><global offwidth='{args.size}' offheight='{args.size}'/></visual>"
        "\n<worldbody>", 1)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = 0.5
    model.vis.headlight.diffuse[:] = 0.7
    renderer = mujoco.Renderer(model, height=args.size, width=args.size)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.0]
    cam.distance, cam.elevation, cam.azimuth = 0.62, -90.0, 90.0

    rng = np.random.default_rng(args.seed)
    anchors = anchor_grid(model, *args.grid)
    print(f"collecting {len(anchors)} libraries ...")
    payload = {"anchors": anchors}
    for i, a in enumerate(anchors):
        rec = collect_anchor(model, data, a, args.T, rng)
        payload[f"u_{i}"], payload[f"q_{i}"], payload[f"tip_{i}"] = (
            rec["u"], rec["q"], rec["tip"])

    bank = trajectory_bank(payload, args.T_ini, args.N, stride=args.stride)
    print(f"  pooled bank: {bank['Up'].shape[1]} columns, tau dim {bank['tau'].shape[0]}")
    fixed, _ = make_controller(payload, T_ini=args.T_ini, N=args.N,
                               lambda_g=args.lambda_g)
    select = SelectDPC(
        bank, anchor_headings=np.zeros(1),
        Q=np.diag([0.0] * NQ_ARM + [TIP_WEIGHT] * 2), R=1e-3 * np.eye(NQ_ARM),
        T_ini=args.T_ini, N=args.N, lambda_g=args.lambda_g, lambda_y=7.5e3,
        u_bounds=(-np.ones(NQ_ARM), np.ones(NQ_ARM)), solver="SCS",
        n_cols=args.n_sel, n_max=args.n_max)

    rng = np.random.default_rng(args.seed + 99)
    eps = [(sample_config(model, data, rng)[0], sample_goal(rng))
           for _ in range(args.episodes)]

    print(f"\n{'ep':>3}{'need':>8}{'fixed':>12}{'Select-DPC':>14}   winner")
    res = {"fixed": [], "select": []}
    for i, (q0, goal) in enumerate(eps):
        rf, _ = episode(model, data, q0, goal, args, fixed)
        rs, _ = episode(model, data, q0, goal, args, select)
        res["fixed"].append(rf)
        res["select"].append(rs)
        win = ("Select" if rs["final"] < rf["final"] - 1e-4
               else "fixed" if rf["final"] < rs["final"] - 1e-4 else "tie")
        print(f"{i:>3}{rf['need'] * 1e3:>7.0f}mm"
              f"{rf['final'] * 1e3:>9.1f}mm{'*' if rf['reached'] else ' '}"
              f"{rs['final'] * 1e3:>11.1f}mm{'*' if rs['reached'] else ' '}   {win}")

    print(f"\n  {'':<14}{'reached':>9}{'median final':>15}{'path/net':>10}{'iters':>8}")
    for k, lab in (("fixed", f"{len(anchors)} fixed anchors"), ("select", "Select-DPC")):
        r = res[k]
        print(f"  {lab:<14}{sum(x['reached'] for x in r):>4}/{len(r):<4}"
              f"{np.median([x['final'] for x in r]) * 1e3:>12.1f} mm"
              f"{np.nanmedian([x['eff'] for x in r]):>10.1f}"
              f"{np.mean([x['iters'] for x in r]):>8.2f}")
    wins = sum(s["final"] < f["final"] - 1e-4
               for s, f in zip(res["select"], res["fixed"]))
    print(f"  Select-DPC closer on {wins}/{len(eps)} episodes (paired)")

    # --- videos: one Select-DPC success, one failure, + fixed on the same failure
    os.makedirs(args.out_dir, exist_ok=True)
    succ = [i for i, r in enumerate(res["select"]) if r["reached"]]
    fail = [i for i, r in enumerate(res["select"]) if not r["reached"]]
    jobs = []
    if succ:
        jobs.append((min(succ, key=lambda i: res["select"][i]["final"]),
                     "success", select, "Select-DPC"))
    if fail:
        w = max(fail, key=lambda i: res["select"][i]["final"])
        jobs.append((w, "failure", select, "Select-DPC"))
        jobs.append((w, "failure_fixed", fixed, f"{len(anchors)} fixed anchors"))
    print()
    for i, name, ctrl, lab in jobs:
        q0, goal = eps[i]
        r, frames = episode(model, data, q0, goal, args, ctrl, renderer, cam, lab)
        path = os.path.join(args.out_dir, f"{name}.mp4")
        encode_video(frames, path, args.fps)
        print(f"  {name:>14} (ep{i}): {'REACH' if r['reached'] else ' miss'} "
              f"final {r['final'] * 1e3:6.1f} mm  {len(frames)} frames -> {path}")

    # --- figure
    fx = np.array([r["final"] for r in res["fixed"]]) * 1e3
    sx = np.array([r["final"] for r in res["select"]]) * 1e3
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.0))
    order = np.argsort(fx)
    x = np.arange(len(fx))
    ax[0].bar(x - 0.2, fx[order], 0.4, color=SERIES[1], zorder=3, label="fixed anchors")
    ax[0].bar(x + 0.2, sx[order], 0.4, color=SERIES[2], zorder=3, label="Select-DPC")
    ax[0].axhline(args.tol * 1e3, color=CRITICAL, lw=1.2, ls="--", zorder=4)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("episode (sorted by fixed-anchor result)")
    ax[0].set_ylabel("final distance (mm)")
    ax[0].set_title("A · Paired, per episode", loc="left", color=INK)
    ax[0].legend(frameon=False, fontsize=8)

    lim = max(fx.max(), sx.max()) * 1.3
    ax[1].scatter(fx, sx, s=42, c=[GOOD if s < f else CRITICAL for f, s in zip(fx, sx)],
                  edgecolors="#fcfcfb", linewidths=1.2, zorder=3)
    ax[1].plot([0.5, lim], [0.5, lim], color=MUTED, lw=1.2, zorder=1)
    ax[1].axhline(args.tol * 1e3, color=CRITICAL, lw=1, ls="--", zorder=1)
    ax[1].axvline(args.tol * 1e3, color=CRITICAL, lw=1, ls="--", zorder=1)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(0.5, lim)
    ax[1].set_ylim(0.5, lim)
    ax[1].set_xlabel("fixed anchors, final (mm)")
    ax[1].set_ylabel("Select-DPC, final (mm)")
    ax[1].set_title("B · Below the line = Select-DPC better", loc="left", color=INK)

    labs = [f"{len(anchors)} fixed\nanchors", "Select-DPC"]
    vals = [sum(r["reached"] for r in res["fixed"]),
            sum(r["reached"] for r in res["select"])]
    ax[2].bar([0, 1], [100 * v / len(eps) for v in vals], 0.55,
              color=[SERIES[1], SERIES[2]], zorder=3)
    for i, v in enumerate(vals):
        ax[2].annotate(f"{v}/{len(eps)}", xy=(i, 100 * v / len(eps)), xytext=(0, 5),
                       textcoords="offset points", ha="center", fontsize=9, color=INK2)
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(labs, fontsize=9)
    ax[2].set_ylim(0, 105)
    ax[2].set_ylabel("reach rate (%)")
    ax[2].set_title("C · Same data, same QP size", loc="left", color=INK)

    for a in ax:
        a.grid(True, axis="y", zorder=0)
        a.set_axisbelow(True)
    fig.suptitle(f"Select-DPC on Reacher-v5 — n_sel={args.n_sel}, n_max={args.n_max}, "
                 f"{bank['Up'].shape[1]} pooled columns, {args.episodes} paired episodes",
                 x=0.005, ha="left", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
    fig.savefig(args.fig, dpi=160)
    print(f"\nwrote {args.fig}")


if __name__ == "__main__":
    main()

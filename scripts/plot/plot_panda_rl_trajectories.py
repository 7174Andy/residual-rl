"""Panda RL arms vs their clone: tip paths per checkpoint, final-policy comparison, video.

    uv run python scripts/plot/plot_panda_rl_trajectories.py            # figures
    uv run python scripts/plot/plot_panda_rl_trajectories.py --video    # + mp4

Default scenarios are one per case, picked from the scored CSVs:
  s54  clone already succeeds (just: step 148 of 150)
  s49  clone fails badly (1.13 m off), every RL policy reaches
  s68  the Select-DPC expert fails too
  s10  vanilla regresses mid-training (reaches at 250k, 251 mm off at 300k)

Outputs (docs/reference/):
  panda_rl_traj_s<id>.png   per scenario: one column per arm, every 50k checkpoint
                            on a light->dark ramp, clone dashed; top view + distance
  panda_rl_vs_clone.png     final (400k) policies vs the clone on every scenario above
  panda_rl_vs_clone_steps.png  all 78 scenarios: clone steps vs each arm's steps
  videos/panda_rl_vs_clone.mp4 (--video)  2x2: clone | residual / vanilla | warm

Every path comes from `run_scenarios`, the harness that scored the CSVs, so a
plotted path is the scored episode. Paths are cached in --cache.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

from panda import scenarios as sc
from panda.env import PandaReachEnv
from panda.eval import run_scenarios
from panda.rl_env import PandaResidualEnv, PandaVanillaEnv, PandaWindowEnv, RLPolicy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/
from validate_panda_clone import ClonePolicy  # noqa: E402  # ty: ignore[unresolved-import]

CLONE = "data/panda_clone_box_r2.pt"
ARMS = {  # arm -> (label, hue)
    "residual": ("Residual", "#2a78d6"),
    "vanilla": ("Vanilla", "#1baf7a"),
    "warm": ("Warm start", "#eb6834"),
}
KS = list(range(50, 401, 50))
TOL = 0.05
INK, INK2, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#b8b7b2", "#e8e7e3", "#fcfcfb"


def rl_env(arm: str):
    if arm == "residual":
        return PandaResidualEnv(CLONE)
    if arm == "warm":
        return PandaWindowEnv()
    return PandaVanillaEnv()


def policy(agent: str):
    """agent = 'clone' or '<arm>_<k>k'."""
    from rl.clone import load_clone
    from rl.sb3 import load_policy
    if agent == "clone":
        return ClonePolicy(load_clone(CLONE), 5, 0.2)
    arm, k = agent.rsplit("_", 1)
    model = load_policy(f"data/panda_{arm}_400k_ckpt/ckpt_{int(k[:-1]) * 1000}_steps.zip",
                        "sac")
    return RLPolicy(rl_env(arm), model)


class TipLogger:
    def __init__(self, pol):
        self.pol, self.tips = pol, []

    def __call__(self, env):
        self.tips.append(env.y.copy())
        return self.pol(env)


def rollout(agent, sid, scen, frames=None):
    env = PandaReachEnv(render_mode="rgb_array" if frames is not None else None)
    log = TipLogger(policy(agent))
    try:
        run_scenarios(env, log, [sid], scen, agent, frames=frames)
        return np.vstack(log.tips + [env.y.copy()])
    finally:
        env.close()


def ramp(hue: str, n: int) -> list:
    """Sequential, one hue: light (early checkpoint) -> full hue -> darker."""
    import matplotlib.colors as mc
    c = np.array(mc.to_rgb(hue))
    t = np.linspace(0, 1, n)
    light, dark = 0.55 * np.ones(3) + 0.45 * c, 0.55 * c
    return [tuple(light + (dark - light) * x) if x > 0.5 else
            tuple(light + (c - light) * (x / 0.5)) for x in t]


def style(plt):
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
        "ytick.color": INK2, "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.8,
    })


def mark_goal(ax, plt, start, goal, i, j):
    ax.add_patch(plt.Circle((goal[i], goal[j]), TOL, fill=False, ls="--", color=INK2, lw=1))
    ax.plot(goal[i], goal[j], "*", ms=13, color=INK, mec=BG, zorder=5)
    ax.plot(start[i], start[j], "s", ms=7, color=INK, mec=BG, mew=1.5, zorder=5)


def dist_mm(P, goal):
    return 1000 * np.linalg.norm(P - goal, axis=1)


def fig_checkpoints(paths, sid, scen, case, out, plt):
    goal, P_clone = np.asarray(scen["goal"][sid]), paths[f"clone_{sid}"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), gridspec_kw={"height_ratios": [1.15, 1]})
    for col, (arm, (label, hue)) in enumerate(ARMS.items()):
        top, dist = axes[0, col], axes[1, col]
        for k, c in zip(KS, ramp(hue, len(KS))):
            P = paths[f"{arm}_{k}k_{sid}"]
            d = dist_mm(P, goal)
            ok = d.min() < 1000 * TOL
            top.plot(P[:, 0], P[:, 1], color=c, lw=1.6)
            top.plot(P[-1, 0], P[-1, 1], "o" if ok else "X", ms=6, color=c, mec=BG)
            dist.plot(d, color=c, lw=1.6, label=f"{k}k {'✓' if ok else '✗'}")
        for ax, xy in ((top, (P_clone[:, 0], P_clone[:, 1])), (dist, (dist_mm(P_clone, goal),))):
            ax.plot(*xy, color=INK2, ls="--", lw=1.4, label="clone" if ax is dist else None)
        mark_goal(top, plt, P_clone[0], goal, 0, 1)
        top.set_aspect("equal", adjustable="datalim")
        top.set(xlabel="x (m)", ylabel="y (m)")
        top.set_title(f"{label} · top view", loc="left", fontsize=10)
        dist.axhline(1000 * TOL, color=INK2, ls=":", lw=1)
        # headroom so the 3-row legend sits above the curves, not on them
        dist.set(xlabel="control step (50 Hz)", ylabel="tip distance to goal (mm)",
                 ylim=(0, 1.45 * dist_mm(P_clone, goal).max()))
        dist.set_title(f"{label} · distance to goal", loc="left", fontsize=10)
        dist.legend(fontsize=7.5, ncol=3, frameon=False, loc="upper right")
        for ax in (top, dist):
            ax.grid(True)
    fig.suptitle(f"s{sid:02d} — {case}.  Each 50k checkpoint, light → dark = 50k → 400k;  "
                 "dashed = clone   (■ start  ★ goal  -- 50 mm  ● reached  ✕ missed)",
                 x=0.005, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_finals(paths, cases, scen, out, plt):
    fig, axes = plt.subplots(len(cases), 3, figsize=(14, 3.6 * len(cases)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for row, (sid, case) in zip(axes, cases.items()):
        goal = np.asarray(scen["goal"][sid])
        runs = [("clone", "Clone", INK2, "--")] + [
            (f"{a}_400k", lab, hue, "-") for a, (lab, hue) in ARMS.items()]
        for ax, (i, j, xl, yl, view) in zip(row[:2], [(0, 1, "x (m)", "y (m)", "top view"),
                                                      (0, 2, "x (m)", "z (m)", "side view")]):
            for key, lab, c, ls in runs:
                P = paths[f"{key}_{sid}"]
                ax.plot(P[:, i], P[:, j], color=c, ls=ls, label=lab)
            mark_goal(ax, plt, paths[f"clone_{sid}"][0], goal, i, j)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set(xlabel=xl, ylabel=yl)
            ax.set_title(f"s{sid:02d} · {view}", loc="left", fontsize=10)
            ax.grid(True)
        d = row[2]
        for key, lab, c, ls in runs:
            dd = dist_mm(paths[f"{key}_{sid}"], goal)
            ok = dd.min() < 1000 * TOL
            d.plot(dd, color=c, ls=ls, label=f"{lab}: {'reached in ' + str(int(np.argmax(dd < 1000 * TOL))) + ' steps' if ok else 'missed, closest %.0f mm' % dd.min()}")
        d.axhline(1000 * TOL, color=INK2, ls=":", lw=1)
        d.set(xlabel="control step (50 Hz)", ylabel="tip distance to goal (mm)", ylim=(0, None))
        d.set_title(f"s{sid:02d} · {case}", loc="left", fontsize=10)
        d.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor="none")
        d.grid(axis="y")
    fig.suptitle("Clone vs the final (400k) policy of each RL arm, same start and goal   "
                 "(■ start  ★ goal  -- 50 mm tolerance)", x=0.005, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_steps(results_csv, out, plt):
    rows = {}
    for r in csv.DictReader(open(results_csv)):
        rows.setdefault(r["method"], {})[int(r["scenario_id"])] = r
    clone = rows["panda_clone_box_r2"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, (arm, (label, hue)) in zip(axes, ARMS.items()):
        m = rows[f"panda_{arm}_400k"]
        xs = [int(clone[i]["steps"]) for i in range(78)]
        ys = [int(m[i]["steps"]) for i in range(78)]
        miss = [clone[i]["reached"] != "True" for i in range(78)]
        ax.plot([0, 150], [0, 150], color=MUTED, lw=1, zorder=0)
        ax.axvline(150, color=MUTED, lw=1, ls=":")
        ax.scatter([x for x, f in zip(xs, miss) if not f], [y for y, f in zip(ys, miss) if not f],
                   s=36, color=hue, edgecolor=BG, linewidth=1.5, label="clone reached")
        ax.scatter([x for x, f in zip(xs, miss) if f], [y for y, f in zip(ys, miss) if f],
                   s=40, marker="X", color=hue, edgecolor=BG, linewidth=1, label="clone missed (150 = timeout)")
        med_c, med_a = np.median(xs), np.median(ys)
        ax.set_title(f"{label}: median {med_a:.0f} steps vs clone {med_c:.0f}", loc="left", fontsize=10)
        ax.set(xlabel="clone: steps to reach", xlim=(0, 158), ylim=(0, 158))
        ax.set_aspect("equal")
        ax.grid(True)
    axes[0].set_ylabel("RL policy (400k): steps to reach")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("All 78 box scenarios: every point below the diagonal is a scenario the "
                 "RL policy finishes faster than the clone", x=0.005, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def video(cases, scen, out, fps):
    from PIL import Image, ImageDraw

    from core.video_encoding import encode_video
    tiles = [("clone", "Clone (r2)")] + [(f"{a}_400k", f"{lab} RL, 400k")
                                         for a, (lab, _) in ARMS.items()]
    frames = []
    for sid, case in cases.items():
        clips = []
        for agent, name in tiles:
            fr: list = []
            rollout(agent, sid, scen, frames=fr)
            clips.append((name, fr))
        n = max(len(f) for _, f in clips)
        for t in range(n + fps):                     # hold the last frame ~1 s
            imgs = []
            for name, fr in clips:
                done = t >= len(fr) - 1
                im = Image.fromarray(fr[min(t, len(fr) - 1)])
                d = ImageDraw.Draw(im)
                step = min(t, len(fr) - 1)
                d.rectangle((0, 0, im.width, 22), fill=(252, 252, 251))
                d.text((6, 5), f"{name}   step {step}{'  (done)' if done else ''}",
                       fill=(11, 11, 11))
                imgs.append(np.asarray(im))
            top, bot = np.hstack(imgs[:2]), np.hstack(imgs[2:])
            grid = Image.fromarray(np.vstack([top, bot]))
            ImageDraw.Draw(grid).text((6, grid.height - 16), f"s{sid:02d}: {case}",
                                      fill=(11, 11, 11))
            frames.append(np.asarray(grid))
    encode_video(frames, out, fps)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario-ids", type=int, nargs="+", default=[54, 49, 68, 10])
    p.add_argument("--cases", nargs="+", default=[
        "clone already succeeds (just)", "clone fails badly, RL rescues",
        "expert fails too", "vanilla regresses mid-training"])
    p.add_argument("--scenarios", default="data/panda_scenarios_box_v1.npz")
    p.add_argument("--results", default="data/panda_results.csv")
    p.add_argument("--cache", default="data/panda_rl_traj.npz")
    p.add_argument("--outdir", default="docs/reference")
    p.add_argument("--video", action="store_true")
    p.add_argument("--video-out", default="videos/panda_rl_vs_clone.mp4")
    p.add_argument("--fps", type=int, default=25, help="50 Hz control shown at half speed")
    args = p.parse_args()
    if len(args.cases) != len(args.scenario_ids):
        p.error("--cases needs one label per --scenario-ids entry")
    cases = dict(zip(args.scenario_ids, args.cases))

    scen = sc.load(args.scenarios)
    paths = dict(np.load(args.cache)) if os.path.exists(args.cache) else {}
    agents = ["clone"] + [f"{a}_{k}k" for a in ARMS for k in KS]
    for agent in agents:
        todo = [s for s in cases if f"{agent}_{s}" not in paths]
        for sid in todo:
            paths[f"{agent}_{sid}"] = rollout(agent, sid, scen)
        if todo:
            np.savez(args.cache, **paths)
            print(f"  {agent}: {len(todo)} scenario(s)", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    for sid, case in cases.items():
        out = os.path.join(args.outdir, f"panda_rl_traj_s{sid:02d}.png")
        fig_checkpoints(paths, sid, scen, case, out, plt)
        print(f"wrote {out}")
    fig_finals(paths, cases, scen, os.path.join(args.outdir, "panda_rl_vs_clone.png"), plt)
    fig_steps(args.results, os.path.join(args.outdir, "panda_rl_vs_clone_steps.png"), plt)
    print(f"wrote {args.outdir}/panda_rl_vs_clone.png, panda_rl_vs_clone_steps.png")
    if args.video:
        video(cases, scen, args.video_out, args.fps)
        print(f"wrote {args.video_out}")


if __name__ == "__main__":
    main()

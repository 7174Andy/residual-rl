"""Deployed reach rate vs training steps for the Reacher RL arms (journey 13).

The training-return curve measures the behaviour policy (exploration noise,
whatever starts training sampled). This sweeps the saved checkpoints with
greedy actions over the frozen scenarios instead — the sample-efficiency
crossover in the form that matters: where does vanilla RL catch the clone, and
where does it catch the residual?

Consumes the checkpoints written by `train_reacher_{residual,vanilla}.py
--checkpoint-dir` (`ckpt_<steps>_steps.zip`). Select-DPC and the clone are not
re-run: both are training-step-independent flat lines spliced from the eval CSV
written by `scripts/eval_reacher_residual.py` (CSVs are gitignored repo-wide;
rerun that eval on a fresh clone).

    uv run python scripts/sweep_reacher_checkpoints.py

The sweep costs ~10 min of rollouts; to only redraw the figure from a previous
run's sweep CSV, pass `--from-csv docs/reference/reacher_crossover.csv`.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rl.stats import wilson_ci  # noqa: E402

BLUE, ORANGE, MUTED2, CRITICAL = "#2a78d6", "#eb6834", "#8a8a86", "#d03b3b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})


def checkpoints(dirpath: str) -> list[tuple[int, str]]:
    out = []
    for f in os.listdir(dirpath):
        m = re.fullmatch(r"ckpt_(\d+)_steps\.zip", f)
        if m:
            out.append((int(m.group(1)), os.path.join(dirpath, f)))
    return sorted(out)


def eval_residual(res_env, model, eps):
    rows = []
    for q0, g in eps:
        obs, info = res_env.reset(seed=0, options={"qpos": q0, "goal": g})
        best, reached = float(info["dist"]), False
        for _t in range(res_env.base.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, _term, trunc, info = res_env.step(action)
            best = min(best, float(info["dist"]))
            reached = reached or bool(info["reached"])
            if trunc:
                break
        rows.append({"best": best, "final": float(info["dist"]),
                     "reached": reached})
    return rows


def eval_vanilla(env, model, eps):
    from reacher.eval import run_episode

    def policy(env_, _info):
        action, _ = model.predict(env_.unwrapped.build_obs(),
                                  deterministic=True)
        return action

    return [run_episode(env, policy, q0, g) for q0, g in eps]


def sweep(args) -> tuple[dict, int]:
    """Evaluate every checkpoint; returns {arm: [(steps, reached, best_mm,
    final_mm)]} and the scenario count."""
    import gymnasium as gym

    import reacher  # noqa: F401  registers the Gym ID
    from reacher.residual_env import ResidualSelectEnv
    from rl.sb3 import load_policy

    with np.load(args.scenarios) as z:
        eps = [(z["qpos"][i], z["goal"][i]) for i in range(len(z["qpos"]))]
    n = len(eps)

    def agg_of(rows):
        return (sum(x["reached"] for x in rows),
                1e3 * float(np.median([x["best"] for x in rows])),
                1e3 * float(np.median([x["final"] for x in rows])))

    agg = defaultdict(list)
    res_env = ResidualSelectEnv(clone_path=args.clone,
                                residual_frac=args.residual_frac)
    for steps, path in checkpoints(args.residual_ckpts):
        model = load_policy(path, algo=args.algo, device="cpu")
        agg["residual"].append((steps, *agg_of(eval_residual(res_env, model,
                                                             eps))))
    res_env.close()

    van_env = gym.make("ReacherGoal-v0")
    for steps, path in checkpoints(args.vanilla_ckpts):
        model = load_policy(path, algo=args.algo, device="cpu")
        agg["vanilla"].append((steps, *agg_of(eval_vanilla(van_env, model,
                                                           eps))))
    van_env.close()
    return agg, n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", default="data/reacher_scenarios_v1.npz")
    p.add_argument("--clone", default="data/dagger_clone_r3.pt")
    p.add_argument("--residual-ckpts",
                   default="data/reacher_ckpt_seeds/resf2_s0")
    p.add_argument("--vanilla-ckpts", default="data/reacher_van_ckpt_400k")
    p.add_argument("--algo", default="sac")
    p.add_argument("--residual-frac", type=float, default=2.0,
                   help="eval-env authority; must match the frac the residual "
                        "checkpoints were trained with (journey 13's are 2.0)")
    p.add_argument("--reference-csv",
                   default="docs/reference/reacher_residual_200k_rerun.csv",
                   help="eval CSV supplying the flat Select-DPC / "
                        "clone reference lines")
    p.add_argument("--wandb-project", default=None,
                   help="log the sweep table + figure to Weights & Biases")
    p.add_argument("--from-csv", default=None,
                   help="redraw the figure from a previous run's sweep CSV "
                        "instead of re-evaluating the checkpoints")
    p.add_argument("--out", default="docs/reference/reacher_crossover.png")
    args = p.parse_args()

    if args.from_csv:
        agg, n = defaultdict(list), 0
        with open(args.from_csv) as f:
            next(f)  # producing-command comment
            for x in csv.DictReader(f):
                n = int(x["n"])
                agg[x["arm"]].append((int(x["steps"]), int(x["reached"]),
                                      float(x["best_med_mm"]),
                                      float(x["final_med_mm"])))
    else:
        agg, n = sweep(args)

    ref = defaultdict(list)
    with open(args.reference_csv) as f:
        next(f)
        for x in csv.DictReader(f):
            ref[x["controller"]].append(x)
    expert_rate = 100 * sum(int(x["reached"]) for x in ref["Select-DPC"]) / n
    clone_rate = 100 * sum(int(x["reached"]) for x in ref["clone"]) / n
    clone_final = np.median([float(x["final"]) for x in ref["clone"]]) * 1e3

    by_steps = defaultdict(dict)
    for arm in ("residual", "vanilla"):
        for steps, k, _b, _f in agg[arm]:
            by_steps[steps][arm] = k
    print(f"\n  {n} frozen scenarios, greedy, full horizon\n")
    print(f"  {'steps':>8}{'residual':>12}{'vanilla':>12}{'diff (pp)':>12}")
    for steps in sorted(by_steps):
        kr = by_steps[steps].get("residual")
        kv = by_steps[steps].get("vanilla")
        d = f"{100 * (kr - kv) / n:+.1f}" if kr is not None and kv is not None \
            else ""
        cr = "" if kr is None else f"{kr}/{n}"
        cv = "" if kv is None else f"{kv}/{n}"
        print(f"  {steps:>8}{cr:>12}{cv:>12}{d:>12}")

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))
    for arm, c in (("residual", BLUE), ("vanilla", ORANGE)):
        s = np.array([row[0] for row in agg[arm]]) / 1e3
        k = np.array([row[1] for row in agg[arm]])
        ci = np.array([wilson_ci(int(v), n) for v in k]) * 100
        ax[0].plot(s, 100 * k / n, color=c, marker="o", ms=4, zorder=3,
                   label="clone + residual" if arm == "residual" else
                         "vanilla RL")
        ax[0].fill_between(s, ci[:, 0], ci[:, 1], color=c, alpha=0.15,
                           lw=0, zorder=2)
        ax[1].plot(s, [row[3] for row in agg[arm]], color=c, marker="o",
                   ms=4, zorder=3)
    ax[0].axhline(expert_rate, color=MUTED2, lw=1.2, ls=":", zorder=1)
    ax[0].axhline(clone_rate, color=MUTED2, lw=1.2, ls="--", zorder=1)
    ax[0].text(398, expert_rate + 1.5, "Select-DPC expert", ha="right",
               fontsize=8, color=MUTED2)
    ax[0].text(398, clone_rate + 1.5, "clone (residual's floor)", ha="right",
               fontsize=8, color=MUTED2)
    ax[0].set_xlabel("training steps (k)")
    ax[0].set_ylabel("reach rate (%)  ·  Wilson 95% band")
    ax[0].set_ylim(0, 104)
    ax[0].set_title("A · Deployed reach rate vs training", loc="left",
                    color=INK)
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")

    ax[1].axhline(10.0, color=CRITICAL, lw=1.2, ls="--", zorder=1)
    ax[1].text(398, 10.8, "10 mm tolerance", ha="right", fontsize=8,
               color=CRITICAL)
    ax[1].axhline(clone_final, color=MUTED2, lw=1.2, ls="--", zorder=1)
    ax[1].text(398, clone_final * 1.05, "clone", ha="right", fontsize=8,
               color=MUTED2)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("training steps (k)")
    ax[1].set_ylabel("median final distance (mm)")
    ax[1].set_title("B · Precision vs training", loc="left", color=INK)

    # C: the residual's advantage, checkpoint by checkpoint
    steps = sorted(s for s in by_steps
                   if {"residual", "vanilla"} <= by_steps[s].keys())
    delta = np.array([100 * (by_steps[s]["residual"] - by_steps[s]["vanilla"])
                      / n for s in steps])
    xs = np.array(steps) / 1e3
    ax[2].bar(xs, delta, width=18, zorder=3,
              color=[BLUE if d > 0 else ORANGE for d in delta])
    ax[2].axhline(0, color=INK2, lw=1.0, zorder=4)
    peak = int(np.argmax(delta))
    ax[2].annotate(f"+{delta[peak]:.0f} pp", (xs[peak], delta[peak]),
                   textcoords="offset points", xytext=(14, -6), ha="left",
                   fontsize=8, color=INK2)
    ax[2].set_ylim(delta.min() - 5, delta.max() + 9)
    ax[2].text(0.03, 0.93, "residual ahead", transform=ax[2].transAxes,
               fontsize=8, color=BLUE)
    ax[2].text(0.97, 0.05, "vanilla ahead", transform=ax[2].transAxes,
               ha="right", fontsize=8, color=ORANGE)
    ax[2].set_xlabel("training steps (k)")
    ax[2].set_ylabel("reach-rate difference (pp)")
    ax[2].set_title("C · Residual − vanilla, same scenarios", loc="left",
                    color=INK)

    for a in ax:
        a.grid(True, zorder=0)
        a.set_axisbelow(True)
    fig.suptitle("Reacher — the sample-efficiency crossover, 120 frozen "
                 "scenarios, greedy", x=0.005, ha="left", color=INK2,
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"\nwrote {args.out}")

    if not args.from_csv:
        out_csv = args.out.replace(".png", ".csv")
        with open(out_csv, "w") as fh:
            fh.write("# scripts/sweep_reacher_checkpoints.py\n")
            fh.write("arm,steps,reached,n,best_med_mm,final_med_mm\n")
            for arm in ("residual", "vanilla"):
                for steps_, k, b, f in agg[arm]:
                    fh.write(f"{arm},{steps_},{k},{n},{b:.2f},{f:.2f}\n")
        print(f"wrote {out_csv}")

    if args.wandb_project:
        from rl.wb import finish, init_run, log_image, log_table
        run = init_run(args.wandb_project,
                       name=os.path.basename(args.out).replace(".png", ""),
                       config=vars(args), tags=["reacher", "sweep"])
        log_table(run, "checkpoint_sweep",
                  ["arm", "steps", "reached", "n", "best_med_mm",
                   "final_med_mm"],
                  [[arm, s_, k_, n, b_, f_]
                   for arm in ("residual", "vanilla")
                   for s_, k_, b_, f_ in agg[arm]])
        log_image(run, "crossover", args.out)
        finish(run)


if __name__ == "__main__":
    main()

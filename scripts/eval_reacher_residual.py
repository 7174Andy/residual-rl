"""The results table: Select-DPC, clone, clone+residual, vanilla, random.

Every row is scored by `reacher/eval.py::run_episode` on the SAME frozen
scenarios with the full horizon, so no row can differ by protocol. Metric
definitions and what each one is load-bearing for: docs/reference/metrics.md.

The two columns to read first:

  best -> final   journey 12 measured 2.1-2.3x on every controller and traced it
                  to the receding-horizon cost having no terminal term. The
                  residual's reward pays distance every step, so THIS ratio is
                  the novel claim; a residual that only lifts reach rate has
                  reproduced the unicycle, not advanced it.
  vs vanilla      journey 09's precedent is that vanilla may win. Reported
                  either way.

Context for reading the `clone` row: the clone did NOT pass the fidelity gate
(see `.superpowers/sdd/2026-08-16-reacher-residual-rl/task-6-diagnosis.md`). It
reproduces the base's STEERING (path/net 1.6 against 1.5) but loses terminal
precision, landing on the knife edge of the 10 mm criterion. The residual exists
to add exactly that precision, so this table is the test of whether it does.

    uv run python scripts/eval_reacher_residual.py
"""
from __future__ import annotations

import argparse
import os

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import reacher  # noqa: F401,E402  registers the Gym ID
from reacher.clone_data import build_bank, build_select_controller  # noqa: E402
from reacher.eval import (  # noqa: E402
    ClonePolicy, ControllerPolicy, WarmStartClonePolicy, run_episode,
)
from reacher.model import NQ_ARM, load_model  # noqa: E402
from rl.clone import load_clone  # noqa: E402
from rl.sb3 import load_policy  # noqa: E402
from rl.stats import mcnemar_pvalue, wilson_ci  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, CRITICAL = "#0b0b0b", "#52514e", "#b8b7b2", "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})


def _rollout_residual(res_env, model, qpos, goal, seed=0):
    """Residual rows need the residual env, so they get their own loop that
    returns the identical dictionary `run_episode` does."""
    obs, info = res_env.reset(seed=seed, options={"qpos": qpos, "goal": goal})
    need = float(info["dist"])
    best, reached_at = need, None
    for t in range(res_env.base.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, _term, trunc, info = res_env.step(action)
        best = min(best, float(info["dist"]))
        if reached_at is None and info["reached"]:
            reached_at = t + 1
        if trunc:
            break
    net = need - best
    return {"need": need, "best": best, "final": float(info["dist"]),
            "path": float(info["path"]),
            "eff": float(info["path"] / net) if net > 1e-4 else float("nan"),
            "reached": reached_at is not None, "steps": reached_at,
            "steps_run": t + 1}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", default="data/reacher_scenarios_v1.npz")
    p.add_argument("--clone", default="data/dagger_clone_r3.pt")
    p.add_argument("--residual",
                   default="data/reacher_ckpt_seeds/resf2_s0/ckpt_200000_steps.zip")
    p.add_argument("--residual-frac", type=float, default=2.0,
                   help="eval-env authority; must match the frac the residual "
                        "checkpoint was trained with (journey 13's are 2.0)")
    p.add_argument("--vanilla", default="data/reacher_vanilla_200k.zip")
    p.add_argument("--algo", default="sac")
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--out", default="docs/reference/reacher_residual.png")
    p.add_argument("--memoryless", action="store_true",
                   help="build the Select-DPC row with carry_prediction=False. "
                        "MUST match the expert the clone was trained against, or "
                        "the clone is scored against a controller it never imitated.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with np.load(args.scenarios) as z:
        eps = [(z["qpos"][i], z["goal"][i]) for i in range(len(z["qpos"]))]
    eps = eps[: args.episodes]

    from reacher.residual_env import ResidualSelectEnv

    model_mj, data_mj = load_model()
    bank, _ = build_bank(model_mj, data_mj, np.random.default_rng(args.seed))
    predictor = load_clone(args.clone, device="cpu")
    rng = np.random.default_rng(args.seed)

    def random_policy(_env, _info):
        return rng.uniform(-1, 1, NQ_ARM)

    env = gym.make("ReacherGoal-v0")
    rows = {}
    for label, pol in (
        ("Select-DPC", ControllerPolicy(build_select_controller(
            bank, carry_prediction=not args.memoryless))),
        ("clone", ClonePolicy(predictor)),
        ("clone + warmstart", WarmStartClonePolicy(
            predictor, build_select_controller(
                bank, carry_prediction=not args.memoryless))),
        ("random", random_policy),
    ):
        rows[label] = [run_episode(env, pol, q0, g) for q0, g in eps]
    env.close()

    res_env = ResidualSelectEnv(clone_path=args.clone,
                                residual_frac=args.residual_frac)
    res_model = load_policy(args.residual, algo=args.algo, device="cpu")
    rows["clone + residual"] = [_rollout_residual(res_env, res_model, q0, g)
                                for q0, g in eps]
    res_env.close()

    van_env = gym.make("ReacherGoal-v0")
    van_model = load_policy(args.vanilla, algo=args.algo, device="cpu")

    def vanilla_policy(env_, _info):
        action, _ = van_model.predict(env_.unwrapped.build_obs(),
                                      deterministic=True)
        return action

    rows["vanilla RL"] = [run_episode(van_env, vanilla_policy, q0, g)
                          for q0, g in eps]
    van_env.close()

    order = ["Select-DPC", "clone", "clone + warmstart", "clone + residual",
             "vanilla RL", "random"]
    n = len(eps)
    print(f"\n  {n} frozen scenarios, full horizon, early stopping OFF\n")
    print(f"  {'controller':<19}{'reach rate (95% CI)':>22}{'best':>10}"
          f"{'final':>10}{'best->final':>13}{'path/net':>10}")
    for label in order:
        r = rows[label]
        k = sum(x["reached"] for x in r)
        lo, hi = wilson_ci(k, n)
        b = np.median([x["best"] for x in r])
        f = np.median([x["final"] for x in r])
        print(f"  {label:<19}{k:>4}/{n:<4}[{100*lo:>3.0f}-{100*hi:<3.0f}%]"
              f"{b*1e3:>9.1f}mm{f*1e3:>9.1f}mm{f/max(b,1e-9):>12.1f}x"
              f"{np.nanmedian([x['eff'] for x in r]):>10.1f}")

    base = rows["clone"]
    print("\n  paired against the clone (the residual's own baseline):")
    for label in ("clone + residual", "vanilla RL"):
        r = rows[label]
        rescue = sum(x["reached"] and not y["reached"] for x, y in zip(r, base))
        regress = sum(y["reached"] and not x["reached"] for x, y in zip(r, base))
        closer = sum(x["best"] < y["best"] for x, y in zip(r, base))
        print(f"    {label:<19}{rescue:>3} rescues, {regress:>3} regressions, "
              f"closer on {closer}/{n}, McNemar p = "
              f"{mcnemar_pvalue(rescue, regress):.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(13.0, 4.1))
    colours = dict(zip(order, SERIES + [SERIES[0], MUTED]))
    y = np.arange(len(order))
    ax[0].barh(y, [100 * sum(x["reached"] for x in rows[lab]) / n for lab in order],
               0.55, color=[colours[lab] for lab in order], zorder=3)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(order, fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("reach rate (%)")
    ax[0].set_title("A · Reach rate", loc="left", color=INK)

    for label in order:
        v = np.sort([x["final"] for x in rows[label]]) * 1e3
        ax[1].step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                   color=colours[label], zorder=3, label=label)
    ax[1].axvline(10.0, color=CRITICAL, lw=1.2, ls="--", zorder=2)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("uncensored final distance (mm)")
    ax[1].set_ylabel("fraction of scenarios below")
    ax[1].set_title("B · Does it HOLD?", loc="left", color=INK)
    ax[1].legend(frameon=False, fontsize=7, loc="upper left")

    ratios = [np.median([x["final"] for x in rows[lab]])
              / max(np.median([x["best"] for x in rows[lab]]), 1e-9)
              for lab in order]
    ax[2].barh(y, ratios, 0.55, color=[colours[lab] for lab in order], zorder=3)
    ax[2].axvline(1.0, color=CRITICAL, lw=1.2, ls="--", zorder=4)
    ax[2].set_yticks(y)
    ax[2].set_yticklabels(order, fontsize=8)
    ax[2].invert_yaxis()
    ax[2].set_xlabel("best → final ratio (1.0 = holds position)")
    ax[2].set_title("C · The drift", loc="left", color=INK)

    for a in ax:
        a.grid(True, zorder=0)
        a.set_axisbelow(True)
    fig.suptitle("Reacher: Select-DPC → clone → residual, uncensored",
                 x=0.005, ha="left", color=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"\nwrote {args.out}")

    csv = args.out.replace(".png", ".csv")
    with open(csv, "w") as fh:
        fh.write("# scripts/eval_reacher_residual.py\n")
        fh.write("controller,scenario,need,best,final,path,eff,reached\n")
        for label in order:
            for i, x in enumerate(rows[label]):
                fh.write(f"{label},{i},{x['need']:.5f},{x['best']:.5f},"
                         f"{x['final']:.5f},{x['path']:.5f},{x['eff']:.4f},"
                         f"{int(x['reached'])}\n")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()

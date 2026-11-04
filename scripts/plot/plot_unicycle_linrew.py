"""Two-wheel (unicycle) RL under the linear reward: training return and deployed reach,
residual / vanilla / warm start / cold (soft SAC, lr 3e-4, seed 0, 100k steps).

    D=data/unicycle_linrew; SQ=data/unicycle_dagger_clone_r3_ego_squash.pt
    uv run python scripts/train/train_residual.py --algo sac --lr 3e-4 --timesteps 100000 --reward linear \
        --checkpoint-freq 5000 --out $D/lin_ego_warm.zip --checkpoint-dir $D/lin_ego_warm/res_s0 \
        --monitor-out $D/lin_ego_warm --warm --ego-obs --clone $SQ      # cold: + --cold
    # residual: --residual-frac 2.0 --clone data/clone.pt (no --warm); vanilla: train_vanilla.py
    uv run python scripts/sweep_checkpoints.py --ckpt-root $D/lin_ego_warm --arms residual --algo sac \
        --train-seeds 0 --steps 0,5000,...,100000 --label lin_ego_warm --warm --ego-obs --clone $SQ \
        --out $D/sweep_lin_ego_warm.csv
    uv run python scripts/plot/plot_unicycle_linrew.py                       # the four arms
    uv run python scripts/plot/plot_unicycle_linrew.py --set residual_dawn   # + --warmup-steps 20000
                                                                        #   [--critic-layernorm]

The `--reward linear` option (TwoWheelGoal-v0 `reward="linear"`: -||p - g|| - u'Ru + 100
on reach) was rolled back and then restored on `train_residual.py` only (2026-10-27), so
every run here except vanilla (`train_vanilla.py` has no `--reward`) re-runs as above.

Return is the trailing 100-episode mean of the linear reward (`reward="linear"`) from
each run's Monitor CSV. Both panels start at step 0 from the run's step-0 policy
(`ckpt_0_steps.zip`) acting alone: for the return that is its deterministic linear return
on the 78 eval seeds, recomputed from the rollout and counted as episode 0 of the
trailing window (`rl.curves`), so the curve starts where the reach curve does. DAWN's
20k warmup transitions precede step 0 and are not on this axis. Reach is the greedy policy on the 78 canonical eval seeds at
every 5k checkpoint; step 0 is the prior acting alone (vanilla saves no step-0 policy).
The world-frame warm arm (same recipe, world-frame clone window) is the dashed reference.
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rl.curves import return_curve  # noqa: E402
from rl.sb3 import load_policy  # noqa: E402
from two_wheel_robot.rl.residual_env import ResidualDeePCEnv, WarmDeePCEnv  # noqa: E402
from two_wheel_robot.rl.residual_eval import run_residual_closed_loop_with_actions  # noqa: E402

N = 78
# figure -> (title, [(label, run, colour (categorical slots in fixed order), line style)])
SETS = {
    "arms": ("residual, vanilla, warm-start and cold RL under the linear reward", [
        ("residual RL (base: DeePC clone)", "lin_res", "#2a78d6", "-"),
        ("vanilla RL (from scratch)", "lin_van", "#eb6834", "-"),
        ("warm-start RL, robot-centred window", "lin_ego_warm", "#1baf7a", "-"),
        ("cold start, robot-centred window", "lin_ego_cold", "#eda100", "-"),
        ("warm-start RL, world-frame window (before)", "lin_warm", "#e87ba4", "--")]),
    "residual_dawn": ("residual RL with vs without DAWN under the linear reward "
                      "(DAWN's 20k warmup transitions precede step 0)", [
        ("residual RL (base: DeePC clone)", "lin_res", "#2a78d6", "-"),
        ("+ DAWN warmup data (20k clone transitions)", "lin_res_dawn", "#eb6834", "-"),
        ("+ full DAWN (warmup + LayerNorm critic)", "lin_res_dawnln", "#1baf7a", "-")]),
}
SEEDS = [4104626029 + i for i in range(N)]
EGO = "data/unicycle_dagger_clone_r3_ego_squash.pt"
# run -> the env its step-0 policy acts in (vanilla saves no step-0 policy)
ENVS = {
    **{r: lambda: ResidualDeePCEnv("data/clone.pt", residual_frac=2.0)
       for r in ("lin_res", "lin_res_dawn", "lin_res_dawnln")},
    **{r: lambda: WarmDeePCEnv(EGO, ego_obs=True) for r in ("lin_ego_warm", "lin_ego_cold")},
    "lin_warm": lambda: WarmDeePCEnv("data/unicycle_dagger_clone_r3_squash.pt"),
}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9.5,
})


def reach(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["steps"]))
    return (np.array([int(r["steps"]) for r in rows]),
            100 * np.array([int(r["reach"]) for r in rows]) / N)


def step0_return(d: str, run: str) -> float | None:
    """Mean linear return of the run's step-0 policy, greedy, on the 78 eval seeds.

    Recomputed from each rollout with the formula the run trained on: per step
    -||p - g|| - u'Ru on the post-step position and applied action, +100 on reach.
    """
    path = f"{d}/{run}/res_s0/ckpt_0_steps.zip"
    if run not in ENVS or not os.path.exists(path):
        return None
    model, env = load_policy(path, "sac", "cpu"), ENVS[run]()
    rets = []
    for s in SEEDS:
        reached, traj, acts = run_residual_closed_loop_with_actions(model, env, s)
        dist = np.linalg.norm(traj[1:, :2] - env.base.goal, axis=1)
        rets.append(-dist.sum() - np.einsum("ti,ij,tj->", acts, env.base.R, acts)
                    + 100.0 * reached)
    env.close()
    return float(np.mean(rets))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="data/unicycle_linrew")
    p.add_argument("--set", choices=sorted(SETS), default="arms")
    p.add_argument("--out", default=None,
                   help="default docs/journey/figures/unicycle_linrew[_residual_dawn].png")
    args = p.parse_args()
    title, runs = SETS[args.set]
    args.out = args.out or ("docs/journey/figures/unicycle_linrew"
                            + ("" if args.set == "arms" else f"_{args.set}") + ".png")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    for label, run, col, ls in runs:
        ret0 = step0_return(args.dir, run)
        s, _ep, r = return_curve(f"{args.dir}/{run}.monitor.csv", step0=ret0)
        ax[0].plot(s / 1e3, r, color=col, ls=ls, lw=2)
        x, y = reach(f"{args.dir}/sweep_{run}.csv")
        full = x[y >= 100]
        first = f"100% at {full[0] // 1000}k" if len(full) else f"{y[-1]:.0f}% at 100k"
        ax[1].plot(x / 1e3, y, color=col, ls=ls, lw=2, marker="o", ms=4,
                   label=f"{label}: {first}")
    for a in ax:
        a.set_xlim(0, 100)
        a.set_xlabel("RL training steps (k)")
        a.grid(color=MUTED, alpha=0.35, lw=0.6)
    ax[0].set(ylabel="training episode return (trailing 100-ep mean)",
              title="Training return (exploring policy, linear reward)")
    ax[1].set(ylabel=f"reached (% of {N} eval seeds, deterministic)", ylim=(0, 105),
              title="Deployed reach rate (checkpoint every 5k steps)")
    # One shared legend (same colours in both panels), in the return panel's empty corner.
    ax[0].legend(*ax[1].get_legend_handles_labels(), fontsize=8.5, loc="lower right",
                 frameon=False)
    fig.suptitle(f"Two-wheel (unicycle) experiment: {title}\n(soft SAC, lr 3e-4, seed 0)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

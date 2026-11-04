"""Warm-start RL with vs without DAWN on all three systems: training return (top) and
deployed reach (bottom), soft SAC, seed 0.

    uv run python scripts/plot/plot_warmstart_dawn.py

Runs (each system's own warm-start recipe; DAWN = `--warmup-steps 20000`, full DAWN
adds `--critic-layernorm`):

* Two-wheel (unicycle), linear reward: base `data/unicycle_linrew/lin_ego_warm`, DAWN
  `data/unicycle_ego_dawn/ego_warm_{dawn,dawnln}_lin` -- `train_residual.py --algo sac
  --lr 3e-4 --reward linear --warm --ego-obs --clone
  data/unicycle_dagger_clone_r3_ego_squash.pt`, 100k, 5k checkpoints, 78 seeds.
* Reacher: `data/repro/{q_warm,dawn_w_warmup,dawn_w}_150k` -- journey 20's runs
  (`scripts/plot/plot_dawn.py`, 100k) re-trained to 150k with identical settings:
  `train_reacher_window.py --clone data/dagger_clone_r3_squash.pt --steps 150000
  --checkpoint-freq 2000`, swept by `sweep_checkpoints_perscn.py --env window`;
  2k checkpoints, 120 scenarios.
* Panda: `data/panda_warm_{base,dawn,dawnln}_200k` (journey 20,
  `scripts/plot/plot_panda_dawn.py --arm warm`), 200k, 5k checkpoints, 78 scenarios.

Every curve starts at step 0 from the system's squash clone acting alone: its reach,
and its deterministic mean return counted as episode 0 of the trailing window
(`rl.curves`). DAWN's 20k warmup transitions precede step 0 and are not on the axis.
Returns are each system's own env reward, so they are not comparable across columns.
"""
from __future__ import annotations

import argparse
import collections
import csv
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rl.curves import return_curve  # noqa: E402
from rl.sb3 import load_policy  # noqa: E402
from two_wheel_robot.rl.residual_env import WarmDeePCEnv  # noqa: E402

VARIANTS = (  # key, label, colour (categorical slots 1-3, fixed order)
    ("base", "warm-start RL", "#2a78d6"),
    ("dawn", "+ DAWN warmup data (20k clone transitions)", "#eb6834"),
    ("dawnln", "+ full DAWN (warmup + LayerNorm critic)", "#1baf7a"),
)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9.5,
})


UNI_CLONE = "data/unicycle_dagger_clone_r3_ego_squash.pt"
_uni_ret0: list[float] = []


def unicycle_step0_return(ckpt: str) -> float:
    """The clone's greedy mean LINEAR return on the 78 eval seeds, summed from the env.

    The sweep's `return_mean` is the DeePC-form reward, not what these runs trained
    on. All three variants share one step-0 policy (the clone), so it is computed once.
    """
    if not _uni_ret0:
        model = load_policy(ckpt, "sac", "cpu")
        env = WarmDeePCEnv(UNI_CLONE, ego_obs=True, reward="linear")
        rets = []
        for seed in range(4104626029, 4104626029 + 78):
            obs, _ = env.reset(seed=seed)
            done, total = False, 0.0
            while not done:
                obs, r, term, trunc, _ = env.step(model.predict(obs, deterministic=True)[0])
                total, done = total + r, term or trunc
            rets.append(total)
        env.close()
        _uni_ret0.append(float(np.mean(rets)))
    return _uni_ret0[0]


def unicycle(key: str):
    """(monitor, steps, reach %, step-0 return) from the 5k sweep's summary rows."""
    d, run = {"base": ("data/unicycle_linrew", "lin_ego_warm"),
              "dawn": ("data/unicycle_ego_dawn", "ego_warm_dawn_lin"),
              "dawnln": ("data/unicycle_ego_dawn", "ego_warm_dawnln_lin")}[key]
    with open(f"{d}/sweep_{run}.csv") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["steps"]))
    return (f"{d}/{run}.monitor.csv", np.array([int(r["steps"]) for r in rows]),
            100 * np.array([int(r["reach"]) for r in rows]) / 78,
            unicycle_step0_return(f"{d}/{run}/res_s0/ckpt_0_steps.zip"))


def reacher(key: str):
    """Per-scenario sweep rows; step 0 is the squash clone (79/120, return 6.69,
    measured 2026-10-22 -- see plot_reacher_warmstart_returns.py)."""
    run = {"base": "q_warm", "dawn": "dawn_w_warmup", "dawnln": "dawn_w"}[key] + "_150k"
    k: dict[int, int] = collections.defaultdict(int)
    with open(f"data/repro/sweep_{run}.csv") as f:
        for r in csv.DictReader(f):
            k[int(r["steps"])] += int(r["reached"])
    pts = sorted({0: 79, **k}.items())
    return (f"data/repro/{run}.monitor.csv", np.array([s for s, _ in pts]),
            100 * np.array([n for _, n in pts]) / 120, 6.69)


def panda(key: str):
    """Per-scenario sweep rows keyed by checkpoint path; step 0 from panda_results.csv."""
    run = f"panda_warm_{key}_200k"
    reached = collections.Counter()
    with open("data/panda_dawn_sweep_warm.csv") as f:
        for r in csv.DictReader(f):
            if f"/{run}_ckpt/" in r["method"]:
                reached[int(re.findall(r"ckpt_(\d+)_steps", r["method"])[0])] += r["reached"] == "True"
    with open("data/panda_results.csv") as f:
        prior = [r for r in csv.DictReader(f) if r["method"] == "panda_clone_box_r2_squash"]
    pts = sorted({0: sum(r["reached"] == "True" for r in prior), **reached}.items())
    return (f"data/{run}.monitor.csv", np.array([s for s, _ in pts]),
            100 * np.array([n for _, n in pts]) / 78,
            float(np.mean([float(r["ret"]) for r in prior])))


SYSTEMS = (  # column title, loader, x-axis limit (k steps)
    ("Two-wheel (unicycle), robot-centred window, linear reward", unicycle, 100),
    ("Reacher", reacher, 150),
    ("Panda", panda, 200),
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="docs/journey/figures/warmstart_dawn_3sys.png")
    args = p.parse_args()

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    for col, (title, load, kmax) in enumerate(SYSTEMS):
        for key, label, colour in VARIANTS:
            monitor, x, y, ret0 = load(key)
            s, _ep, r = return_curve(monitor, step0=ret0)
            ax[0, col].plot(s / 1e3, r, color=colour, lw=2, label=label)
            ax[1, col].plot(x / 1e3, y, color=colour, lw=2, marker="o", ms=3)
        ax[1, col].axhline(y[0], color=INK2, ls=":", lw=1)
        ax[1, col].text(kmax * 0.6, y[0] + 2, f"the clone: {y[0]:.0f}%", fontsize=8, color=INK2)
        ax[0, col].set_title(title)
        ax[1, col].set_ylim(0, 105)
        for a in ax[:, col]:
            a.set_xlim(0, kmax)
            a.set_xlabel("RL training steps (k)")
            a.grid(color=MUTED, alpha=0.35, lw=0.6)
    ax[0, 0].set_ylabel("training episode return\n(trailing 100-ep mean, each system's reward)")
    ax[1, 0].set_ylabel("deployed reach rate (% of eval set, deterministic)")
    fig.suptitle("Warm-start RL with vs without DAWN on all three systems (soft SAC, seed 0). "
                 "Every curve starts at step 0 from the squash clone acting alone;\nDAWN's 20k "
                 "warmup transitions precede step 0 and are not on the axis.", fontsize=11)
    # One legend for the whole figure: the same three variants in every panel.
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3,
               fontsize=9.5, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Panda RL with vs without DAWN: training return and deployed reach over training steps.

    # score every checkpoint on the 78 box scenarios (same harness as train_panda_rl.py's final eval),
    # one results file per checkpoint, then concatenate them into --sweep
    uv run python scripts/train/train_panda_rl.py --arm residual --clone data/panda_clone_box_r2.pt \
        --eval <ckpt.zip> --method <ckpt.zip> --results <one-file-per-ckpt.csv>
    uv run python scripts/plot/plot_panda_dawn.py --arm residual     # or --arm warm

Both panels start at step 0 from the arm's prior acting alone (deterministic, 78
box scenarios, from data/panda_results.csv): the base clone for the residual, the
squash clone for the warm start. For the return that value is episode 0 of the
trailing window (`rl.curves`). DAWN's 20k warmup transitions are collected before
step 0 and are not on this axis.
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

# arm -> (prior's method in panda_results.csv, steps, title, [(label, run, color, lw)])
ARMS = {
    "residual": ("panda_clone_box_r2", 150, "Panda residual RL (residual_frac 1, soft SAC, seed 0)", [
        ("pure residual RL", "panda_res_base_150k", "#c92a2a", 2.6),
        ("+ DAWN warmup data (20k base-clone transitions)", "panda_res_dawn_150k", "#1c7ed6", 2.6),
        ("+ full DAWN (warmup + LayerNorm critic)", "panda_res_dawnln_150k", "#7b2cbf", 1.4)]),
    "warm": ("panda_clone_box_r2_squash", 200, "Panda warm-start RL (soft SAC, seed 0)", [
        ("pure warm-start RL", "panda_warm_base_200k", "#c92a2a", 2.6),
        ("+ DAWN warmup data (20k clone transitions)", "panda_warm_dawn_200k", "#1c7ed6", 2.6),
        ("+ full DAWN (warmup + LayerNorm critic)", "panda_warm_dawnln_200k", "#7b2cbf", 1.4)]),
}


def per_method(path: str):
    reached, ret = collections.Counter(), collections.defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            reached[r["method"]] += r["reached"] == "True"
            ret[r["method"]].append(float(r["ret"]))
    return reached, {m: float(np.mean(v)) for m, v in ret.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=sorted(ARMS), default="residual")
    p.add_argument("--sweep", default=None, help="default data/panda_dawn_sweep[_warm].csv")
    p.add_argument("--results", default="data/panda_results.csv")
    p.add_argument("--out", default=None, help="default docs/journey/figures/panda_dawn[_warm].png")
    args = p.parse_args()
    sfx = "" if args.arm == "residual" else f"_{args.arm}"
    base, kmax, title, runs = ARMS[args.arm]

    sweep, _ = per_method(args.sweep or f"data/panda_dawn_sweep{sfx}.csv")
    base_reach, base_ret = per_method(args.results)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for lab, run, col, lw in runs:
        s, _ep, r = return_curve(f"data/{run}.monitor.csv", step0=base_ret[base])
        ax[0].plot(s / 1e3, r, color=col, lw=lw, label=lab)
        pts = sorted((int(re.findall(r"ckpt_(\d+)_steps", m)[0]), k)
                     for m, k in sweep.items() if f"/{run}_ckpt/" in m)
        x = np.array([0, *[q for q, _ in pts]]) / 1e3
        y = 100 * np.array([base_reach[base], *[k for _, k in pts]]) / 78
        ax[1].plot(x, y, color=col, lw=lw, marker="o", ms=3, label=f"{lab}, lowest {y.min():.0f}%")
    clone = 100 * base_reach[base] / 78
    ax[1].axhline(clone, color="k", ls=":", lw=1)
    ax[1].text(kmax * 2 / 3, clone + 2, f"the clone: {clone:.0f}%", fontsize=8)
    ax[0].axhline(base_ret[base], color="k", ls=":", lw=1)
    ax[0].text(kmax * 2 / 3, base_ret[base] + 3, f"the clone: {base_ret[base]:.0f}", fontsize=8)
    for a in ax:
        a.set_xlim(0, kmax)
        a.set_xlabel("RL training steps (k)")
        a.grid(alpha=0.3)
    ax[0].set(ylabel="training episode return (trailing 100-ep mean)",
              title="Training return (exploring policy)")
    ax[1].set(ylabel="reached (% of 78 box scenarios, deterministic)", ylim=(0, 105),
              title="Deployed reach rate (checkpoint every 5k steps)")
    ax[1].legend(fontsize=8.5, loc="lower right")
    fig.suptitle(f"{title}: with vs without DAWN. All curves start at step 0 from the clone acting "
                 "alone.\nDAWN's 20k warmup transitions happen before step 0 and are not on this axis.",
                 fontsize=10)
    fig.tight_layout()
    out = args.out or f"docs/journey/figures/panda_dawn{sfx}.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

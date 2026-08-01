#!/usr/bin/env python
"""Deterministic reach rate vs environment steps, for the residual and vanilla arms.

The training-return curve (`plot_training_return.py`) measures the *behaviour* policy:
it includes exploration noise and scores whatever start states training happened to
sample. This script measures the *deployed* policy instead — greedy actions, the
canonical 78-seed evaluation sweep — at intermediate training checkpoints, which is
the sample-efficiency claim in the form that matters for deployment.

Consumes the checkpoints written by `train_residual.py --checkpoint-dir` /
`train_vanilla.py --checkpoint-dir`, laid out one directory per training seed:

    data/ckptsweep/res_s0/ckpt_25000_steps.zip
    data/ckptsweep/van_s0/ckpt_25000_steps.zip

    uv run python scripts/sweep_checkpoints.py --ckpt-root data/ckptsweep

Writes one CSV row per arm x training seed x checkpoint. DeePC and the clone are not
run: both are deterministic and training-step-independent, so they are flat reference
lines the caller can splice in from `docs/journey/figures/reach_rates.csv`.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from two_wheel_robot.rl.deepc_setup import canonical_action_bounds
from two_wheel_robot.rl.residual_env import ResidualDeePCEnv
from two_wheel_robot.rl.residual_eval import (
    episode_return,
    run_residual_closed_loop_with_actions,
    run_vanilla_closed_loop_with_actions,
)
from two_wheel_robot.rl.train_sb3 import load_residual
from two_wheel_robot.rl.wrappers import vanilla_rl_env


def _ckpt_path(root: str, arm_prefix: str, train_seed: int, steps: int) -> str | None:
    """`<root>/<prefix>_s<seed>/ckpt_<steps>_steps.zip`, or the run's final .zip.

    The final save and the last periodic checkpoint are the same policy, but a run
    trained to a total that is not a multiple of the checkpoint frequency only has
    the former — hence the fallback.
    """
    for path in (os.path.join(root, f"{arm_prefix}_s{train_seed}", f"ckpt_{steps}_steps.zip"),
                 os.path.join(root, f"{arm_prefix}_s{train_seed}.zip")):
        if os.path.exists(path):
            return path
    return None


def _evaluate(kind: str, model, env, seeds) -> dict:
    """Greedy rollout on every eval seed.

    Returns the reach count plus the per-episode means of every term the DeePC-form
    cost splits into. The split is what makes reach rate interpretable: arms with the
    same reach rate can differ several-fold on heading and control cost, and episode
    lengths differ enough between arms that per-episode sums alone mislead (divide by
    `steps_mean` for the per-step figure).
    """
    reach = 0
    eps = []
    for s in seeds:
        if kind == "residual":
            reached, traj, acts = run_residual_closed_loop_with_actions(model, env, s)
            goal = env.base.goal.copy()
        else:
            reached, traj, acts = run_vanilla_closed_loop_with_actions(model, env, s)
            goal = env.unwrapped.goal.copy()
        reach += int(reached)
        eps.append(episode_return(traj, acts, goal))
    mean = {k: float(np.mean([e[k] for e in eps]))
            for k in ("total", "steps", "position_cost", "heading_cost", "control_cost")}
    return {
        "reach": reach,
        "return_mean": mean["total"],
        "steps_mean": mean["steps"],
        "position_cost_mean": mean["position_cost"],
        "heading_cost_mean": mean["heading_cost"],
        "control_cost_mean": mean["control_cost"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-root", default="data/ckptsweep")
    p.add_argument("--steps", default="25000,50000,100000,200000,400000",
                   help="comma-separated checkpoint step counts to evaluate")
    p.add_argument("--train-seeds", default="0,1,2,3,4")
    p.add_argument("--arms", default="residual,vanilla",
                   help="which arms to sweep; --algo applies to all of them, so a "
                        "SAC arm is a separate invocation with --arms residual")
    p.add_argument("--label", default=None,
                   help="name written to the CSV's arm column (default: the arm), "
                        "e.g. residual_sac to merge with a TD3 sweep")
    p.add_argument("--clone", default="data/clone.pt")
    p.add_argument("--libraries", default="data/libraries_v0.npz")
    p.add_argument("--residual-frac", type=float, default=2.0)
    p.add_argument("--algo", default="td3", choices=["td3", "sac"])
    p.add_argument("--n_seeds", type=int, default=78)
    p.add_argument("--base_seed", type=int, default=4104626029)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="data/checkpoint_sweep.csv")
    args = p.parse_args()

    steps = [int(x) for x in args.steps.split(",")]
    train_seeds = [int(x) for x in args.train_seeds.split(",")]
    eval_seeds = [args.base_seed + i for i in range(args.n_seeds)]

    # One env per arm, built only if that arm is swept and reused across every
    # checkpoint -- constructing ResidualDeePCEnv reloads the clone and the Hankel
    # libraries, which dwarfs a 78-episode rollout.
    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    envs = {}
    if "residual" in wanted:
        envs["residual"] = ResidualDeePCEnv(
            clone_path=args.clone, libraries_path=args.libraries,
            residual_frac=args.residual_frac, device=args.device,
        )
    if "vanilla" in wanted:
        envs["vanilla"] = vanilla_rl_env(canonical_action_bounds(args.libraries))
    rows = []
    try:
        for kind, prefix in (("residual", "res"), ("vanilla", "van")):
            if kind not in envs:
                continue
            arm = args.label or kind
            for ts in train_seeds:
                for n in steps:
                    path = _ckpt_path(args.ckpt_root, prefix, ts, n)
                    if path is None:
                        print(f"  !! missing {arm} seed {ts} @ {n} steps -- skipped")
                        continue
                    t0 = time.time()
                    model = load_residual(path, algo=args.algo, device=args.device)
                    r = _evaluate(kind, model, envs[kind], eval_seeds)
                    rows.append({
                        "arm": arm, "train_seed": ts, "steps": n,
                        "n": args.n_seeds, "reach": r["reach"],
                        "reach_rate": r["reach"] / args.n_seeds,
                        **{k: r[k] for k in ("return_mean", "steps_mean",
                                             "position_cost_mean", "heading_cost_mean",
                                             "control_cost_mean")},
                    })
                    print(f"  {arm:<13} seed {ts}  {n:>7} steps  "
                          f"reach {r['reach']:>3}/{args.n_seeds} "
                          f"({r['reach'] / args.n_seeds:.3f})  "
                          f"return {r['return_mean']:>9.1f}  [{time.time() - t0:.0f}s]")
    finally:
        for env in envs.values():
            env.close()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

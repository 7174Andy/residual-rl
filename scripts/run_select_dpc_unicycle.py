"""Select-DPC vs fixed heading-keyed libraries on the unicycle: closed-loop reach.

The unicycle counterpart of `run_select_dpc_reacher.py` (journey 12). Both
controllers use the canonical config and the SAME data -- the four libraries'
trajectories -- so the only difference is per-step column selection versus one
whole library picked by heading. Scored with `clone_eval.run_deepc_closed_loop`
on the canonical 78 eval seeds, the harness behind DeePC's 30/78.

    OMP_NUM_THREADS=1 uv run python scripts/run_select_dpc_unicycle.py
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import gymnasium as gym
import numpy as np

from two_wheel_robot.rl.clone_eval import run_deepc_closed_loop
from two_wheel_robot.rl.deepc_setup import build_canonical_deepc, build_canonical_select

_W: dict = {}


def _init(kind: str, libraries: str, n_cols: int) -> None:
    if kind == "fixed":
        ctrl, info = build_canonical_deepc(libraries)
    else:
        ctrl, info = build_canonical_select(libraries, n_cols=n_cols,
                                            carry_prediction=kind == "select_carry")
    env = gym.make("TwoWheelGoal-v0", action_bounds=info["action_bounds"])
    _W.update(ctrl=ctrl, info=info, env=env)


def _episode(seed: int) -> tuple[bool, float]:
    reached, traj = run_deepc_closed_loop(_W["ctrl"], _W["info"], _W["env"], seed)
    goal = _W["env"].unwrapped.goal
    return reached, float(np.min(np.linalg.norm(traj[:, :2] - goal, axis=1)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--libraries", default="data/libraries_v0.npz")
    p.add_argument("--kinds", default="fixed,select,select_carry")
    p.add_argument("--n-cols", type=int, default=300)
    p.add_argument("--n-seeds", type=int, default=78)
    p.add_argument("--base-seed", type=int, default=4104626029)
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()

    seeds = [args.base_seed + i for i in range(args.n_seeds)]
    res = {}
    for kind in args.kinds.split(","):
        with mp.Pool(args.workers, _init, (kind, args.libraries, args.n_cols)) as pool:
            res[kind] = pool.map(_episode, seeds)
        hit = np.array([r for r, _ in res[kind]])
        best = np.array([d for _, d in res[kind]])
        print(f"{kind:<13} reach {hit.sum():>2}/{args.n_seeds}   "
              f"closest approach median {np.median(best):.2f}", flush=True)
    if "fixed" in res:
        base = np.array([r for r, _ in res["fixed"]])
        for kind in res:
            if kind != "fixed":
                hit = np.array([r for r, _ in res[kind]])
                print(f"  {kind} vs fixed, paired: gains {np.sum(hit & ~base)}, "
                      f"loses {np.sum(base & ~hit)}")


if __name__ == "__main__":
    main()

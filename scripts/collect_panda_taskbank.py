"""Collect the goal-directed task bank (Phase 1 lever 1).

    uv run python scripts/collect_panda_taskbank.py --out data/panda_taskbank_v1.npz --n-traj 1000 --T 150 --seed 0

Writes TWO files: the primary bank (u_i = delta, for panda/deepc_setup.py)
and a `_sdpc` companion (u_i = ctrl, via panda.task_bank.for_select_dpc) for
disk-path Select-DPC consumers (scripts/run_select_dpc.py --libs,
scripts/measure_selection_distance.py --libs) -- see panda/task_bank.py's
module docstring for why one file cannot serve both.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from panda.env import PandaReachEnv
from panda.task_bank import collect_task_bank, for_select_dpc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/panda_taskbank_v1.npz")
    p.add_argument("--n-traj", type=int, default=1000)
    p.add_argument("--T", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--noise-sigma", type=float, default=0.04)
    args = p.parse_args()

    env = PandaReachEnv()
    try:
        bank = collect_task_bank(env, args.n_traj, args.T, args.seed,
                                  args.alpha, args.noise_sigma)
    finally:
        env.close()
    np.savez(args.out, **bank)
    print(f"wrote {args.out}: {args.n_traj} trajectories x {args.T} steps")

    stem, ext = os.path.splitext(args.out)
    sdpc_out = f"{stem}_sdpc{ext}"
    np.savez(sdpc_out, **for_select_dpc(bank))
    print(f"wrote {sdpc_out}: Select-DPC-compatible companion (u_i = ctrl)")


if __name__ == "__main__":
    main()

"""Freeze a Reacher scenario set: start configuration + goal, both attainable.

Journey 12's numbers were generated from `seed 0` at run time. Freezing the draw
makes every row of the residual-RL table comparable across sessions, and applying
`is_reachable` removes the ~2.1% of disc draws the arm physically cannot attain
at SAFE_MARGIN = 0.02 -- an invisible ceiling on every reach rate otherwise.

    uv run python scripts/make_reacher_scenarios.py --n 120
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from reacher.model import (
    is_reachable, load_model, sample_config, sample_goal,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=120, help="journey 12 used 120")
    p.add_argument("--min-need", type=float, default=0.02,
                   help="reject starts already within this of the goal (m)")
    p.add_argument("--out", default="data/reacher_scenarios_v1.npz")
    # MUST be disjoint from every consumer's stream. `ReacherGoalEnv.reset` draws
    # goal-then-config from `self.np_random`, exactly as this script does, so a
    # scenario file built at seed 0 is BIT-IDENTICAL to training episodes 0..n-1
    # of any env seeded 0 -- verified, it was an identity mapping across all 40.
    # SB3 seeds once and then auto-resets unseeded, so training walks that same
    # stream and the evaluation set lands wholly inside it.
    p.add_argument("--seed", type=int, default=90_000,
                   help="held out from RL/collection seeds (which use 0)")
    args = p.parse_args()

    model, data = load_model()
    rng = np.random.default_rng(args.seed)
    qpos, goal, need = [], [], []
    rejected_goal = rejected_need = 0

    while len(qpos) < args.n:
        g = sample_goal(rng)
        if not is_reachable(model, g):
            rejected_goal += 1
            continue
        q0, tip = sample_config(model, data, rng)
        d = float(np.linalg.norm(tip - g))
        if d < args.min_need:
            rejected_need += 1
            continue
        qpos.append(q0)
        goal.append(g)
        need.append(d)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, qpos=np.array(qpos), goal=np.array(goal),
             need=np.array(need), seed=np.asarray(args.seed),
             min_need=np.asarray(args.min_need))
    print(f"wrote {args.out}: {len(qpos)} scenarios (seed {args.seed})")
    print(f"  rejected {rejected_goal} unreachable goals "
          f"({100 * rejected_goal / (rejected_goal + len(qpos)):.1f}%), "
          f"{rejected_need} starts already inside {args.min_need} m")
    print(f"  need: median {np.median(need) * 1e3:.0f} mm, "
          f"max {np.max(need) * 1e3:.0f} mm")


if __name__ == "__main__":
    main()

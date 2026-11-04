"""Collect the DeePC libraries for QuadrupedGoal-v0.

    .venv/bin/python scripts/collect_quadruped_data.py [--T 1500 --n-anchors 4 --hold-s 0.5 --seed 0 --out data/quadruped_libraries_v0.npz]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from quadruped.data_collection import HOLD_S, PAPER_T, collect_libraries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=PAPER_T)
    p.add_argument("--n-anchors", type=int, default=4)
    p.add_argument("--hold-s", type=float, default=HOLD_S)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/quadruped_libraries_v0.npz")
    a = p.parse_args()
    d = collect_libraries(a.T, a.n_anchors, a.seed, hold_s=a.hold_s)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **d)
    for i in range(a.n_anchors):
        u, y = d[f"u_{i}"], d[f"y_{i}"]
        print(f"lib {i}: anchor={d['anchors'][i]:+.3f} seed={d['seeds'][i]} retries={d['retries'][i]} "
              f"n_wraps={d['n_wraps'][i]} yaw=[{y[:, 2].min():+.2f}, {y[:, 2].max():+.2f}]")
        print(f"  u mean={np.round(u.mean(0), 3)} std={np.round(u.std(0), 3)}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

"""Stability + rough-tracking check for the Go1 trot over the command grid.

    .venv/bin/python scripts/check_go1_gait.py [--period 0.4 --lift 0.07 ...]

Ends with FLOOR PASS / FLOOR FAIL (exit 1): passes only when every grid point
inside `checks.FLOOR` neither falls nor fails tracking, and every random-command
seed survives.
"""
from __future__ import annotations

import argparse
import inspect
import sys

import numpy as np

from quadruped.checks import FLOOR, GRID, run_held, run_random, tracking_ok
from quadruped.gait import TrotGait

GAIT_DEFAULTS = {k: v.default for k, v in inspect.signature(TrotGait).parameters.items()}


def in_floor(cmd) -> bool:
    return all(lo <= c <= hi for c, (lo, hi) in zip(cmd, FLOOR.values()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--random-seeds", type=int, default=3)
    ap.add_argument("--period", type=float, default=GAIT_DEFAULTS["period"])
    ap.add_argument("--lift", type=float, default=GAIT_DEFAULTS["lift"])
    ap.add_argument("--height", type=float, default=GAIT_DEFAULTS["height"])
    ap.add_argument("--k-foot", type=float, default=GAIT_DEFAULTS["k_foot"])
    a = ap.parse_args()
    kw = {"period": a.period, "lift": a.lift, "height": a.height, "k_foot": a.k_foot}
    print("gait:", kw)

    ok = True
    print(f"{'cmd (vx, vy, wz)':>22}  floor  {'fell':>10}  {'v_mean (vx, vy, wz)':>24}  track")
    for cmd in GRID:
        r = run_held(np.array(cmd), seconds=a.seconds, **kw)
        trk = not r["fell"] and tracking_ok(cmd, r["v_mean"])
        fl = in_floor(cmd)
        ok &= trk or not fl
        fell = f"t={r['t_fall']:.2f}" if r["fell"] else "no"
        v = "  ".join(f"{x:+.3f}" for x in r["v_mean"])
        print(f"({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})  {'in ' if fl else 'out'}"
              f"    {fell:>10}  {v:>24}  {'ok' if trk else 'FAIL'}")

    for seed in range(a.random_seeds):
        r = run_random(np.random.default_rng(seed), **kw)
        ok &= not r["fell"]
        print(f"random seed {seed}: " + (f"FELL t={r['t_fall']:.2f}" if r["fell"] else "survived 20 s"))

    print("FLOOR PASS" if ok else "FLOOR FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Offline (u, y) libraries for DeePC on QuadrupedGoal-v0, one per starting heading.

`y_t` is read before `u_t` is applied (same convention as the unicycle's
`two_wheel_robot/controllers/data_collection.py`, whose PAPER_T and anchor headings
are copied here). Excitation mirrors `quadruped.checks.run_random`: a uniform command
held `HOLD_S` seconds plus per-step Gaussian noise, clipped to `SAMPLE_BOUNDS`.
Commands that change every step make the gait fall. A fallen attempt is discarded
and retried with the next seed. Yaw is unwrapped; `n_wraps` counts the raw jumps.

Collected with (output below is from this command):

    .venv/bin/python scripts/collect_quadruped_data.py

    lib 0: anchor=+0.785 seed=0 retries=0 n_wraps=0 yaw=[-0.28, +1.16]
      u mean=[ 0.221 -0.    -0.013] std=[0.232 0.119 0.449]
    lib 1: anchor=+2.356 seed=1 retries=0 n_wraps=0 yaw=[-1.37, +2.32]
      u mean=[ 0.156 -0.016 -0.059] std=[0.256 0.116 0.503]
    lib 2: anchor=-2.356 seed=2 retries=0 n_wraps=0 yaw=[-3.03, -1.40]
      u mean=[ 0.19  -0.02  -0.001] std=[0.266 0.131 0.474]
    lib 3: anchor=-0.785 seed=3 retries=0 n_wraps=0 yaw=[-0.91, +0.78]
      u mean=[ 0.144 -0.001 -0.011] std=[0.25  0.11  0.451]
    wrote data/quadruped_libraries_v0.npz

The excitation does not hold yaw near the anchor. Share of samples within pi/4 of
their own anchor: lib 0 0.95, lib 1 0.16 (drifts toward 0; nothing above yaw 2.32),
lib 2 0.98, lib 3 0.73. Re-measure: `.venv/bin/python scripts/quadruped_library_coverage.py`.
Goal bearings in lib 1's region are where the canonical controller fails (journey 21).
"""
from __future__ import annotations

import numpy as np

from quadruped.env import QuadrupedGoalEnv

SAMPLE_BOUNDS = np.array([[-0.25, 0.6], [-0.2, 0.2], [-0.8, 0.8]])  # = env action bounds
PAPER_T = 1500  # unicycle data_collection.PAPER_T
HOLD_S = 0.5
NOISE = (0.05, 0.05, 0.1)
FAR_GOAL = (100.0, 100.0)  # never reached, so the env never terminates by reaching


def anchor_headings(n: int = 4) -> np.ndarray:
    """(2k+1)pi/n wrapped to [-pi, pi]; n=4 equals the unicycle's PAPER_INIT_HEADINGS."""
    return np.angle(np.exp(1j * (2 * np.arange(n) + 1) * np.pi / n))


def collect_trajectory(T, anchor, rng, *, hold_s=HOLD_S, noise=NOISE, bounds=SAMPLE_BOUNDS):
    """Returns (u (T,3), y (T,3), fell, n_wraps); on a fall the arrays stop at the fall step."""
    env = QuadrupedGoalEnv(max_steps=T)
    env.reset(options={"state": (0.0, 0.0, anchor), "goal": FAR_GOAL})
    lo, hi = bounds[:, 0], bounds[:, 1]
    n_hold = round(hold_s / env.dt)
    u, y, fell = [], [], False
    for k in range(T):
        if k % n_hold == 0:
            base = rng.uniform(lo, hi)
        y.append(env.y)
        _, _, _, _, info = env.step(np.clip(base + rng.normal(0.0, noise), lo, hi))
        u.append(env.last_action.copy())
        if info["fell"]:
            fell = True
            break
    u, y = np.array(u), np.array(y)
    n_wraps = int(np.sum(np.abs(np.diff(y[:, 2])) > np.pi))
    y[:, 2] = np.unwrap(y[:, 2])
    return u, y, fell, n_wraps


def collect_libraries(T=PAPER_T, n_anchors=4, seed=0, *, hold_s=HOLD_S, noise=NOISE, max_retries=20):
    """npz-ready dict. One seed counter starts at `seed`, +1 after every attempt."""
    out = {}
    seeds, retries, wraps = [], [], []
    counter = seed
    for i, anchor in enumerate(anchor_headings(n_anchors)):
        for r in range(max_retries + 1):
            s, counter = counter, counter + 1
            u, y, fell, nw = collect_trajectory(
                T, anchor, np.random.default_rng(s), hold_s=hold_s, noise=noise)
            if not fell:
                break
        else:
            raise RuntimeError(f"library {i}: {max_retries + 1} falls (last seed {s})")
        out[f"u_{i}"], out[f"y_{i}"] = u, y
        seeds.append(s), retries.append(r), wraps.append(nw)
    out.update(sample_bounds=SAMPLE_BOUNDS, anchors=anchor_headings(n_anchors),
               seeds=np.array(seeds), retries=np.array(retries), n_wraps=np.array(wraps),
               hold_s=np.array(hold_s), noise=np.array(noise), T=np.array(T))
    return out

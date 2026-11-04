"""Stability and rough-tracking checks for the Go1 trot (spec's "Done" 2, 3, 3b).

Every run starts from the `home` keyframe and settles for `SETTLE_S` at zero
command before the test command starts. Times (`t_fall`) are seconds since reset.
"""
from __future__ import annotations

from itertools import chain, repeat

import numpy as np

from quadruped.gait import TrotGait
from quadruped.model import Go1Sim

SETTLE_S = 0.5
TRACK_MIN = np.array([0.15, 0.15, 0.3])  # |cmd| below this on an axis -> axis not checked
TRACK_TOL = 0.5                          # |v - c| <= TRACK_TOL * |c|
FLOOR = {"vx": (0.0, 0.6), "vy": (-0.2, 0.2), "yaw_rate": (-0.8, 0.8)}
GRID = (
    [(vx, 0.0, 0.0) for vx in (-0.5, -0.25, 0.0, 0.25, 0.5, 0.6, 0.75, 1.0)]
    + [(0.0, vy, 0.0) for vy in (0.2, -0.2, 0.4, -0.4)]
    + [(0.0, 0.0, wz) for wz in (0.4, -0.4, 0.8, -0.8, 1.5, -1.5)]
    + [(0.5, 0.2, 0.0), (0.5, 0.0, 0.8), (0.3, -0.2, -0.5)]
)


def _start(gait_kw):
    sim, gait = Go1Sim(), TrotGait(**gait_kw)
    sim.reset()
    gait.reset()
    return sim, gait


def _run(sim, gait, cmds):
    """Settle, then step through `cmds`. Yields (t, past_settle, fell) after each step."""
    n_settle = round(SETTLE_S / gait.dt)
    for k, cmd in enumerate(chain(repeat(np.zeros(3), n_settle), cmds)):
        sim.step(gait.step(cmd, v_meas=sim.body_vel()[:2]))
        yield (k + 1) * gait.dt, k >= n_settle, sim.fell()


def run_held(cmd, seconds=10.0, **gait_kw) -> dict:
    sim, gait = _start(gait_kw)
    cmd = np.asarray(cmd, dtype=float)
    n = round(seconds / gait.dt)
    t_avg = SETTLE_S + seconds - 5.0
    vs = []
    for t, live, fell in _run(sim, gait, repeat(cmd, n)):
        if fell:
            return {"fell": True, "t_fall": t, "v_mean": np.full(3, np.nan)}
        if live and t > t_avg:
            vs.append(sim.body_vel())
    return {"fell": False, "t_fall": None, "v_mean": np.mean(vs, axis=0)}


def tracking_ok(cmd, v_mean) -> bool:
    cmd, v = np.asarray(cmd, dtype=float), np.asarray(v_mean, dtype=float)
    chk = np.abs(cmd) >= TRACK_MIN
    c, v = cmd[chk], v[chk]
    return bool(np.all((np.sign(v) == np.sign(c)) & (np.abs(v - c) <= TRACK_TOL * np.abs(c))))


def run_random(rng: np.random.Generator, seconds=20.0, ranges=FLOOR, hold_s=0.5,
               noise=(0.05, 0.05, 0.1), **gait_kw) -> dict:
    sim, gait = _start(gait_kw)
    lo = np.array([ranges[k][0] for k in ("vx", "vy", "yaw_rate")])
    hi = np.array([ranges[k][1] for k in ("vx", "vy", "yaw_rate")])
    n, n_hold = round(seconds / gait.dt), round(hold_s / gait.dt)

    def cmds():
        for k in range(n):
            if k % n_hold == 0:
                base = rng.uniform(lo, hi)
            yield np.clip(base + rng.normal(0.0, noise), lo, hi)

    for t, _, fell in _run(sim, gait, cmds()):
        if fell:
            return {"fell": True, "t_fall": t}
    return {"fell": False, "t_fall": None}

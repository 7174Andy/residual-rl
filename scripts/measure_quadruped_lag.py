"""Measure the Go1 trot's command -> velocity step response (time to 63%) per axis.

Settle, hold c0 for --hold-s, step to c1 for --hold-s. The measured axis velocity is
smoothed with a trailing 20-step (one gait cycle) moving average, which itself adds
~10 steps (0.2 s) of delay, so both the raw and an approximately corrected lag (raw - 0.2 s,
+-0.05 s depending on the response shape) are printed.
"""
from __future__ import annotations

import argparse
from itertools import chain, repeat

import numpy as np

from quadruped.checks import SETTLE_S, _run, _start

AXES = {"vx": 0, "vy": 1, "yaw": 2}
STEPS = [("vx", 0.0, 0.3), ("vx", 0.3, 0.6), ("vx", 0.0, -0.2),
         ("vy", 0.0, 0.2), ("vy", 0.0, -0.2),
         ("yaw", 0.0, 0.8), ("yaw", 0.0, -0.8), ("yaw", 0.4, -0.4)]
MA = 20          # one gait cycle
MA_DELAY = MA // 2  # steps of delay the trailing average adds


def smooth(v, n=MA):
    c = np.cumsum(np.insert(v, 0, 0.0))
    k = np.minimum(np.arange(1, len(v) + 1), n)
    return (c[1:] - c[np.maximum(np.arange(1, len(v) + 1) - n, 0)]) / k


def lag_63(vs, step_idx, dt, tail_s=0.5):
    """vs: smoothed series; the command changes at index step_idx.

    Returns (lag_s, baseline, final); lag_s is nan if 63% is never reached.
    """
    n = round(tail_s / dt)
    base, final = vs[step_idx - n:step_idx].mean(), vs[-n:].mean()
    thr = base + 0.63 * (final - base)
    after = vs[step_idx:]
    hit = np.nonzero(after >= thr if final > base else after <= thr)[0]
    return (hit[0] * dt if len(hit) else np.nan), base, final


def measure(axis, c0, c1, hold_s):
    sim, gait = _start({})
    n = round(hold_s / gait.dt)
    cmd = lambda c: np.eye(3)[AXES[axis]] * c  # noqa: E731
    v = []
    for _, _, fell in _run(sim, gait, chain(repeat(cmd(c0), n), repeat(cmd(c1), n))):
        if fell:
            return None
        v.append(sim.body_vel()[AXES[axis]])
    return smooth(np.array(v)), round(SETTLE_S / gait.dt) + n, gait.dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hold-s", type=float, default=2.0)
    hold_s = ap.parse_args().hold_s
    print(f"{'axis':<5}{'c0':>6}{'c1':>7}{'raw63_s':>9}{'corr63_s(-0.2s MA)':>20}{'final':>8}{'gain':>7}")
    for axis, c0, c1 in STEPS:
        m = measure(axis, c0, c1, hold_s)
        if m is None:
            print(f"{axis:<5}{c0:>6}{c1:>7}  fell")
            continue
        vs, idx, dt = m
        lag, _, final = lag_63(vs, idx, dt)
        print(f"{axis:<5}{c0:>6}{c1:>7}{lag:>9.2f}{lag - MA_DELAY * dt:>20.2f}{final:>8.3f}{final / c1:>7.2f}")


if __name__ == "__main__":
    main()

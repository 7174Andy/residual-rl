"""Training-return curves from SB3 Monitor CSVs, shared by every plot script.

A plain `np.convolve(..., mode="valid")` over 100 episodes first plots once
100 episodes exist: 5k steps on Reacher, ~15k on the Panda and the unicycle.
On the warm-start arms the collapse happens inside that blind spot, so the
curve starts already low and the dip never appears. The trailing mean here
averages over the episodes seen so far until `window` have accumulated, and
starts at `min_episodes`: the very first episode can be mostly SB3's
`learning_starts` random actions (the Panda's scores ~-210) and would set the
axis on its own.
"""
from __future__ import annotations

import csv
import io

import numpy as np


def read_monitor(path: str) -> tuple[np.ndarray, np.ndarray]:
    """(per-episode return, per-episode length) from one SB3 Monitor CSV."""
    with open(path) as f:
        body = "".join(ln for ln in f if not ln.startswith("#"))
    rows = list(csv.DictReader(io.StringIO(body)))
    return (np.asarray([float(x["r"]) for x in rows]),
            np.asarray([int(x["l"]) for x in rows]))


def trailing_mean(r: np.ndarray, window: int = 100, min_episodes: int = 10) -> np.ndarray:
    """Mean of the last `window` episodes (fewer while fewer exist), from episode `min_episodes` on."""
    c = np.concatenate([[0.0], np.cumsum(r)])
    i = np.arange(1, len(r) + 1)
    lo = np.maximum(i - window, 0)
    return ((c[i] - c[lo]) / (i - lo))[min_episodes - 1:]


def return_curve(path: str, window: int = 100, min_episodes: int = 10, step0: float | None = None):
    """(cumulative env steps, episode index, trailing-mean return), aligned, from `min_episodes` on.

    `step0`: the prior's deterministic return at step 0 -- the same policy the
    reach rate scores at step 0. It is plotted at step 0 and counted as episode 0
    of the trailing window (weight 1/window once the window is full), so the
    return curve starts where the reach-rate curve starts.
    """
    r, length = read_monitor(path)
    steps = np.cumsum(length)[min_episodes - 1:]
    eps = np.arange(min_episodes, len(r) + 1)
    if step0 is None:
        return steps, eps, trailing_mean(r, window, min_episodes)
    m = trailing_mean(np.concatenate([[step0], r]), window, 1)     # m[k] = mean through episode k
    return (np.concatenate([[0], steps]), np.concatenate([[0], eps]),
            np.concatenate([[m[0]], m[min_episodes:]]))


if __name__ == "__main__":
    r = np.arange(1.0, 201.0)
    m = trailing_mean(r, window=100, min_episodes=1)
    assert m[0] == 1.0 and m[9] == 5.5                  # partial windows at the start
    assert m[99] == 50.5 and m[199] == 150.5            # full 100-episode window after
    assert len(trailing_mean(r, 100, 10)) == 191        # starts at the 10th episode
    m0 = trailing_mean(np.concatenate([[0.0], r[:20]]), 100, 1)
    assert m0[0] == 0.0 and m0[10] == 5.0               # step 0 counts as episode 0 of the window
    print("ok")

"""The frozen evaluation scenario set for QuadrupedGoal-v0.

Every controller (DeePC, clone, residual) must be scored on identical episodes.
`env.reset(seed=k)` maps a seed to a scenario through the env's sampling code,
so any change there silently remaps every seed. So scenarios are generated once,
written to a versioned file, and replayed via `reset(options={"state","goal"})`.
Collection may use seeds freely; evaluation must not.

118 scenarios: ids 0-39 for tuning, 40-117 for evaluation (disjoint). `state`
is the PRE-settle pose request (the env settles after placing it).
Mirrors panda/scenarios.py.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

SCENARIOS_PATH = "data/quadruped_scenarios_v1.npz"

N_TUNE = 40
N_EVAL = 78
TUNE_IDS = range(N_TUNE)
EVAL_IDS = range(N_TUNE, N_TUNE + N_EVAL)

_ENV_PARAM_KEYS = ("goal_tolerance", "min_start_goal_dist", "max_steps", "dt", "settle_s")
_ENV_PARAM_INT_KEYS = frozenset({"max_steps"})


def _env_param(env, k):
    if k == "settle_s":  # not an env attribute; the env settles for this constant
        from quadruped.checks import SETTLE_S
        return SETTLE_S
    return getattr(env, k)


def generate(n: int = N_TUNE + N_EVAL) -> dict:
    """Sample `n` scenarios via `reset(seed=i)`, i in range(n), then freeze them."""
    from quadruped.env import QuadrupedGoalEnv

    env = QuadrupedGoalEnv()
    try:
        state = np.empty((n, 3), dtype=np.float64)
        goal = np.empty((n, 2), dtype=np.float64)
        for i in range(n):
            env.reset(seed=i)
            state[i] = env.start_request
            goal[i] = env.goal
        out = {"state": state, "goal": goal, "seed": np.arange(n, dtype=np.int64)}
        for k in _ENV_PARAM_KEYS:
            v = _env_param(env, k)
            out[k] = np.int64(v) if k in _ENV_PARAM_INT_KEYS else np.float64(v)
    finally:
        env.close()
    return out


def save(path: str, data: dict) -> None:
    np.savez(path, **data)


def load(path: str = SCENARIOS_PATH) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def reset_to(env, scenarios: dict, i: int):
    return env.reset(options={"state": scenarios["state"][i], "goal": scenarios["goal"][i]})


def checksum(scenarios: dict) -> str:
    h = hashlib.sha256()
    for k in ("state", "goal"):
        h.update(np.ascontiguousarray(scenarios[k], dtype=np.float64).tobytes())
    return h.hexdigest()


def validate_against_env(env, scenarios: dict) -> None:
    """Raise if env config disagrees with what the scenarios recorded."""
    bad = []
    for k in _ENV_PARAM_KEYS:
        rec, cur = scenarios[k], _env_param(env, k)
        ok = int(rec) == int(cur) if k in _ENV_PARAM_INT_KEYS else math.isclose(float(rec), float(cur))
        if not ok:
            bad.append(f"{k}: recorded={rec!r}, env={cur!r}")
    if bad:
        raise ValueError("env configuration disagrees with recorded scenarios: " + "; ".join(bad))

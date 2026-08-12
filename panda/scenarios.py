"""The frozen evaluation scenario set for PandaReach-v0.

Every stage of the pipeline -- DeePC, the imitation clone, the RL residual --
must be scored on identical episodes, or the comparison between them is
meaningless. Calling `env.reset(seed=k)` does not achieve that: it maps a seed to
a scenario *through the env's sampling code*, so any change to that code silently
remaps every seed. That already happened once in this project.

So scenarios are generated once, written to a versioned file, and thereafter
loaded and replayed via `reset(options=...)`. Collection may use seeds freely;
evaluation must not.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

SCENARIOS_PATH = "data/panda_scenarios_v1.npz"

N_SCENARIOS = 78          # matches the unicycle's evaluation-set size
SWEEP_IDS = range(20)     # phase 3a, the lambda grid
EVAL_IDS = range(78)      # phase 3b and every later stage
SHOWCASE_IDS = (0, 7, 19, 33, 51, 66)   # trace figures and video

_ENV_PARAM_KEYS = (
    "delta_max", "goal_tolerance", "min_start_goal_dist", "max_steps", "frame_skip",
)
# The two integer-valued params, stored/compared as int rather than float so
# they round-trip as 150 / 10, not 150.0 / 10.0, and so they compare exactly
# rather than via a tolerance.
_ENV_PARAM_INT_KEYS = frozenset({"max_steps", "frame_skip"})


def generate(n: int = N_SCENARIOS) -> dict:
    """Sample `n` scenarios from the env's own reset, then freeze them.

    Uses `reset(seed=i)` for `i in range(n)` -- seeds are fine *here*, because
    this runs once and the result is what gets frozen. The `seed` column is kept
    for provenance only; nothing replays by seed.
    """
    # Local import: keeps this module itself free of a module-level dependency
    # on the env (only `numpy` is imported above). This is a structural
    # preference, not a load-time saving -- `panda/__init__.py` eagerly does
    # `from .env import PandaReachEnv` at package-import time, so `import
    # panda.scenarios` already pulls in gymnasium and mujoco via the parent
    # package before this function is ever called.
    from panda.env import PandaReachEnv

    env = PandaReachEnv()
    try:
        qpos = np.empty((n, env.nq), dtype=np.float64)
        goal = np.empty((n, 3), dtype=np.float64)
        for i in range(n):
            env.reset(seed=i)
            qpos[i] = env.data.qpos
            goal[i] = env.goal
        out = {
            "qpos": qpos,
            "goal": goal,
            "seed": np.arange(n, dtype=np.int64),
        }
        for k in _ENV_PARAM_KEYS:
            value = getattr(env, k)
            out[k] = np.int64(value) if k in _ENV_PARAM_INT_KEYS else np.float64(value)
    finally:
        env.close()
    return out


def save(path: str, data: dict) -> None:
    np.savez(path, **data)


def load(path: str = SCENARIOS_PATH) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def reset_to(env, scenarios: dict, i: int):
    """Reset `env` into scenario `i`. Returns whatever `env.reset` returns."""
    return env.reset(options={
        "qpos": scenarios["qpos"][i],
        "goal": scenarios["goal"][i],
    })


def checksum(scenarios: dict) -> str:
    """SHA-256 over the scenario geometry. Pinned by a test as the freeze."""
    h = hashlib.sha256()
    for k in ("qpos", "goal"):
        h.update(np.ascontiguousarray(scenarios[k], dtype=np.float64).tobytes())
    return h.hexdigest()


def validate_against_env(env, scenarios: dict) -> None:
    """Raise if `env`'s configuration disagrees with what the scenarios recorded.

    The checksum protects the geometry; this protects the *context*. Two runs
    scored under different `max_steps` are not comparable even when every qpos
    and goal is bit-identical, and nothing else in the pipeline would notice.
    """
    mismatches = []
    for k in _ENV_PARAM_KEYS:
        recorded, current = scenarios[k], getattr(env, k)
        agree = (
            int(recorded) == int(current)
            if k in _ENV_PARAM_INT_KEYS
            else math.isclose(float(recorded), float(current))
        )
        if not agree:
            mismatches.append(f"{k}: recorded={recorded!r}, env={current!r}")
    if mismatches:
        raise ValueError(
            "env configuration disagrees with recorded scenarios -- these "
            "results would not be comparable to ones already recorded against "
            "this file: " + "; ".join(mismatches)
        )

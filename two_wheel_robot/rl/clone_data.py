# two_wheel_robot/rl/clone_data.py
"""Offline (features -> u_DeePC) dataset generation for the behavioral clone.

Hybrid coverage:
  * synthetic  -- random pose+goal; a realistic T_ini past built by rolling the
                 true dynamics forward under random admissible actions.
  * degenerate -- a constant/held-still past (the v-collapse regime).
  * onpolicy   -- states the real canonical DeePC actually visits.
Every config is labeled by calling the real `DeePC.act`.
"""
from __future__ import annotations

import warnings
from typing import cast

import gymnasium as gym
import numpy as np

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from two_wheel_robot.env.dynamics import step_unicycle, wrap_to_pi
from two_wheel_robot.env.env import UnicycleGoalEnv
from two_wheel_robot.rl.deepc_setup import bearing_y_ref
from two_wheel_robot.rl.features import featurize

_WORKSPACE = np.array([[-10.0, 10.0], [-10.0, 10.0]], dtype=np.float64)
_MIN_GOAL_DIST = 2.0
# Matches the env's default integration step (env.py UnicycleGoalEnv dt=0.025);
# the synthetic rollout must use the same dt the real dynamics/DeePC saw.
_DT = 0.025


def _sample_pose(rng: np.random.Generator) -> np.ndarray:
    x = rng.uniform(*_WORKSPACE[0])
    y = rng.uniform(*_WORKSPACE[1])
    delta = rng.uniform(-np.pi, np.pi)
    return np.array([x, y, delta], dtype=np.float64)


def _sample_goal(rng: np.random.Generator, near: np.ndarray) -> np.ndarray:
    for _ in range(100):
        gx = rng.uniform(*_WORKSPACE[0])
        gy = rng.uniform(*_WORKSPACE[1])
        if np.hypot(gx - near[0], gy - near[1]) >= _MIN_GOAL_DIST:
            return np.array([gx, gy], dtype=np.float64)
    # Fallback (geometrically near-impossible for a 20x20 box, 2-unit threshold):
    # return the last draw even though it is closer than _MIN_GOAL_DIST.
    return np.array([gx, gy], dtype=np.float64)


def make_synthetic_config(
    rng: np.random.Generator,
    action_bounds: np.ndarray,
    dt: float,
    T_ini: int,
    degenerate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One synthetic `(u_ini, y_ini, y_current, goal)` config.

    Non-degenerate: roll dynamics forward T_ini steps under random admissible
    actions, so `y_ini = states[0:T_ini]`, `y_current = states[T_ini]` (one step
    ahead, matching DeePC's buffer/current lag). Degenerate: a held-still past.
    """
    low, high = action_bounds[:, 0], action_bounds[:, 1]
    if degenerate:
        s = _sample_pose(rng)
        u_const = rng.uniform(low, high)
        u_const[0] = 0.0  # no forward motion -> constant pose -> collapse regime
        u_ini = np.tile(u_const, (T_ini, 1))
        y_ini = np.tile(s, (T_ini, 1))
        y_current = s.copy()
    else:
        s = _sample_pose(rng)
        states = [s.copy()]
        actions = []
        for _ in range(T_ini):
            u = rng.uniform(low, high)
            s = step_unicycle(s, u, dt)
            s[0] = np.clip(s[0], *_WORKSPACE[0])
            s[1] = np.clip(s[1], *_WORKSPACE[1])
            s[2] = float(wrap_to_pi(s[2]))
            actions.append(u)
            states.append(s.copy())
        u_ini = np.asarray(actions, dtype=np.float64)
        y_ini = np.asarray(states[:-1], dtype=np.float64)
        y_current = states[-1]
    goal = _sample_goal(rng, y_current)
    return u_ini, y_ini, y_current, goal


def _reset_deepc_solver_state(deepc) -> None:
    """Rebuild the CVXPY problem to flush any SCS warm-start / dual variable state.

    SCS (the default solver) retains internal state across `solve()` calls,
    which causes `generate_clone_dataset` to produce slightly different outputs
    when called on a previously-used DeePC object versus a fresh one — even if
    both start with `g.value = None`. Rebuilding the compiled problem resets
    all internal SCS/CVXPY caches, restoring determinism across calls.
    """
    deepc._build_problem()
    deepc._u_buf = None
    deepc._y_buf = None
    deepc._prev_idx = -1


def _label(deepc, u_ini, y_ini, y_current, goal):
    """Prime + act; returns `(u_deepc, idx, y_ref)` or None on QP failure."""
    deepc.prime_buffer(u_ini, y_ini)
    y_ref = bearing_y_ref(y_current, goal)
    try:
        u_t = deepc.act(y_current, y_ref)
    except RuntimeError:
        return None
    return u_t, int(deepc.last_library_idx), y_ref


def _collect_onpolicy(deepc, info, n_episodes, seed, max_steps):
    """Run canonical DeePC closed-loop; record visited (buffer, current, u)."""
    env = gym.make("TwoWheelGoal-v0", action_bounds=info["action_bounds"])
    base = cast(UnicycleGoalEnv, env.unwrapped)
    feats, targs, idxs = [], [], []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        deepc.reset(base.y, u_initial=info["u_init_midpoint"])
        term = trunc = False
        steps = 0
        while not (term or trunc) and steps < max_steps:
            u_buf, y_buf = deepc.past_buffer  # read BEFORE act slides it
            y_cur = base.y
            y_ref = bearing_y_ref(base.state, base.goal)
            try:
                u_t = deepc.act(y_cur, y_ref)
            except RuntimeError:
                break
            feats.append(featurize(u_buf, y_buf, y_cur, y_ref, info["anchors"]))
            targs.append(u_t.copy())
            idxs.append(int(deepc.last_library_idx))
            _, _, term, trunc, _ = env.step(u_t)
            steps += 1
    env.close()
    return feats, targs, idxs


def generate_clone_dataset(
    deepc,
    info: dict,
    n_synthetic: int = 20000,
    p_degenerate: float = 0.25,
    n_onpolicy_episodes: int = 100,
    seed: int = 0,
    max_steps: int = 200,
) -> dict:
    """Build the full hybrid dataset. Returns a dict of stacked arrays + meta."""
    # Flush any SCS warm-start state so that the output is deterministic
    # regardless of how many prior solves `deepc` has performed.
    _reset_deepc_solver_state(deepc)
    rng = np.random.default_rng(seed)
    T_ini = info["T_ini"]
    dt = info.get("dt", _DT)
    feats, targs, idxs, regimes = [], [], [], []

    # --- synthetic + degenerate ---
    n_failed = 0
    for _ in range(n_synthetic):
        degenerate = rng.random() < p_degenerate
        u_ini, y_ini, y_current, goal = make_synthetic_config(
            rng, info["action_bounds"], dt, T_ini, degenerate
        )
        labeled = _label(deepc, u_ini, y_ini, y_current, goal)
        if labeled is None:
            n_failed += 1
            continue
        u_t, idx, y_ref = labeled
        feats.append(featurize(u_ini, y_ini, y_current, y_ref, info["anchors"]))
        targs.append(u_t)
        idxs.append(idx)
        regimes.append("degenerate" if degenerate else "synthetic")

    # Silent QP failures would shrink the dataset below n_synthetic unnoticed;
    # surface them when they exceed a small fraction of the requested samples.
    if n_synthetic and n_failed > 0.05 * n_synthetic:
        warnings.warn(
            f"{n_failed}/{n_synthetic} synthetic configs dropped on QP failure "
            f"({n_failed / n_synthetic:.1%}) — check data coverage / action bounds.",
            stacklevel=2,
        )

    # --- on-policy (seeds disjoint from synthetic RNG; offset to be safe) ---
    op_feats, op_targs, op_idxs = _collect_onpolicy(
        deepc, info, n_onpolicy_episodes, seed=seed + 1_000_000, max_steps=max_steps
    )
    feats.extend(op_feats)
    targs.extend(op_targs)
    idxs.extend(op_idxs)
    regimes.extend(["onpolicy"] * len(op_feats))

    return {
        "features": np.asarray(feats, dtype=np.float64),
        "targets": np.asarray(targs, dtype=np.float64),
        "library_idx": np.asarray(idxs, dtype=np.int64),
        "regime": np.asarray(regimes, dtype=object).astype("U16"),
        "anchors": info["anchors"],
        "T_ini": np.int64(T_ini),
        "n_lib": np.int64(len(info["anchors"])),
    }


def save_dataset(path: str, ds: dict) -> None:
    """Save a dataset dict to a .npz."""
    np.savez(path, **ds)

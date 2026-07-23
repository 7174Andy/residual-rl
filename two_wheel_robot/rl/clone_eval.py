# two_wheel_robot/rl/clone_eval.py
"""Layered fidelity gate for the deep-lcc clone.

Proves behavioral equivalence, not just low regression error:
  (1) regime-conditioned action regression,
  (2) closed-loop trajectory fidelity (clone-only vs DeePC from the same seeds),
  (3) paired per-seed outcome agreement (McNemar + Wilson reach-rate CIs).
Pure-numpy statistics (no scipy).
"""
from __future__ import annotations

from math import erfc, sqrt
from typing import cast

import gymnasium as gym
import numpy as np

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from two_wheel_robot.env.env import UnicycleGoalEnv
from two_wheel_robot.rl.deepc_setup import bearing_y_ref
from two_wheel_robot.rl.features import featurize


# ----- statistics -------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_pvalue(b: int, c: int) -> float:
    """McNemar p-value, continuity-corrected, via the exact chi2(1) survival.

    `b` = #(DeePC reach, clone fail), `c` = #(DeePC fail, clone reach). The
    statistic is `chi2(1)`-distributed, whose survival function is
    `erfc(sqrt(stat/2))`.
    """
    n = b + c
    if n == 0:
        return 1.0
    stat = max(0.0, abs(b - c) - 1.0) ** 2 / n
    # stat == 0 (|b - c| <= 1) -> erfc(0) == 1.0; the guard just makes that explicit.
    if stat <= 0.0:
        return 1.0
    return erfc(sqrt(stat / 2.0))


# ----- (1) regression by regime ----------------------------------------------

def regression_by_regime(pred: np.ndarray, true: np.ndarray, regime: np.ndarray) -> dict:
    """Per-regime MAE/RMSE on v and w."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    regime = np.asarray(regime)
    out: dict[str, dict] = {}
    for r in np.unique(regime):
        m = regime == r
        err = pred[m] - true[m]
        out[str(r)] = {
            "n": int(m.sum()),
            "mae_v": float(np.abs(err[:, 0]).mean()),
            "mae_w": float(np.abs(err[:, 1]).mean()),
            "rmse_v": float(np.sqrt((err[:, 0] ** 2).mean())),
            "rmse_w": float(np.sqrt((err[:, 1] ** 2).mean())),
        }
    return out


# ----- (2)/(3) closed-loop runners -------------------------------------------

def _reached(info: dict) -> bool:
    return bool(info.get("reached", False))


def run_clone_closed_loop_with_actions(predictor, info: dict, env, seed: int):
    """Like `run_clone_closed_loop`, also returning the per-step applied action.

    Returns `(reached, trajectory (T+1, 3), actions (T, 2))`.
    """
    base = cast(UnicycleGoalEnv, env.unwrapped)
    env.reset(seed=seed)
    T_ini = info["T_ini"]
    u_buf = np.tile(info["u_init_midpoint"], (T_ini, 1))
    y_buf = np.tile(base.y, (T_ini, 1))
    traj = [base.state.copy()]
    actions = []
    term = trunc = False
    last_info: dict = {}
    while not (term or trunc):
        y_cur = base.y
        y_ref = bearing_y_ref(base.state, base.goal)
        feat = featurize(u_buf, y_buf, y_cur, y_ref, info["anchors"])
        u = np.clip(
            predictor.predict(feat),
            info["action_bounds"][:, 0], info["action_bounds"][:, 1],
        )
        _, _, term, trunc, last_info = env.step(u)
        u_buf = np.roll(u_buf, -1, axis=0); u_buf[-1] = u
        y_buf = np.roll(y_buf, -1, axis=0); y_buf[-1] = y_cur
        traj.append(base.state.copy())
        actions.append(u.copy())
    return _reached(last_info), np.asarray(traj), np.asarray(actions)


def run_clone_closed_loop(predictor, info: dict, env, seed: int):
    """Run clone-only in the loop. Returns `(reached, trajectory (T+1, 3))`."""
    reached, traj, _ = run_clone_closed_loop_with_actions(predictor, info, env, seed)
    return reached, traj


def run_deepc_closed_loop(deepc, info: dict, env, seed: int):
    """Run the real DeePC in the loop. Returns `(reached, trajectory (T+1, 3))`."""
    base = cast(UnicycleGoalEnv, env.unwrapped)
    env.reset(seed=seed)
    deepc.reset(base.y, u_initial=info["u_init_midpoint"])
    traj = [base.state.copy()]
    term = trunc = False
    last_info: dict = {}
    while not (term or trunc):
        y_ref = bearing_y_ref(base.state, base.goal)
        try:
            u = deepc.act(base.y, y_ref)
        except RuntimeError:
            break
        _, _, term, trunc, last_info = env.step(u)
        traj.append(base.state.copy())
    return _reached(last_info), np.asarray(traj)


def trajectory_deviation(traj_clone: np.ndarray, traj_deepc: np.ndarray) -> dict:
    """Position/heading deviation between two trajectories over the shared horizon."""
    h = min(len(traj_clone), len(traj_deepc))
    a, b = traj_clone[:h], traj_deepc[:h]
    pos = np.linalg.norm(a[:, :2] - b[:, :2], axis=1)
    head = np.abs((a[:, 2] - b[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    return {
        "pos_median": float(np.median(pos)),
        "pos_p95": float(np.percentile(pos, 95)),
        "head_median": float(np.median(head)),
        "head_p95": float(np.percentile(head, 95)),
    }


def paired_outcomes(deepc, predictor, info: dict, seeds, action_bounds) -> dict:
    """Run both controllers on each seed; return the paired confusion + stats."""
    env_d = gym.make("TwoWheelGoal-v0", action_bounds=action_bounds)
    env_c = gym.make("TwoWheelGoal-v0", action_bounds=action_bounds)
    both = neither = b = c = 0  # b: deepc-only reach, c: clone-only reach
    devs = []
    d_reach = k_reach = 0
    for s in seeds:
        rd, traj_d = run_deepc_closed_loop(deepc, info, env_d, int(s))
        rc, traj_c = run_clone_closed_loop(predictor, info, env_c, int(s))
        d_reach += int(rd); k_reach += int(rc)
        if rd and rc: both += 1
        elif rd and not rc: b += 1
        elif rc and not rd: c += 1
        else: neither += 1
        devs.append(trajectory_deviation(traj_c, traj_d))
    env_d.close(); env_c.close()
    n = len(seeds)
    return {
        "n": n,
        "confusion": {"both": both, "deepc_only": b, "clone_only": c, "neither": neither},
        "agreement_rate": (both + neither) / n if n else 0.0,
        "mcnemar_p": mcnemar_pvalue(b, c),
        "deepc_reach_rate": d_reach / n if n else 0.0,
        "clone_reach_rate": k_reach / n if n else 0.0,
        "deepc_reach_ci": wilson_ci(d_reach, n),
        "clone_reach_ci": wilson_ci(k_reach, n),
        # Cross-seed medians of the per-seed deviation summaries (so
        # traj_pos_median_p95 is the median over seeds of each seed's p95).
        "traj_pos_median": float(np.median([d["pos_median"] for d in devs])) if devs else float("nan"),
        "traj_pos_median_p95": float(np.median([d["pos_p95"] for d in devs])) if devs else float("nan"),
    }

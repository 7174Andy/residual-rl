# two_wheel_robot/rl/residual_eval.py
"""Benchmark: DeePC vs clone-only vs clone+residual (RL + MPC), optionally vs vanilla RL.

Reuses clone_eval's DeePC/clone closed-loop runners and statistics verbatim so the
numbers are directly comparable to the clone fidelity gate.
"""
from __future__ import annotations

from typing import cast

import gymnasium as gym
import numpy as np

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from two_wheel_robot.env.env import UnicycleGoalEnv
from two_wheel_robot.rl.clone_eval import (
    mcnemar_pvalue,
    run_clone_closed_loop_with_actions,
    run_deepc_closed_loop_with_actions,
    trajectory_deviation,
    wilson_ci,
)
from two_wheel_robot.rl.residual_env import ResidualDeePCEnv
from two_wheel_robot.rl.trace_reward import DEFAULT_Q, DEFAULT_R, recompute_reward


def episode_return(traj: np.ndarray, actions: np.ndarray, goal: np.ndarray) -> dict:
    """Total return for one episode, split into the terms of the DeePC-form cost.

    Uses `trace_reward.recompute_reward` (the same arithmetic `env.step` runs) so
    the total matches what the env actually paid out. The split is recomputed here
    because `recompute_reward` returns only the summed reward:

        position = (x-g_x)^2 + (y-g_y)^2      the part the body-frame obs can see
        heading  = Q[2,2] * delta^2          absolute world heading, invisible to vanilla
        control  = u^T R u
    """
    traj = np.asarray(traj, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64).reshape(2)
    # Row 0 is post-reset (no action yet); pad actions to match trace_reward's schema.
    v = np.concatenate([[0.0], actions[:, 0]]) if len(actions) else np.zeros(len(traj))
    w = np.concatenate([[0.0], actions[:, 1]]) if len(actions) else np.zeros(len(traj))
    rw = recompute_reward(traj[:, 0], traj[:, 1], traj[:, 2], v, w, goal)

    pos = (traj[1:, 0] - goal[0]) ** 2 + (traj[1:, 1] - goal[1]) ** 2
    head = DEFAULT_Q[2, 2] * traj[1:, 2] ** 2
    ctrl = np.einsum("ti,ij,tj->t", actions, DEFAULT_R, actions) if len(actions) else np.zeros(0)
    return {
        "total": float(rw["cum_reward"][-1]),
        "steps": int(len(traj) - 1),
        "position_cost": float(pos.sum()),
        "heading_cost": float(head.sum()),
        "control_cost": float(ctrl.sum()),
    }


def run_residual_closed_loop_with_actions(model, res_env: ResidualDeePCEnv, seed: int):
    """Like `run_residual_closed_loop`, also returning the per-step applied action.

    The applied action is read back off `res_env.base.last_action` (the inner
    env's post-clip `u`), not the raw actor output -- that raw output is the
    residual `a_res` in `[-1, 1]^2`, not the real-unit `(v, w)` this is for.
    Returns `(reached, trajectory (T+1, 3), actions (T, 2))`.
    """
    obs, _ = res_env.reset(seed=int(seed))
    traj = [res_env.base.state.copy()]
    actions = []
    term = trunc = False
    last_info: dict = {}
    while not (term or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, last_info = res_env.step(action)
        traj.append(res_env.base.state.copy())
        actions.append(res_env.base.last_action.copy())
    return bool(last_info.get("reached", False)), np.asarray(traj), np.asarray(actions)


def run_residual_closed_loop(model, res_env: ResidualDeePCEnv, seed: int):
    """Run clone+residual in the loop. Returns (reached, trajectory (T+1, 3))."""
    reached, traj, _ = run_residual_closed_loop_with_actions(model, res_env, seed)
    return reached, traj


def run_vanilla_closed_loop_with_actions(model, env, seed: int):
    """Like `run_vanilla_closed_loop`, also returning the per-step applied action.

    `env` is a `wrappers.vanilla_rl_env`, whose action space already *is* DeePC's
    `u_bounds`, so the policy output is `(v, w)` in physical units. The action is
    still read back off `base.last_action` (post-clip) to match the other runners.
    Returns `(reached, trajectory (T+1, 3), actions (T, 2))`.
    """
    base = cast(UnicycleGoalEnv, env.unwrapped)
    obs, _ = env.reset(seed=int(seed))
    traj = [base.state.copy()]
    actions = []
    term = trunc = False
    last_info: dict = {}
    while not (term or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, last_info = env.step(action)
        traj.append(base.state.copy())
        actions.append(base.last_action.copy())
    return bool(last_info.get("reached", False)), np.asarray(traj), np.asarray(actions)


def run_vanilla_closed_loop(model, env, seed: int):
    """Run the from-scratch RL policy in the loop. Returns (reached, trajectory (T+1, 3))."""
    reached, traj, _ = run_vanilla_closed_loop_with_actions(model, env, seed)
    return reached, traj


def benchmark(model, deepc, predictor, res_env, info, seeds,
              vanilla_model=None, vanilla_env=None) -> dict:
    """Run all three controllers on each seed; return reach rates + paired stats.

    `regressions` = #(clone reached, residual failed) — target 0.
    `rescued`     = #(clone failed, residual reached) — the collapse seeds fixed.
    McNemar is on the (regressions, rescued) discordant pair.

    Passing `vanilla_model` + `vanilla_env` adds the from-scratch RL baseline on the
    same seeds: `vanilla_reach`/`_rate`/`_ci` and McNemar vs the residual.
    """
    run_vanilla = vanilla_model is not None and vanilla_env is not None
    v_reach = v_only = r_only = 0
    action_bounds = info["action_bounds"]
    env_d = gym.make("TwoWheelGoal-v0", action_bounds=action_bounds)
    env_c = gym.make("TwoWheelGoal-v0", action_bounds=action_bounds)
    d_reach = c_reach = r_reach = 0
    regressions = rescued = 0
    devs = []
    rets: dict[str, list] = {"deepc": [], "clone": [], "residual": [], "vanilla": []}
    n = 0
    try:
        for s in seeds:
            s = int(s)
            rd, traj_d, act_d = run_deepc_closed_loop_with_actions(deepc, info, env_d, s)
            # Read the goal AFTER the run — the runner resets the env internally, so
            # reading it first would pick up the previous seed's goal. Every arm resets
            # from the same seed, so one read serves all four.
            goal = cast(UnicycleGoalEnv, env_d.unwrapped).goal.copy()
            rets["deepc"].append(episode_return(traj_d, act_d, goal))
            rc, traj_c, act_c = run_clone_closed_loop_with_actions(predictor, info, env_c, s)
            rets["clone"].append(episode_return(traj_c, act_c, goal))
            rr, traj_r, act_r = run_residual_closed_loop_with_actions(model, res_env, s)
            rets["residual"].append(episode_return(traj_r, act_r, goal))
            if run_vanilla:
                rv, traj_v, act_v = run_vanilla_closed_loop_with_actions(
                    vanilla_model, vanilla_env, s
                )
                rets["vanilla"].append(episode_return(traj_v, act_v, goal))
                v_reach += int(rv)
                v_only += int(rv and not rr)
                r_only += int(rr and not rv)
            d_reach += int(rd)
            c_reach += int(rc)
            r_reach += int(rr)
            if rc and not rr:
                regressions += 1
            if rr and not rc:
                rescued += 1
            devs.append(trajectory_deviation(traj_c, traj_r))
            n += 1
    finally:
        env_d.close()
        env_c.close()
    vanilla = {
        "vanilla_reach": v_reach,
        "vanilla_reach_rate": v_reach / n if n else 0.0,
        "vanilla_ci": wilson_ci(v_reach, n),
        "mcnemar_vanilla_vs_residual": mcnemar_pvalue(v_only, r_only),
    } if run_vanilla else {}
    returns = {
        f"return_{arm}": {
            "mean": float(np.mean([e["total"] for e in eps])),
            "median": float(np.median([e["total"] for e in eps])),
            "std": float(np.std([e["total"] for e in eps])),
            "mean_steps": float(np.mean([e["steps"] for e in eps])),
            "mean_position_cost": float(np.mean([e["position_cost"] for e in eps])),
            "mean_heading_cost": float(np.mean([e["heading_cost"] for e in eps])),
            "mean_control_cost": float(np.mean([e["control_cost"] for e in eps])),
        }
        for arm, eps in rets.items() if eps
    }
    return {
        **vanilla,
        **returns,
        "n": n,
        "deepc_reach": d_reach,
        "clone_reach": c_reach,
        "residual_reach": r_reach,
        "deepc_reach_rate": d_reach / n if n else 0.0,
        "clone_reach_rate": c_reach / n if n else 0.0,
        "residual_reach_rate": r_reach / n if n else 0.0,
        "deepc_ci": wilson_ci(d_reach, n),
        "clone_ci": wilson_ci(c_reach, n),
        "residual_ci": wilson_ci(r_reach, n),
        "mcnemar_residual_vs_clone": mcnemar_pvalue(regressions, rescued),
        "regressions": regressions,
        "rescued": rescued,
        "traj_dev_vs_clone_median": (
            float(np.median([d["pos_median"] for d in devs])) if devs else float("nan")
        ),
    }

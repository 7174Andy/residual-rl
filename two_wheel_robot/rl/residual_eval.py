# two_wheel_robot/rl/residual_eval.py
"""Three-way benchmark: DeePC vs clone-only vs clone+residual (RL + MPC).

Reuses clone_eval's DeePC/clone closed-loop runners and statistics verbatim so the
numbers are directly comparable to the clone fidelity gate.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from two_wheel_robot.rl.clone_eval import (
    mcnemar_pvalue,
    run_clone_closed_loop,
    run_deepc_closed_loop,
    trajectory_deviation,
    wilson_ci,
)
from two_wheel_robot.rl.residual_env import ResidualDeePCEnv


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


def benchmark(model, deepc, predictor, res_env, info, seeds) -> dict:
    """Run all three controllers on each seed; return reach rates + paired stats.

    `regressions` = #(clone reached, residual failed) — target 0.
    `rescued`     = #(clone failed, residual reached) — the collapse seeds fixed.
    McNemar is on the (regressions, rescued) discordant pair.
    """
    action_bounds = info["action_bounds"]
    env_d = gym.make("TwoWheelGoal-v0", action_bounds=action_bounds)
    env_c = gym.make("TwoWheelGoal-v0", action_bounds=action_bounds)
    d_reach = c_reach = r_reach = 0
    regressions = rescued = 0
    devs = []
    n = 0
    try:
        for s in seeds:
            s = int(s)
            rd, _ = run_deepc_closed_loop(deepc, info, env_d, s)
            rc, traj_c = run_clone_closed_loop(predictor, info, env_c, s)
            rr, traj_r = run_residual_closed_loop(model, res_env, s)
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
    return {
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

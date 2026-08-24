"""Decompose episode return for the Reacher RL arms (journey 13).

    return = -sum(dist) + reach_bonus * (steps inside tolerance)
             - ctrl_cost * sum(|u|^2)

Rolls both policies deterministically over the same frozen scenarios and
prints the three terms side by side, plus paired per-scenario counts. This is
the provenance for journey 13's claim that vanilla's higher training return is
almost entirely the station-keeping bonus.

    uv run python scripts/decompose_reacher_returns.py
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

import reacher  # noqa: F401  registers the Gym ID
from reacher.residual_env import ResidualSelectEnv
from rl.sb3 import load_policy


def rollout(env, predict, q0, goal, base):
    """`predict(obs) -> action`; `base` supplies max_steps and reward constants."""
    obs, _ = env.reset(seed=0, options={"qpos": q0, "goal": goal})
    dist_sum = ctrl_sum = in_tol = 0
    first = None
    for t in range(base.max_steps):
        obs, _r, _term, trunc, info = env.step(predict(obs))
        dist_sum += info["dist"]
        u = np.asarray(info["action"])
        ctrl_sum += float(u @ u)
        if info["reached"]:
            in_tol += 1
            if first is None:
                first = t + 1
        if trunc:
            break
    ret = -dist_sum + base.reach_bonus * in_tol - base.ctrl_cost * ctrl_sum
    return dict(ret=ret, dist=dist_sum, tol=in_tol, ctrl=ctrl_sum,
                first=first if first is not None else np.nan)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", default="data/reacher_scenarios_v1.npz")
    p.add_argument("--clone", default="data/dagger_clone_r3.pt")
    p.add_argument("--residual", default="data/reacher_residual_dagger_400k.zip")
    p.add_argument("--vanilla", default="data/reacher_vanilla_400k.zip")
    p.add_argument("--algo", default="sac")
    args = p.parse_args()

    with np.load(args.scenarios) as z:
        eps = [(z["qpos"][i], z["goal"][i]) for i in range(len(z["qpos"]))]
    n = len(eps)

    res_env = ResidualSelectEnv(clone_path=args.clone)
    res = load_policy(args.residual, algo=args.algo, device="cpu")
    R = [rollout(res_env, lambda o: res.predict(o, deterministic=True)[0],
                 q0, g, res_env.base) for q0, g in eps]
    res_env.close()

    van_env = gym.make("ReacherGoal-v0")
    van = load_policy(args.vanilla, algo=args.algo, device="cpu")
    # the vanilla policy reads the env's own observation, not rollout's copy
    V = [rollout(van_env,
                 lambda _o: van.predict(van_env.unwrapped.build_obs(),
                                        deterministic=True)[0],
                 q0, g, van_env.unwrapped) for q0, g in eps]
    van_env.close()

    for name, X in (("residual", R), ("vanilla ", V)):
        print(f"{name}:  return {np.mean([x['ret'] for x in X]):6.2f}   "
              f"dist integral {np.mean([x['dist'] for x in X]):5.2f}   "
              f"in-tol steps {np.mean([x['tol'] for x in X]):5.1f}   "
              f"first reach @ {np.nanmedian([x['first'] for x in X]):4.1f}   "
              f"ctrl cost {1e-3 * np.mean([x['ctrl'] for x in X]):.4f}")

    rt = np.array([x["tol"] for x in R])
    vt = np.array([x["tol"] for x in V])
    print(f"\nin-tol steps, paired: residual longer on "
          f"{int((rt > vt).sum())}/{n}, vanilla longer on "
          f"{int((vt > rt).sum())}/{n}, tied {int((rt == vt).sum())}")
    rd = np.array([x["dist"] for x in R])
    vd = np.array([x["dist"] for x in V])
    print(f"dist integral, paired: residual smaller on {int((rd < vd).sum())}/{n}")


if __name__ == "__main__":
    main()

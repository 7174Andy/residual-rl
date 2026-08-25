"""Unicycle analog of journey 13's on/off-policy disagreement measurement.

Rolls the same held-out seeds two ways:
  expert-driven: canonical DeePC drives; at each visited state the clone's
                 action is compared to the expert's (the training regime).
  clone-driven:  the clone drives its own closed loop; the expert is queried
                 (prime_buffer + act) at the clone's states (the deployment
                 regime).
Reports the median ||u_clone - u_expert|| under each distribution + the ratio.

    uv run python scripts/measure_clone_disagreement.py --clone data/clone.pt --episodes 15
"""
from __future__ import annotations

import argparse
from typing import cast

import gymnasium as gym
import numpy as np

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from rl.clone import load_clone
from two_wheel_robot.env.env import UnicycleGoalEnv
from two_wheel_robot.rl.clone_data import _label, _reset_deepc_solver_state
from two_wheel_robot.rl.deepc_setup import bearing_y_ref, build_canonical_deepc
from two_wheel_robot.rl.features import featurize


def expert_driven(deepc, info, predictor, env, seed, max_steps=200):
    base = cast(UnicycleGoalEnv, env.unwrapped)
    env.reset(seed=seed)
    deepc.reset(base.y, u_initial=info["u_init_midpoint"])
    lo, hi = info["action_bounds"][:, 0], info["action_bounds"][:, 1]
    dis, term, trunc, steps = [], False, False, 0
    while not (term or trunc) and steps < max_steps:
        u_buf, y_buf = deepc.past_buffer  # read BEFORE act slides it
        y_cur = base.y
        y_ref = bearing_y_ref(base.state, base.goal)
        try:
            u_star = deepc.act(y_cur, y_ref)
        except RuntimeError:
            break
        feat = featurize(u_buf, y_buf, y_cur, y_ref, info["anchors"])
        u_clone = np.clip(predictor.predict(feat), lo, hi)
        dis.append(float(np.linalg.norm(u_clone - u_star)))
        _, _, term, trunc, _ = env.step(u_star)
        steps += 1
    return dis


def clone_driven(deepc, info, predictor, env, seed, max_steps=200):
    base = cast(UnicycleGoalEnv, env.unwrapped)
    env.reset(seed=seed)
    T_ini = info["T_ini"]
    lo, hi = info["action_bounds"][:, 0], info["action_bounds"][:, 1]
    u_buf = np.tile(info["u_init_midpoint"], (T_ini, 1))
    y_buf = np.tile(base.y, (T_ini, 1))
    dis, term, trunc, steps = [], False, False, 0
    while not (term or trunc) and steps < max_steps:
        y_cur = base.y
        y_ref = bearing_y_ref(base.state, base.goal)
        feat = featurize(u_buf, y_buf, y_cur, y_ref, info["anchors"])
        u_clone = np.clip(predictor.predict(feat), lo, hi)
        labeled = _label(deepc, u_buf, y_buf, y_cur, base.goal)
        if labeled is not None:
            u_star, _idx, _yref = labeled
            dis.append(float(np.linalg.norm(u_clone - u_star)))
        _, _, term, trunc, _ = env.step(u_clone)
        u_buf = np.roll(u_buf, -1, axis=0); u_buf[-1] = u_clone
        y_buf = np.roll(y_buf, -1, axis=0); y_buf[-1] = y_cur
        steps += 1
    return dis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clone", default="data/clone.pt")
    p.add_argument("--libraries", default="data/libraries_v0.npz")
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--seed-base", type=int, default=4104626029)
    p.add_argument("--out", default=None, help="optional npz of raw distances")
    args = p.parse_args()

    deepc, info = build_canonical_deepc(libraries_path=args.libraries)
    predictor = load_clone(args.clone, device="cpu")
    env = gym.make("TwoWheelGoal-v0", action_bounds=info["action_bounds"])

    on_expert, on_clone = [], []
    for i in range(args.episodes):
        s = args.seed_base + i
        _reset_deepc_solver_state(deepc)
        d = expert_driven(deepc, info, predictor, env, s)
        on_expert.extend(d)
        _reset_deepc_solver_state(deepc)
        d = clone_driven(deepc, info, predictor, env, s)
        on_clone.extend(d)
        print(f"seed {s}: expert-driven {len(on_expert)} states so far, "
              f"clone-driven {len(on_clone)}", flush=True)
    env.close()

    me = float(np.median(on_expert))
    mc = float(np.median(on_clone))
    print(f"\nclone-vs-expert disagreement ({args.clone}, "
          f"{args.episodes} episodes):")
    print(f"  at expert-visited states: median {me:.4f}  (n={len(on_expert)})")
    print(f"  at clone-visited states:  median {mc:.4f}  (n={len(on_clone)})")
    print(f"  ratio: {mc / me:.2f}x")
    if args.out:
        np.savez(args.out, on_expert=np.array(on_expert),
                 on_clone=np.array(on_clone))
        print("wrote", args.out)


if __name__ == "__main__":
    main()

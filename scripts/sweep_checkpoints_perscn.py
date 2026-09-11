"""Per-scenario greedy sweep of a checkpoint directory (journey 16).

`sweep_reacher_checkpoints.py` reports one aggregate row per checkpoint, which
is exactly what hid journey 16's finding: the 95->100% reach-rate tail is 1-3
individual scenarios flickering across the 10 mm threshold. This sweep keeps
the per-scenario rows so failures can be tracked, classified, and plotted
individually (`scripts/plot_tail_figures.py`).

    # the window-env arms (bcinit / cold / annealed)
    uv run python scripts/sweep_checkpoints_perscn.py \\
        --ckpt-dir data/reacher_bcinit_ckpt_400k --arm bcinit --env window \\
        --out data/tail_perscn_bcinit.csv

    # the plain-env arm (vanilla RL)
    uv run python scripts/sweep_checkpoints_perscn.py \\
        --ckpt-dir data/reacher_van_ckpt_400k --arm vanilla --env vanilla \\
        --out data/tail_perscn_vanilla.csv

Protocol matches every other Reacher eval: greedy actions, full horizon,
reached = ever inside tolerance (journey 12's no-early-stop rule).
"""
from __future__ import annotations

import argparse
import csv
import os
import re

import numpy as np


def episode(env, env_kind, model, q0, g):
    obs, info = env.reset(seed=0, options={"qpos": q0, "goal": g})
    base = env.unwrapped
    need = best = float(info["dist"])
    reached, trunc = False, False
    while not trunc:
        x = obs if env_kind == "window" else base.build_obs()
        a, _ = model.predict(x, deterministic=True)
        obs, _r, _t, trunc, info = env.step(a)
        best = min(best, float(info["dist"]))
        reached = reached or bool(info["reached"])
    return need, best, float(info["dist"]), reached


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--arm", required=True,
                   help="label written into the CSV's `arm` column")
    p.add_argument("--env", default="window", choices=["window", "vanilla"],
                   help="observation the checkpoints were trained on: 43-D "
                        "clone window (train_reacher_window.py) or the plain "
                        "8-D ReacherGoal obs (train_reacher_vanilla.py)")
    p.add_argument("--algo", default="sac")
    p.add_argument("--scenarios", default="data/reacher_scenarios_v1.npz")
    p.add_argument("--min-steps", type=int, default=0,
                   help="skip checkpoints below this step count")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import gymnasium as gym

    import reacher  # noqa: F401  registers the Gym ID
    from reacher.window_env import CloneWindowEnv
    from rl.sb3 import load_policy

    with np.load(args.scenarios) as z:
        eps = [(z["qpos"][i], z["goal"][i]) for i in range(len(z["qpos"]))]

    cks = sorted((int(m.group(1)), os.path.join(args.ckpt_dir, f))
                 for f in os.listdir(args.ckpt_dir)
                 if (m := re.fullmatch(r"ckpt_(\d+)_steps\.zip", f))
                 and int(m.group(1)) >= args.min_steps)
    env = (CloneWindowEnv() if args.env == "window"
           else gym.make("ReacherGoal-v0"))
    rows = []
    for steps, path in cks:
        model = load_policy(path, algo=args.algo, device="cpu")
        k = 0
        for i, (q0, g) in enumerate(eps):
            need, best, final, reached = episode(env, args.env, model, q0, g)
            k += reached
            rows.append([args.arm, steps, i, f"{1e3 * need:.2f}",
                         f"{1e3 * best:.3f}", f"{1e3 * final:.3f}",
                         int(reached)])
        print(f"{args.arm} {steps}: {k}/{len(eps)}", flush=True)
    env.close()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        w = csv.writer(f)
        w.writerow(["arm", "steps", "scn", "need_mm", "best_mm", "final_mm",
                    "reached"])
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

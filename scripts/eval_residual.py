# scripts/eval_residual.py
"""Benchmark DeePC vs clone-only vs clone+residual on the seed sweep.

Add `--vanilla data/vanilla_td3.zip` for a four-way comparison against the
from-scratch RL baseline on the same seeds.
"""
from __future__ import annotations

import argparse

from two_wheel_robot.rl.clone import load_clone
from two_wheel_robot.rl.deepc_setup import build_canonical_deepc
from two_wheel_robot.rl.residual_env import ResidualDeePCEnv
from two_wheel_robot.rl.residual_eval import benchmark
from two_wheel_robot.rl.train_sb3 import load_residual
from two_wheel_robot.rl.wrappers import vanilla_rl_env


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/residual_td3.zip")
    p.add_argument("--vanilla", default=None,
                   help="from-scratch RL checkpoint to include as a fourth arm")
    p.add_argument("--vanilla-algo", default="td3", choices=["td3", "sac"])
    p.add_argument("--clone", default="data/clone.pt")
    p.add_argument("--libraries", default="data/libraries_v0.npz")
    p.add_argument("--n_seeds", type=int, default=78)
    p.add_argument("--base_seed", type=int, default=4104626029)
    p.add_argument("--residual-frac", type=float, default=1.0)
    p.add_argument("--algo", default="td3", choices=["td3", "sac"])
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    deepc, info = build_canonical_deepc(libraries_path=args.libraries)
    predictor = load_clone(args.clone, device=args.device)
    model = load_residual(args.model, algo=args.algo, device=args.device)
    res_env = ResidualDeePCEnv(
        clone_path=args.clone, libraries_path=args.libraries,
        residual_frac=args.residual_frac, device=args.device,
    )
    van_model = (
        load_residual(args.vanilla, algo=args.vanilla_algo, device=args.device)
        if args.vanilla else None
    )
    van_env = vanilla_rl_env(info["action_bounds"]) if args.vanilla else None
    seeds = [args.base_seed + i for i in range(args.n_seeds)]
    try:
        rep = benchmark(model, deepc, predictor, res_env, info, seeds,
                        vanilla_model=van_model, vanilla_env=van_env)
    finally:
        res_env.close()
        if van_env is not None:
            van_env.close()

    print(f"=== benchmark over {rep['n']} seeds (base {args.base_seed}) ===")
    print(f"  DeePC    reach {rep['deepc_reach']:3d}/{rep['n']} = {rep['deepc_reach_rate']:.3f}"
          f"  CI {tuple(round(x, 3) for x in rep['deepc_ci'])}")
    print(f"  clone    reach {rep['clone_reach']:3d}/{rep['n']} = {rep['clone_reach_rate']:.3f}"
          f"  CI {tuple(round(x, 3) for x in rep['clone_ci'])}")
    print(f"  residual reach {rep['residual_reach']:3d}/{rep['n']} = {rep['residual_reach_rate']:.3f}"
          f"  CI {tuple(round(x, 3) for x in rep['residual_ci'])}")
    if "vanilla_reach" in rep:
        print(f"  vanilla  reach {rep['vanilla_reach']:3d}/{rep['n']} = "
              f"{rep['vanilla_reach_rate']:.3f}"
              f"  CI {tuple(round(x, 3) for x in rep['vanilla_ci'])}")
        print(f"  vanilla vs residual: McNemar "
              f"p={rep['mcnemar_vanilla_vs_residual']:.4f}")
    print(f"  residual vs clone: McNemar p={rep['mcnemar_residual_vs_clone']:.4f}  "
          f"rescued={rep['rescued']}  regressions={rep['regressions']}")
    print(f"  traj dev vs clone (median): {rep['traj_dev_vs_clone_median']:.3f}")

    print("\n=== cumulative episode return (DeePC-form reward, same seeds) ===")
    print(f"  {'arm':<9} {'mean':>10} {'median':>10} {'std':>9} "
          f"{'steps':>7} {'pos':>9} {'head':>8} {'ctrl':>7}")
    for arm in ("deepc", "clone", "residual", "vanilla"):
        r = rep.get(f"return_{arm}")
        if r is None:
            continue
        print(f"  {arm:<9} {r['mean']:>10.1f} {r['median']:>10.1f} {r['std']:>9.1f} "
              f"{r['mean_steps']:>7.1f} {r['mean_position_cost']:>9.1f} "
              f"{r['mean_heading_cost']:>8.1f} {r['mean_control_cost']:>7.2f}")
    print("  (pos/head/ctrl are mean per-episode COST contributions, so lower is better;")
    print("   return = -(pos + head + ctrl) + 100 per reached step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

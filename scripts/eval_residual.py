# scripts/eval_residual.py
"""Three-way benchmark: DeePC vs clone-only vs clone+residual on the seed sweep."""
from __future__ import annotations

import argparse

from two_wheel_robot.rl.clone import load_clone
from two_wheel_robot.rl.deepc_setup import build_canonical_deepc
from two_wheel_robot.rl.residual_env import ResidualDeePCEnv
from two_wheel_robot.rl.residual_eval import benchmark
from two_wheel_robot.rl.train_sb3 import load_residual


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/residual_td3.zip")
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
    seeds = [args.base_seed + i for i in range(args.n_seeds)]
    try:
        rep = benchmark(model, deepc, predictor, res_env, info, seeds)
    finally:
        res_env.close()

    print(f"=== three-way benchmark over {rep['n']} seeds (base {args.base_seed}) ===")
    print(f"  DeePC    reach {rep['deepc_reach']:3d}/{rep['n']} = {rep['deepc_reach_rate']:.3f}"
          f"  CI {tuple(round(x, 3) for x in rep['deepc_ci'])}")
    print(f"  clone    reach {rep['clone_reach']:3d}/{rep['n']} = {rep['clone_reach_rate']:.3f}"
          f"  CI {tuple(round(x, 3) for x in rep['clone_ci'])}")
    print(f"  residual reach {rep['residual_reach']:3d}/{rep['n']} = {rep['residual_reach_rate']:.3f}"
          f"  CI {tuple(round(x, 3) for x in rep['residual_ci'])}")
    print(f"  residual vs clone: McNemar p={rep['mcnemar_residual_vs_clone']:.4f}  "
          f"rescued={rep['rescued']}  regressions={rep['regressions']}")
    print(f"  traj dev vs clone (median): {rep['traj_dev_vs_clone_median']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

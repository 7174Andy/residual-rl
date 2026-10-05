"""SAC over the Panda box clone: residual (Eq. 18), warm start (Eq. 17), cold control,
and vanilla (the env's own observation, no clone anywhere).

    uv run python scripts/train_panda_rl.py --arm residual --clone data/panda_clone_box_r2.pt
    uv run python scripts/train_panda_rl.py --arm warm --clone data/panda_clone_box_r2_squash.pt
    uv run python scripts/train_panda_rl.py --arm cold
    uv run python scripts/train_panda_rl.py --arm vanilla

`warm` and `cold` share `PandaWindowEnv` and differ ONLY in the actor init, so
the cold arm is the control that says whether the clone's weights helped. After
training, the final policy is scored on the frozen 78 box scenarios through
`panda/eval.py::run_scenarios` -- the same harness as the clone rows -- and
appended to the results CSV as `--method`. Score an existing zip with --eval.
"""
from __future__ import annotations

import argparse
import math
import os

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from panda import scenarios as sc
from panda.eval import append_results, run_scenarios
from panda.rl_env import PandaResidualEnv, PandaVanillaEnv, PandaWindowEnv, RLPolicy
from rl.sb3 import (
    AnnealTargetEntropyCallback,
    build_model,
    ckpt_cb,
    init_actor_from_clone,
    load_policy,
    zero_init_actor,
)
from rl.wb import callbacks, finish, init_run, sb3_callback


def make_env(args):
    if args.arm == "residual":
        return PandaResidualEnv(args.clone, residual_frac=args.residual_frac)
    if args.arm == "vanilla":
        return PandaVanillaEnv()
    return PandaWindowEnv()


def evaluate(model, args) -> None:
    scen = sc.load(args.scenarios)
    rl_env = make_env(args)
    try:
        rows = run_scenarios(rl_env.base, RLPolicy(rl_env, model), list(range(78)),
                             scen, args.method)
    finally:
        rl_env.close()
    append_results(rows, args.results)
    print(f"{args.method}: reached {sum(r['reached'] for r in rows)}/78")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=["residual", "warm", "cold", "vanilla"])
    p.add_argument("--clone", default=None,
                   help="residual: the delta clone. warm: the SQUASH clone. cold: omit.")
    p.add_argument("--out", default=None, help="default data/panda_<arm>_<steps//1000>k.zip")
    p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--residual-frac", type=float, default=1.0)
    p.add_argument("--init-log-std", type=float, default=math.log(0.1),
                   help="warm arm's behaviour std at t=0 (see train_reacher_window.py)")
    p.add_argument("--anneal-entropy-from", type=int, default=None, metavar="STEP",
                   help="linearly lower SAC's target entropy from this step to "
                        "--steps (see rl/sb3.py::AnnealTargetEntropyCallback)")
    p.add_argument("--anneal-entropy-to", type=float, default=-21.0,
                   help="final target entropy (nats). Default -21 = 3x SAC's -7 "
                        "default: the reacher's -2 -> -6 sigma shrink, per joint")
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--scenarios", default="data/panda_scenarios_box_v1.npz")
    p.add_argument("--results", default="data/panda_results.csv")
    p.add_argument("--method", default=None, help="results-CSV name; default = --out stem")
    p.add_argument("--eval", default=None, metavar="ZIP", help="score this zip, no training")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default=None)
    args = p.parse_args()

    if args.arm in ("residual", "warm") and not args.clone:
        p.error(f"--arm {args.arm} needs --clone")
    if args.arm in ("cold", "vanilla") and args.clone:
        p.error(f"--arm {args.arm} takes no --clone; that is the whole control")
    args.out = args.eval or args.out or f"data/panda_{args.arm}_{args.steps // 1000}k.zip"
    stem = os.path.splitext(os.path.basename(args.out))[0]
    args.method = args.method or stem

    if args.eval:
        evaluate(load_policy(args.eval, "sac", args.device), args)
        return

    run = init_run(args.wandb_project, name=stem, config=vars(args),
                   tags=["panda", args.arm, "sac"], sync_tensorboard=True)
    venv = DummyVecEnv([lambda: Monitor(make_env(args), filename=f"data/{stem}")])
    model = build_model("sac", venv, args.lr, args.device, args.seed, 1, 0.1,
                        tensorboard_log="data/tb" if run else None)
    try:
        if args.arm == "residual":
            zero_init_actor(model)
        elif args.arm == "warm":
            init_actor_from_clone(model, args.clone, init_log_std=args.init_log_std,
                                  device=args.device)
        model.learn(total_timesteps=args.steps, progress_bar=False,
                    callback=callbacks(ckpt_cb(f"data/{stem}_ckpt", args.checkpoint_freq),
                                       sb3_callback(run, prefix="panda"),
                                       AnnealTargetEntropyCallback(
                                           args.anneal_entropy_from, args.steps,
                                           args.anneal_entropy_to)
                                       if args.anneal_entropy_from is not None else None))
        model.save(args.out)
    finally:
        venv.close()
        finish(run)
    print(f"wrote {args.out}")
    evaluate(model, args)


if __name__ == "__main__":
    main()

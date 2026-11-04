"""Train the zero-init RL residual over the frozen clone of the base controller.

SAC rather than TD3: journey 08 records TD3 needing the SAC fallback in the
hard-exploration regime, and this env's reward is dense, which suits SAC's
entropy exploration. The actor's mean head is zero-initialized so the policy
starts bit-for-bit identical to the clone -- the floor is "no worse than clone",
and `tests/test_reacher_residual_env.py` pins that invariant.

    uv run python scripts/train/train_reacher_residual.py --steps 200000
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from reacher.residual_env import ResidualSelectEnv
from rl.sb3 import (
    build_model,
    check_algo,
    ckpt_cb,
    seed_replay_buffer,
    set_actor_lr,
    zero_init_actor,
)
from rl.wb import callbacks, finish, init_run, sb3_callback


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clone", default="data/dagger_clone_r3.pt")
    p.add_argument("--out", default="data/reacher_residual_dagger_200k.zip")
    p.add_argument("--algo", default="sac", choices=["sac", "td3"])
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--residual-frac", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--action-noise-sigma", type=float, default=0.1)
    p.add_argument("--actor-lr", type=float, default=None,
                   help="actor-only learning rate (rl/sb3.py::set_actor_lr); "
                        "critic and alpha keep --lr")
    p.add_argument("--ent-coef", default="auto",
                   help="SAC entropy weight alpha: 'auto' (SB3 default, starts at 1), "
                        "'auto_<init>' (auto-tuned from <init>), or a fixed number")
    p.add_argument("--warmup-steps", type=int, default=0, metavar="M",
                   help="DAWN data-anchored warmup (arXiv:2602.10539): before "
                        "learning, add M transitions of the base clone acting "
                        "alone (zero residual + N(0, --warmup-sigma)) to the "
                        "replay buffer, where they stay. Paper default M = 20000.")
    p.add_argument("--warmup-sigma", type=float, default=0.0,
                   help="noise on the RESIDUAL action; torque spread is "
                        "sigma * --residual-frac")
    p.add_argument("--critic-layernorm", action="store_true",
                   help="DAWN critic normalization (rl/sb3.py::LayerNormCritic)")
    p.add_argument("--learning-starts", type=int, default=100,
                   help="uniform-random steps before learning (SB3 default 100). "
                        "--warmup-steps sets 0 (rl/sb3.py::seed_replay_buffer), so "
                        "0 here is the warmup arms' matched no-data control")
    p.add_argument("--monitor", default="data/reacher_residual")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--checkpoint-freq", type=int, default=25_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default=None,
                   help="log this run to Weights & Biases (opt-in)")
    args = p.parse_args()

    algo = check_algo(args.algo)
    run = init_run(args.wandb_project, name=os.path.basename(args.out),
                   config=vars(args), tags=["reacher", "residual", algo]
                   + (["warmup"] if args.warmup_steps else [])
                   + (["criticLN"] if args.critic_layernorm else []),
                   sync_tensorboard=True)

    def _factory():
        return Monitor(ResidualSelectEnv(clone_path=args.clone, residual_frac=args.residual_frac,
                                         device=args.device), filename=args.monitor)

    venv = DummyVecEnv([_factory])
    model = build_model(algo, venv, args.lr, args.device, args.seed, 1,
                        args.action_noise_sigma,
                        tensorboard_log="data/tb" if run else None,
                        critic_layernorm=args.critic_layernorm,
                        ent_coef=(args.ent_coef if args.ent_coef.startswith("auto")
                                  else float(args.ent_coef)))
    if args.actor_lr is not None:
        set_actor_lr(model, args.actor_lr)
    zero_init_actor(model)
    model.learning_starts = args.learning_starts
    try:
        if args.warmup_steps:
            warm_env = ResidualSelectEnv(clone_path=args.clone,
                                         residual_frac=args.residual_frac,
                                         device=args.device)
            n = seed_replay_buffer(model, warm_env, lambda _o: np.zeros(2),
                                   args.warmup_steps, args.warmup_sigma, args.seed)
            warm_env.close()
            print(f"warmup: {n} base-clone (zero-residual) transitions in the replay buffer")
        model.learn(total_timesteps=args.steps, progress_bar=False,
                    callback=callbacks(
                        ckpt_cb(args.checkpoint_dir, args.checkpoint_freq),
                        sb3_callback(run, prefix="reacher")))
        model.save(args.out)
    finally:
        venv.close()
        finish(run)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

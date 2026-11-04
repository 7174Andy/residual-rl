"""Warm-start RL on Reacher -- the paper's Eq. 17 -- and its cold-start control.

    u_k = mu(s_k | theta_mu),   theta_mu initialized to the clone's weights

`scripts/train/train_reacher_residual.py` is the Eq. 18 counterpart, where the prior
enters the ACTION and the actor is zero-initialized. Here the prior enters the
WEIGHTS and there is no clone at runtime -- one network per step, not two.

The arm and its control are this one script, switched by a single flag, so
"only the initialization changed" is guaranteed by the code path rather than
argued from a diff:

    uv run python scripts/train/train_reacher_window.py \\
        --clone data/dagger_clone_r3_squash.pt --out data/bcinit_s0.zip
    uv run python scripts/train/train_reacher_window.py --out data/cold_s0.zip

SAC is the default and the only algo the warm start supports --
`init_actor_from_clone` is SAC-only and raises on a TD3 actor. `--algo td3`
remains usable for the cold arm. Journey 10 measured the optimizer choice on
this repo's systems, and every published Reacher arm is SAC; the paper uses
DDPG, whose explicit noise process `N` SAC's policy standard deviation plays
the role of, which is why `--init-log-std` exists at all (see
rl/sb3.py::init_actor_from_clone).

The 400k default is deliberate, and differs from `train_reacher_vanilla.py`'s
200k CLI default. Journey 13 retrained both published RL arms to 400k after
finding their curves had visibly not plateaued at 200k, so the artifacts this
script must be comparable against (`data/reacher_van_ckpt_400k/`,
`data/reacher_ckpt_seeds/resf2_s*`) are 400k runs. The vanilla script's default
is stale relative to its own published results; matching the published budget
matters more than matching a sibling's default.
"""
from __future__ import annotations

import argparse
import math
import os

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from reacher.window_env import CloneWindowEnv
from rl.sb3 import (
    build_model,
    check_algo,
    ckpt_cb,
    init_actor_from_clone,
    seed_replay_buffer,
    set_actor_lr,
)
from rl.clone import load_clone
from rl.wb import callbacks, finish, init_run, sb3_callback


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clone", default=None,
                   help="squash-headed clone to initialize the actor from. "
                        "OMIT for the cold-start control -- that is the only "
                        "difference between the two arms.")
    p.add_argument("--out", default="data/reacher_bcinit_400k.zip")
    p.add_argument("--algo", default="sac", choices=["sac", "td3"])
    p.add_argument("--learning-starts", type=int, default=100,
                   help="uniform-random steps before learning (SB3 default 100). "
                        "0 keeps them out of a warm arm's buffer: they drive most "
                        "of its first-updates drift (measured, memory "
                        "warmstart-dip-mechanism)")
    p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--action-noise-sigma", type=float, default=0.1)
    p.add_argument("--init-log-std", type=float, default=math.log(0.1),
                   help="behaviour std at t=0 for the warm arm; ignored when "
                        "--clone is omitted. Default log(0.1) sits on SAC's own "
                        "target entropy (-2 nats -> sigma ~ 0.089), so alpha "
                        "has little reason to inflate the policy away from the "
                        "prior in the opening steps.")
    p.add_argument("--actor-lr", type=float, default=None,
                   help="actor-only learning rate (rl/sb3.py::set_actor_lr); "
                        "critic and alpha keep --lr")
    p.add_argument("--ent-coef", default="auto",
                   help="SAC entropy weight alpha: 'auto' (SB3 default, starts at 1), "
                        "'auto_<init>' (auto-tuned from <init>), or a fixed number")
    p.add_argument("--warmup-steps", type=int, default=0, metavar="M",
                   help="DAWN data-anchored warmup (arXiv:2602.10539): before "
                        "learning, add M transitions of the clone acting alone "
                        "(+ N(0, --warmup-sigma)) to the replay buffer, where they "
                        "stay. Needs --clone. Paper default M = 20000.")
    p.add_argument("--warmup-sigma", type=float, default=0.0)
    p.add_argument("--critic-layernorm", action="store_true",
                   help="DAWN critic normalization: LayerNorm after each hidden "
                        "layer of both Q heads (rl/sb3.py::LayerNormCritic)")
    p.add_argument("--monitor", default="data/reacher_window")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--checkpoint-freq", type=int, default=25_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default=None,
                   help="log this run to Weights & Biases (opt-in)")
    args = p.parse_args()

    algo = check_algo(args.algo)
    arm = "bcinit" if args.clone else "cold"
    if args.warmup_steps:
        if not args.clone:
            p.error("--warmup-steps needs --clone: the warmup policy is the clone")
        arm += f"+warmup{args.warmup_steps // 1000}k" + (
            f"-sig{args.warmup_sigma:g}" if args.warmup_sigma else "")
    if args.critic_layernorm:
        arm += "+criticLN"
    if args.actor_lr is not None:
        arm += f"+alr{args.actor_lr:g}"
    if args.ent_coef != "auto":
        arm += f"+ent{args.ent_coef}"
    if args.learning_starts != 100:
        arm += f"+ls{args.learning_starts}"
    run = init_run(args.wandb_project, name=os.path.basename(args.out),
                   config=vars(args), tags=["reacher", "window", arm, algo],
                   sync_tensorboard=True)

    venv = DummyVecEnv([lambda: Monitor(CloneWindowEnv(), filename=args.monitor)])
    model = build_model(algo, venv, args.lr, args.device, args.seed, 1,
                        args.action_noise_sigma,
                        tensorboard_log="data/tb" if run else None,
                        critic_layernorm=args.critic_layernorm,
                        ent_coef=(args.ent_coef if args.ent_coef.startswith("auto")
                                  else float(args.ent_coef)))
    if args.actor_lr is not None:
        set_actor_lr(model, args.actor_lr)
    model.learning_starts = args.learning_starts
    try:
        # Inside the try: init_actor_from_clone raises ValueError on a
        # non-squash checkpoint, a shape mismatch, or a non-SAC actor, and the
        # env and W&B run must still be torn down when it does.
        if args.clone:
            init_actor_from_clone(model, args.clone,
                                  init_log_std=args.init_log_std,
                                  device=args.device)
        if args.warmup_steps:
            clone = load_clone(args.clone, device="cpu")
            warm_env = CloneWindowEnv()
            n = seed_replay_buffer(model, warm_env, clone.predict, args.warmup_steps,
                                   args.warmup_sigma, args.seed)
            warm_env.close()
            print(f"warmup: {n} clone transitions in the replay buffer")
        model.learn(total_timesteps=args.steps, progress_bar=False,
                    callback=callbacks(
                        ckpt_cb(args.checkpoint_dir, args.checkpoint_freq),
                        sb3_callback(run, prefix="reacher")))
        model.save(args.out)
    finally:
        venv.close()
        finish(run)
    print(f"wrote {args.out}  (arm: {arm})")


if __name__ == "__main__":
    main()

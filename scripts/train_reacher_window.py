"""Warm-start RL on Reacher -- the paper's Eq. 17 -- and its cold-start control.

    u_k = mu(s_k | theta_mu),   theta_mu initialized to the clone's weights

`scripts/train_reacher_residual.py` is the Eq. 18 counterpart, where the prior
enters the ACTION and the actor is zero-initialized. Here the prior enters the
WEIGHTS and there is no clone at runtime -- one network per step, not two.

The arm and its control are this one script, switched by a single flag, so
"only the initialization changed" is guaranteed by the code path rather than
argued from a diff:

    uv run python scripts/train_reacher_window.py \\
        --clone data/dagger_clone_r3_squash.pt --out data/bcinit_s0.zip
    uv run python scripts/train_reacher_window.py --out data/cold_s0.zip

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
    AnnealTargetEntropyCallback,
    FreezeActorCallback,
    build_model,
    check_algo,
    ckpt_cb,
    init_actor_from_clone,
)
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
    p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--action-noise-sigma", type=float, default=0.1)
    p.add_argument("--init-log-std", type=float, default=math.log(0.1),
                   help="behaviour std at t=0 for the warm arm; ignored when "
                        "--clone is omitted. Default log(0.1) sits on SAC's own "
                        "target entropy (-2 nats -> sigma ~ 0.089), so alpha "
                        "has little reason to inflate the policy away from the "
                        "prior in the opening steps.")
    p.add_argument("--freeze-actor-steps", type=int, default=0,
                   help="hold the actor's lr at 0 for this many timesteps while "
                        "the critic fits the warm-started policy's returns. 0 "
                        "disables it, which is the cold-critic arm. Only "
                        "meaningful with --clone.")
    p.add_argument("--monitor", default="data/reacher_window")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--checkpoint-freq", type=int, default=25_000)
    p.add_argument("--anneal-entropy-from", type=int, default=None,
                   metavar="STEP",
                   help="linearly lower SAC's target entropy from this "
                        "timestep to --steps, ending at --anneal-entropy-to. "
                        "Motivation: the greedy mean parks 11-31 mm outside "
                        "tolerance while sampling from pi reaches (13/19 tail "
                        "failures rescued); shrinking sigma late forces the "
                        "mean onto the sampled behaviour. SAC only.")
    p.add_argument("--anneal-entropy-to", type=float, default=-6.0,
                   help="final target entropy (nats). Default -6: sigma "
                        "~ e^((-6+2)/2) = 0.14x the -2-nats default, per dim.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default=None,
                   help="log this run to Weights & Biases (opt-in)")
    args = p.parse_args()

    algo = check_algo(args.algo)
    if args.clone:
        arm = "bcinit"
        if args.freeze_actor_steps:
            arm += f"-frozen{args.freeze_actor_steps}"
    else:
        arm = "cold"
    if args.anneal_entropy_from is not None:
        if algo != "sac":
            p.error("--anneal-entropy-from is SAC-only (target_entropy is "
                    "an SAC attribute)")
        arm += f"-anneal{args.anneal_entropy_from // 1000}k"
    run = init_run(args.wandb_project, name=os.path.basename(args.out),
                   config=vars(args), tags=["reacher", "window", arm, algo],
                   sync_tensorboard=True)

    venv = DummyVecEnv([lambda: Monitor(CloneWindowEnv(),
                                        filename=args.monitor)])
    model = build_model(algo, venv, args.lr, args.device, args.seed, 1,
                        args.action_noise_sigma,
                        tensorboard_log="data/tb" if run else None)
    try:
        # Inside the try: init_actor_from_clone raises ValueError on a
        # non-squash checkpoint, a shape mismatch, or a non-SAC actor, and the
        # env and W&B run must still be torn down when it does.
        if args.clone:
            init_actor_from_clone(model, args.clone,
                                  init_log_std=args.init_log_std,
                                  device=args.device)
        model.learn(total_timesteps=args.steps, progress_bar=False,
                    callback=callbacks(
                        ckpt_cb(args.checkpoint_dir, args.checkpoint_freq),
                        sb3_callback(run, prefix="reacher"),
                        FreezeActorCallback(args.freeze_actor_steps)
                        if args.freeze_actor_steps else None,
                        AnnealTargetEntropyCallback(
                            args.anneal_entropy_from, args.steps,
                            args.anneal_entropy_to)
                        if args.anneal_entropy_from is not None else None))
        model.save(args.out)
    finally:
        venv.close()
        finish(run)
    print(f"wrote {args.out}  (arm: {arm})")


if __name__ == "__main__":
    main()

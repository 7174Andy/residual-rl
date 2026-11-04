"""Train the RL residual over the frozen DeePC clone (RL + MPC), plus the
from-scratch (vanilla) RL baseline that learns the same task with no DeePC at all.

TD3 (default) or SAC (fallback for the hard-exploration collapse regime) — both are
SB3 off-policy continuous-control algorithms on the same env/benchmark. This module
holds the unicycle's env construction and training entrypoints; the algorithm-agnostic
SB3 plumbing (model building, checkpointing, actor zero-init, policy loading) lives in
`rl/sb3.py`. The actor's action/mean head is zero-initialized so the initial residual
is 0 and the policy starts identical to the clone (no-regression at init).
"""
from __future__ import annotations

import math
import os

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.sb3 import (
    build_model,
    check_algo,
    ckpt_cb,
    init_actor_from_clone,
    load_policy,
    seed_replay_buffer,
    zero_init_actor,
)
from rl.wb import callbacks, finish, init_run, sb3_callback
from two_wheel_robot.rl.deepc_setup import DEFAULT_LIBRARIES, canonical_action_bounds
from two_wheel_robot.rl.residual_env import ResidualDeePCEnv, WarmDeePCEnv
from two_wheel_robot.rl.wrappers import vanilla_rl_env

load_residual = load_policy  # kept: scripts/eval_residual.py imports this name


def make_residual_env(
    clone_path: str,
    libraries_path: str,
    residual_frac: float = 1.0,
    device: str = "cpu",
    monitor_path: str | None = None,
    warm: bool = False,
    body_obs: bool = False,
    std_obs: bool = False,
    ego_obs: bool = False,
    reward: str = "deepc",
) -> DummyVecEnv:
    """Single-env DummyVecEnv wrapping a Monitored ResidualDeePCEnv (TD3 is off-policy),
    or a `WarmDeePCEnv` when ``warm`` (``clone_path`` is then the squash clone).

    ``monitor_path`` (optional): when set, SB3 writes per-episode returns to
    ``<monitor_path>.monitor.csv`` — the reproducible source for the training-return
    plot (``scripts/plot_training_return.py --monitor``). ``None`` keeps returns
    in-memory only.
    """
    def _factory():
        return Monitor(_rl_env(clone_path, libraries_path, residual_frac, device, warm,
                               body_obs, std_obs, ego_obs, reward),
                       filename=monitor_path, info_keywords=("reached",))

    return DummyVecEnv([_factory])


def _rl_env(clone_path, libraries_path, residual_frac, device, warm, body_obs=False,
            std_obs=False, ego_obs=False, reward="deepc"):
    if warm:
        return WarmDeePCEnv(clone_path, libraries_path, device=device, body_obs=body_obs,
                            std_obs=std_obs, ego_obs=ego_obs, reward=reward)
    return ResidualDeePCEnv(clone_path=clone_path, libraries_path=libraries_path,
                            residual_frac=residual_frac, device=device, reward=reward)


def make_vanilla_env(
    libraries_path: str = DEFAULT_LIBRARIES,
    monitor_path: str | None = None,
) -> DummyVecEnv:
    """Single-env DummyVecEnv over the from-scratch (vanilla) RL env.

    No DeePC and no clone anywhere in this path — the library file is read only for
    the canonical action bounds. See `wrappers.vanilla_rl_env` for the spaces.
    ``monitor_path`` behaves as in `make_residual_env`.
    """
    action_bounds = canonical_action_bounds(libraries_path)

    def _factory():
        return Monitor(vanilla_rl_env(action_bounds), filename=monitor_path,
                       info_keywords=("reached",))

    return DummyVecEnv([_factory])


def train_residual(
    clone_path: str = "data/clone.pt",
    libraries_path: str = "data/libraries_v0.npz",
    out_path: str = "data/residual_td3.zip",
    algo: str = "td3",
    total_timesteps: int = 200_000,
    residual_frac: float = 1.0,
    action_noise_sigma: float = 0.1,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    verbose: int = 1,
    monitor_path: str | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_freq: int = 25_000,
    wandb_project: str | None = None,
    warm: bool = False,
    init_log_std: float = math.log(0.1),
    warmup_steps: int = 0,
    warmup_sigma: float = 0.0,
    critic_layernorm: bool = False,
    body_obs: bool = False,
    std_obs: bool = False,
    cold: bool = False,
    ego_obs: bool = False,
    reward: str = "deepc",
    warmup_failures_only: bool = False,
):
    """Train and save the RL residual (algo='td3' default, 'sac' fallback). Returns the model.

    ``warm``: the Eq. 17 warm-start arm instead -- SAC actor initialized from the squash
    clone at ``clone_path`` over `WarmDeePCEnv`, behaviour std ``exp(init_log_std)``.
    ``warmup_steps`` / ``warmup_sigma`` / ``critic_layernorm``: DAWN (arXiv:2602.10539),
    see `rl/sb3.py::seed_replay_buffer` and `LayerNormCritic`. ``body_obs``: warm arm
    also observes the normalized body-frame obs (`WarmDeePCEnv`). ``cold``: the warm
    arm's env with SB3's random actor init, the control for the clone's weights.
    ``std_obs``: warm arm's features standardized by the clone's stats. With a ``checkpoint_dir``
    the step-0 policy is also saved as ``ckpt_0_steps.zip``.

    ``monitor_path`` (optional): persist per-episode returns to
    ``<monitor_path>.monitor.csv`` for the training-return plot.
    ``checkpoint_dir`` (optional): also snapshot the policy every ``checkpoint_freq``
    steps, for the reach-rate-vs-steps sweep.
    ``wandb_project`` (optional): stream per-episode returns to Weights & Biases.
    """
    algo = check_algo(algo)
    run = init_run(wandb_project, name=out_path.rsplit("/", 1)[-1],
                   config={"algo": algo, "steps": total_timesteps, "seed": seed,
                           "residual_frac": residual_frac, "lr": learning_rate,
                           "noise_sigma": action_noise_sigma, "warm": warm,
                           "warmup_steps": warmup_steps, "critic_layernorm": critic_layernorm},
                   tags=["unicycle", "warm" if warm else "residual", algo], sync_tensorboard=True)
    venv = make_residual_env(clone_path, libraries_path, residual_frac, device=device,
                             monitor_path=monitor_path, warm=warm, body_obs=body_obs,
                             std_obs=std_obs, ego_obs=ego_obs, reward=reward)
    model = build_model(algo, venv, learning_rate, device, seed, verbose,
                         action_noise_sigma,
                         tensorboard_log="data/tb" if run else None,
                         critic_layernorm=critic_layernorm)
    if cold:
        pass
    elif warm:
        extra = int(venv.observation_space.shape[0]) - 6 * venv.envs[0].unwrapped.T_ini \
            - 6 - len(venv.envs[0].unwrapped.anchors)
        init_actor_from_clone(model, clone_path, init_log_std=init_log_std, device=device,
                              extra_inputs=extra, obs_standardized=std_obs)
    else:
        zero_init_actor(model)
    if checkpoint_dir is not None:
        model.save(os.path.join(checkpoint_dir, "ckpt_0_steps.zip"))
    try:
        if warmup_steps:
            # The prior acting alone = the step-0 actor's deterministic action: a zero
            # residual (the clone) for the residual arm, the squash clone for warm.
            warm_env = _rl_env(clone_path, libraries_path, residual_frac, device, warm,
                               body_obs, std_obs, ego_obs, reward)
            n = seed_replay_buffer(model, warm_env,
                                   lambda o: model.predict(o, deterministic=True)[0],
                                   warmup_steps, warmup_sigma, seed,
                                   failures_only=warmup_failures_only)
            warm_env.close()
            print(f"warmup: {n} transitions of the prior acting alone in the replay buffer")
        model.learn(total_timesteps=total_timesteps, progress_bar=False,
                    callback=callbacks(ckpt_cb(checkpoint_dir, checkpoint_freq),
                                       sb3_callback(run, prefix="unicycle")))
        model.save(out_path)
    finally:
        venv.close()
        finish(run)
    return model


def train_vanilla(
    libraries_path: str = DEFAULT_LIBRARIES,
    out_path: str = "data/vanilla_td3.zip",
    algo: str = "td3",
    total_timesteps: int = 200_000,
    action_noise_sigma: float = 0.1,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    verbose: int = 1,
    monitor_path: str | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_freq: int = 25_000,
    wandb_project: str | None = None,
):
    """Train the from-scratch RL baseline (no DeePC, no clone). Returns the model.

    Same algorithm, hyperparameters, spaces, reward and episode budget as the
    residual run, so the only difference measured is "learn the whole controller"
    vs "learn a correction on top of DeePC". No zero-init here: a from-scratch
    actor has nothing to stay close to.
    """
    algo = check_algo(algo)
    run = init_run(wandb_project, name=out_path.rsplit("/", 1)[-1],
                   config={"algo": algo, "steps": total_timesteps, "seed": seed,
                           "lr": learning_rate, "noise_sigma": action_noise_sigma},
                   tags=["unicycle", "vanilla", algo], sync_tensorboard=True)
    venv = make_vanilla_env(libraries_path, monitor_path=monitor_path)
    model = build_model(algo, venv, learning_rate, device, seed, verbose,
                         action_noise_sigma,
                         tensorboard_log="data/tb" if run else None)
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=False,
                    callback=callbacks(ckpt_cb(checkpoint_dir, checkpoint_freq),
                                       sb3_callback(run, prefix="unicycle")))
        model.save(out_path)
    finally:
        venv.close()
        finish(run)
    return model

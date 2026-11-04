"""stable-baselines3 plumbing shared by every system's residual and vanilla runs.

The only module besides `rl/clone.py` that imports torch, and the only one that
imports stable_baselines3. Env construction stays with each system, because the
env is the one part that is not system-agnostic.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.sac.policies import SACPolicy
from torch import nn

_ALGOS = {"td3": TD3, "sac": SAC}


class LayerNormCritic(ContinuousCritic):
    """SB3's twin Q-network with LayerNorm after every hidden Linear (before ReLU).

    DAWN's critic normalization (arXiv:2602.10539): it bounds how far Q can
    extrapolate to actions the data does not cover. `create_mlp` already supports
    `post_linear_modules`; SB3's critic just doesn't expose it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_dim = self.q_networks[0][0].in_features
        net_arch = [m.out_features for m in self.q_networks[0] if isinstance(m, nn.Linear)][:-1]
        for idx in range(self.n_critics):
            q_net = nn.Sequential(*create_mlp(in_dim, 1, net_arch, nn.ReLU,
                                              post_linear_modules=[nn.LayerNorm]))
            self.add_module(f"qf{idx}", q_net)    # replaces the plain head
            self.q_networks[idx] = q_net


class LayerNormSACPolicy(SACPolicy):
    """SACPolicy whose critic (and so its target) is a `LayerNormCritic`.

    A policy class rather than a post-hoc patch so `.save`/`.load` round-trip:
    SB3 stores the policy class in the zip and rebuilds it on load.
    """

    def make_critic(self, features_extractor=None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return LayerNormCritic(**critic_kwargs).to(self.device)


def seed_replay_buffer(model, env, base_action, n_steps: int, sigma: float, seed: int,
                       failures_only: bool = False) -> int:
    """DAWN's data-anchored warmup: roll `base_action(obs) + N(0, sigma)` for
    `n_steps` in `env` (a separate instance from the training env) and add every
    transition to `model`'s replay buffer, where it stays for the whole run.

    Time limits are stored as cut-offs (`TimeLimit.truncated`), so SB3 keeps
    bootstrapping through them exactly as for online data. Sets
    `learning_starts = 0`: the buffer is already full, and SB3's default 100
    uniform-random steps would only dilute the anchor.

    `failures_only` keeps only episodes whose last step did not report
    `info["reached"]`, still `n_steps` transitions in total: an ablation asking
    whether the anchor helps by showing the critic successful finishes.
    """
    rng = np.random.default_rng(seed)
    n, ep = 0, 0
    while n < n_steps:
        obs, _ = env.reset(seed=seed * 100_000 + 90_000 + ep)
        ep += 1
        done, episode, info = False, [], {}
        while not done and (failures_only or n + len(episode) < n_steps):
            a = np.clip(base_action(obs) + sigma * rng.standard_normal(env.action_space.shape),
                        env.action_space.low, env.action_space.high).astype(np.float32)
            nobs, r, term, trunc, info = env.step(a)
            episode.append((obs, nobs, a, r, term, trunc))
            obs, done = nobs, term or trunc
        if failures_only and info.get("reached", False):
            continue
        for o, no, a, r, term, trunc in episode[: n_steps - n]:
            model.replay_buffer.add(o[None], no[None], a[None], np.array([r]),
                                    np.array([term or trunc]),
                                    [{"TimeLimit.truncated": bool(trunc and not term)}])
        n += min(len(episode), n_steps - n)
    model.learning_starts = 0
    return n


def set_actor_lr(model, lr: float) -> None:
    """Give the actor its own learning rate (SB3 uses one rate for every optimizer).

    Wraps `actor.optimizer.step` to set the rate just before each Adam step, because
    SB3's `_update_learning_rate` resets all optimizers at the start of every
    `train()` call. The critic and alpha keep the model's rate.
    """
    opt = model.actor.optimizer
    real_step = opt.step

    def step(*args, **kwargs):
        for group in opt.param_groups:
            group["lr"] = lr
        return real_step(*args, **kwargs)
    opt.step = step


def build_model(algo: str, venv, learning_rate, device, seed, verbose, action_noise_sigma,
                tensorboard_log: str | None = None, critic_layernorm: bool = False,
                ent_coef: str | float = "auto"):
    """Construct the SB3 model. TD3 gets Gaussian action noise; SAC explores via entropy.

    ``tensorboard_log`` makes SB3 write its internal scalars (losses, entropy
    coef) to tensorboard — set it when a W&B run has ``sync_tensorboard=True``.
    """
    Algo = _ALGOS[algo]
    if critic_layernorm and algo == "td3":
        raise ValueError("--critic-layernorm is implemented for SAC only")
    kwargs = dict(
        policy=LayerNormSACPolicy if critic_layernorm else "MlpPolicy", env=venv, learning_rate=learning_rate,
        policy_kwargs=dict(net_arch=[256, 256]), device=device, seed=seed, verbose=verbose,
        tensorboard_log=tensorboard_log,
    )
    if algo != "td3":
        kwargs["ent_coef"] = ent_coef   # SB3 forms: "auto", "auto_<init>", or a fixed float
    if algo == "td3":
        n_actions = int(venv.action_space.shape[0])
        kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=action_noise_sigma * np.ones(n_actions)
        )
    return Algo(**kwargs)


def ckpt_cb(checkpoint_dir: str | None, checkpoint_freq: int):
    """Periodic policy snapshots -> `<dir>/ckpt_<n>_steps.zip`, or None if unset.

    Feeds `scripts/sweep_checkpoints.py`, which measures deterministic reach rate
    part-way through training (the training-return curve is a behaviour-policy
    metric; reach rate at a checkpoint is the deployed-policy one).
    """
    if checkpoint_dir is None:
        return None
    return CheckpointCallback(save_freq=checkpoint_freq, save_path=checkpoint_dir,
                              name_prefix="ckpt")


def check_algo(algo: str) -> str:
    algo = algo.lower()
    if algo not in _ALGOS:
        raise ValueError(f"algo must be one of {sorted(_ALGOS)}, got {algo!r}")
    return algo


def zero_init_actor(model) -> None:
    """Zero the actor's action/mean head so the residual == 0 at t=0 (policy == clone).

    Works for both algos:
      TD3: `actor.mu` is a Sequential `[..., Linear, Tanh]`; zero the last Linear on the
           online *and* target actor. `tanh(0) == 0` -> no correction.
      SAC: `actor.mu` is a single mean-head Linear (no actor target); zero it. Deterministic
           eval returns `tanh(0) == 0`, so the residual starts at 0 even though the
           stochastic policy still explores around it during training.
    """
    actors = [model.policy.actor]
    if getattr(model.policy, "actor_target", None) is not None:
        actors.append(model.policy.actor_target)
    for actor in actors:
        mu = actor.mu
        if isinstance(mu, nn.Linear):          # SAC: mu is the mean head
            last_linear = mu
        else:                                   # TD3: mu is a Sequential ending in Linear+Tanh
            last_linear = None
            for module in mu.modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
        assert last_linear is not None, "actor has no Linear head to zero"
        with torch.no_grad():
            last_linear.weight.zero_()
            last_linear.bias.zero_()


def init_actor_from_clone(model, clone_path: str,
                          init_log_std: float = math.log(0.1),
                          device: str = "cpu", extra_inputs: int = 0,
                          obs_standardized: bool = False) -> None:
    """Initialize an SAC actor's weights from a squash-headed behavioral clone.

    The paper's Eq. 17 ("Warm Start RL", arXiv:2510.03354): the actor starts as
    the MPC surrogate and is then refined by policy gradient. `zero_init_actor`
    is the Eq. 18 counterpart -- there the prior lives in the action, here it
    lives in the weights.

    Three mismatches are closed here:

    * **Output.** SB3 emits `tanh(mu(x))`; `tanh` is nonlinear so no weight
      change undoes it. Closed upstream instead -- the checkpoint must have been
      trained with `train_clone(squash=True)`, which is asserted.
    * **Input.** SB3 feeds the actor RAW observations (`FlattenExtractor` is the
      identity), while the clone standardizes. Folded here: the clone computes
      `W((x-m)/s) + b`, so `W' = W/s` and `b' = b - W @ (m/s)` is the identical
      function of raw `x`. Exact, and the checkpoint on disk is not modified.
    * **`log_std`.** Left at SB3's random init the behaviour std is order 1 on a
      `[-1,1]` action box: greedy eval would still be the clone, but every
      transition in the replay buffer would be noise and the prior would never
      reach the critic. Pinned so the behaviour policy at t=0 is exactly
      `clone + N(0, exp(init_log_std))`, matching Algorithm 1's `+ N` and the
      TD3 arms' `--action-noise-sigma`.

    ``extra_inputs``: the observation is the clone's features followed by this many
    more columns (e.g. a critic-friendly view); the actor's layer-0 weights on them
    start at zero, so the step-0 actor is still exactly the clone.
    ``obs_standardized``: the env already emits the clone's standardized features,
    so layer 0 is copied as is instead of folded.

    In place, like `zero_init_actor`. SAC only: TD3's `mu` is a Sequential and
    would need its own branch.

    Raises:
        ValueError: on a non-squash checkpoint, a shape mismatch, or a non-SAC
            actor. Every check runs BEFORE any weight is written, so a refused
            transfer leaves the actor exactly as SB3 built it.
    """
    from rl.clone import load_clone

    predictor = load_clone(clone_path, device=device)
    if not predictor.squash:
        raise ValueError(
            f"{clone_path} was not trained with squash=True. Its output is "
            f"de-standardized torque, but SB3's actor emits tanh(mu(x)), so the "
            f"transferred policy would be a silently compressed clone. Retrain "
            f"with `scripts/train/train_reacher_clone.py --squash`.")

    actor = model.policy.actor
    if not isinstance(actor.mu, nn.Linear):
        raise ValueError(
            "init_actor_from_clone supports SAC only (actor.mu must be the "
            f"mean-head Linear); got {type(actor.mu).__name__}.")
    dst = [actor.latent_pi[0], actor.latent_pi[2], actor.mu]

    net = predictor.model.net           # Linear, ReLU, Linear, ReLU, Linear
    src = [net[0], net[2], net[4]]
    for i, (s, d) in enumerate(zip(src, dst)):
        want = (s.weight.shape[0], s.weight.shape[1] + (extra_inputs if i == 0 else 0))
        if tuple(d.weight.shape) != want:
            raise ValueError(
                f"layer {i} shape mismatch: clone has "
                f"{tuple(s.weight.shape)}, actor expects "
                f"{tuple(d.weight.shape)}. The clone's input_dim is "
                f"{net[0].in_features} and the env's observation is "
                f"{dst[0].in_features}-D.")

    # Fold the clone's input standardization into layer 0.
    mean = np.asarray(predictor.feat_mean, dtype=np.float64)
    std = np.asarray(predictor.feat_std, dtype=np.float64)
    if obs_standardized:
        mean, std = np.zeros_like(mean), np.ones_like(std)
    w0 = src[0].weight.detach().cpu().numpy().astype(np.float64)
    b0 = src[0].bias.detach().cpu().numpy().astype(np.float64)
    folded_w0 = w0 / std                       # columnwise
    folded_b0 = b0 - w0 @ (mean / std)
    folded_w0 = np.pad(folded_w0, ((0, 0), (0, extra_inputs)))

    with torch.no_grad():
        dst[0].weight.copy_(torch.as_tensor(folded_w0, dtype=torch.float32))
        dst[0].bias.copy_(torch.as_tensor(folded_b0, dtype=torch.float32))
        for s, d in zip(src[1:], dst[1:]):
            d.weight.copy_(s.weight.detach())
            d.bias.copy_(s.bias.detach())
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(float(init_log_std))


def load_policy(path: str, algo: str = "td3", device: str = "cpu"):
    """Load a trained residual (or vanilla) checkpoint with the matching class.

    Keeps the sb3 import confined to `rl/`.
    """
    # `algo` must match the class the checkpoint was trained with; SB3's .load()
    # does not record it in the zip, so a mismatch fails or loads the wrong policy.
    return _ALGOS[check_algo(algo)].load(path, device=device)

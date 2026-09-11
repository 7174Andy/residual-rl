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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from torch import nn

_ALGOS = {"td3": TD3, "sac": SAC}


def build_model(algo: str, venv, learning_rate, device, seed, verbose, action_noise_sigma,
                tensorboard_log: str | None = None):
    """Construct the SB3 model. TD3 gets Gaussian action noise; SAC explores via entropy.

    ``tensorboard_log`` makes SB3 write its internal scalars (losses, entropy
    coef) to tensorboard — set it when a W&B run has ``sync_tensorboard=True``.
    """
    Algo = _ALGOS[algo]
    kwargs = dict(
        policy="MlpPolicy", env=venv, learning_rate=learning_rate,
        policy_kwargs=dict(net_arch=[256, 256]), device=device, seed=seed, verbose=verbose,
        tensorboard_log=tensorboard_log,
    )
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
                          device: str = "cpu") -> None:
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

    In place, like `zero_init_actor`. SAC only: TD3's `mu` is a Sequential and
    would need its own branch, which arrives with the TD3 cell if it is ever run.

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
            f"with `scripts/train_reacher_clone.py --squash`.")

    actor = model.policy.actor
    if not isinstance(actor.mu, nn.Linear):
        raise ValueError(
            "init_actor_from_clone supports SAC only (actor.mu must be the "
            f"mean-head Linear); got {type(actor.mu).__name__}.")

    net = predictor.model.net           # Linear, ReLU, Linear, ReLU, Linear
    src = [net[0], net[2], net[4]]
    dst = [actor.latent_pi[0], actor.latent_pi[2], actor.mu]
    for i, (s, d) in enumerate(zip(src, dst)):
        if s.weight.shape != d.weight.shape:
            raise ValueError(
                f"layer {i} shape mismatch: clone has "
                f"{tuple(s.weight.shape)}, actor expects "
                f"{tuple(d.weight.shape)}. The clone's input_dim is "
                f"{net[0].in_features} and the env's observation is "
                f"{actor.latent_pi[0].in_features}-D.")

    # Fold the clone's input standardization into layer 0.
    mean = np.asarray(predictor.feat_mean, dtype=np.float64)
    std = np.asarray(predictor.feat_std, dtype=np.float64)
    w0 = src[0].weight.detach().cpu().numpy().astype(np.float64)
    b0 = src[0].bias.detach().cpu().numpy().astype(np.float64)
    folded_w0 = w0 / std                       # columnwise
    folded_b0 = b0 - w0 @ (mean / std)

    with torch.no_grad():
        dst[0].weight.copy_(torch.as_tensor(folded_w0, dtype=torch.float32))
        dst[0].bias.copy_(torch.as_tensor(folded_b0, dtype=torch.float32))
        for s, d in zip(src[1:], dst[1:]):
            d.weight.copy_(s.weight.detach())
            d.bias.copy_(s.bias.detach())
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(float(init_log_std))


class FreezeActorCallback(BaseCallback):
    """Hold the actor still for the first `n_steps` timesteps so the critic
    fits the warm-started policy's returns before the actor moves.

    Measured motivation, not a precaution: a warm-started actor reaching 79/120
    collapses to 12/120 within 2000 steps and 3/120 by 4000, because ~1900 actor
    updates run against a randomly-initialized critic. The source paper
    (arXiv:2510.03354 SIII.C) predicts this and recommends pre-training the
    critic; its remedy trains one in simulation for use on hardware, and the
    analogue here — no sim-to-real gap — is policy evaluation of the frozen
    prior before the actor is allowed to move.

    Freezing via the learning rate DOES NOT WORK and was tried first: SAC's
    `train()` calls `_update_learning_rate([actor.optimizer, critic.optimizer])`
    at the top of every call, unconditionally overwriting `param_groups["lr"]`
    with SB3's own (constant) schedule value before any gradient step runs.
    Measured: lr went 0.0 -> 3e-4 inside the very first `train()` call, and the
    actor's `mu.weight` moved by 1.29e-02 over a run "frozen" for its entire
    duration -- a complete no-op that a same-shaped lr-only test could not see.

    So this freezes the optimizer's `step` instead: gradients are still
    computed and `backward()` still runs (so the critic, whose own optimizer
    is untouched, keeps training normally), but the actor's `Adam.step` is
    stubbed to do nothing while frozen. Adam's moment estimates are never
    touched by a no-op step, so the actor resumes from a clean optimizer state
    once released.
    """

    def __init__(self, n_steps: int):
        super().__init__()
        self.n_steps = int(n_steps)
        self._real_step = None
        self._released = False

    def _on_training_start(self) -> None:
        if self.n_steps <= 0:
            self._released = True
            return
        opt = self.model.actor.optimizer
        if self._real_step is None:
            self._real_step = opt.step
        opt.step = lambda *args, **kwargs: None
        self._released = False

    def _on_step(self) -> bool:
        if not self._released and self.num_timesteps >= self.n_steps:
            self.model.actor.optimizer.step = self._real_step
            self._released = True
        return True

    def _on_training_end(self) -> None:
        # Never hand back a model whose optimizer is still stubbed out -- a
        # caller that trains again would silently never update the actor.
        if not self._released and self._real_step is not None:
            self.model.actor.optimizer.step = self._real_step
            self._released = True


class AnnealTargetEntropyCallback(BaseCallback):
    """Linearly lower SAC's target entropy from `start_step` to `total_steps`.

    Measured motivation: on the Reacher tail, the greedy mean parks at a stable
    point 11-31 mm outside the 10 mm tolerance while SAMPLING from the same
    policy reaches on 13/19 of those failures -- alpha's auto-tuning holds the
    policy at its target entropy forever, so the mean never commits to the part
    of the distribution that already succeeds. Ramping `target_entropy` down
    over the last stretch of training shrinks sigma and forces the mean onto
    the sampled behaviour; the actor keeps learning through the ramp, so this
    is annealing, not a post-hoc squash.

    SAC-only, and only meaningful with `ent_coef='auto'` (the default here):
    `model.target_entropy` is read inside every `train()` gradient step by the
    alpha loss, which is why mutating the attribute from a callback works. With
    a fixed ent_coef the attribute is dead and this callback is a silent no-op.
    """

    def __init__(self, start_step: int, total_steps: int, final: float):
        super().__init__()
        self.start_step = int(start_step)
        self.total_steps = int(total_steps)
        self.final = float(final)
        self._initial: float | None = None

    def _on_training_start(self) -> None:
        self._initial = float(self.model.target_entropy)

    def _on_step(self) -> bool:
        if self.num_timesteps > self.start_step:
            span = max(1, self.total_steps - self.start_step)
            frac = min(1.0, (self.num_timesteps - self.start_step) / span)
            assert self._initial is not None
            self.model.target_entropy = (
                self._initial + frac * (self.final - self._initial))
        return True


def load_policy(path: str, algo: str = "td3", device: str = "cpu"):
    """Load a trained residual (or vanilla) checkpoint with the matching class.

    Keeps the sb3 import confined to `rl/`.
    """
    # `algo` must match the class the checkpoint was trained with; SB3's .load()
    # does not record it in the zip, so a mismatch fails or loads the wrong policy.
    return _ALGOS[check_algo(algo)].load(path, device=device)

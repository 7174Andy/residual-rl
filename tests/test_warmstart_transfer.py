"""The transfer must be exact, or "the policy at t=0 IS the clone" is decoration.

Not bit-for-bit: the fold computes `(W/s)x + (b - Wm/s)` where the clone computes
`W((x-m)/s) + b` -- the same function in a different floating-point order.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import reacher  # noqa: F401  registers the Gym ID

SQUASH_CLONE = "data/dagger_clone_r3_squash.pt"


def _skip_if_missing(path):
    from reacher.clone_features import feature_dim
    try:
        stats = torch.load(path, map_location="cpu",
                           weights_only=False)["stats"]
    except FileNotFoundError:
        pytest.skip(f"{path} not trained; run scripts/train/train_reacher_clone.py "
                    f"--squash --out {path}")
    if int(stats["input_dim"]) != feature_dim(5, 0):
        pytest.skip(f"{path} has input_dim {stats['input_dim']}, features are "
                    f"{feature_dim(5, 0)}-D (stale checkpoint)")
    if not stats.get("squash", False):
        pytest.skip(f"{path} was not trained with --squash")


def _tiny_squash_clone(tmp_path, input_dim=43, hidden=(256, 256), out_dim=2):
    """A cheap squash checkpoint with the production shapes, for tests that only
    need the plumbing rather than a good policy."""
    from rl.clone import save_clone, train_clone

    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, input_dim))
    Y = np.tanh(rng.normal(size=(256, out_dim)))
    model, stats, _h = train_clone(X, Y, n_lib=0, hidden=hidden, epochs=3,
                                   seed=0, device="cpu", squash=True)
    path = str(tmp_path / f"tiny_{input_dim}x{out_dim}.pt")
    save_clone(path, model, stats)
    return path


def _sac_on_window_env():
    from stable_baselines3 import SAC

    from reacher.window_env import CloneWindowEnv
    env = CloneWindowEnv()
    model = SAC("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256]),
                device="cpu", seed=0, verbose=0)
    return model, env


def test_actor_matches_the_predictor_on_random_observations(tmp_path):
    from rl.clone import load_clone
    from rl.sb3 import init_actor_from_clone

    path = _tiny_squash_clone(tmp_path)
    model, env = _sac_on_window_env()
    try:
        init_actor_from_clone(model, path)
        predictor = load_clone(path, device="cpu")

        rng = np.random.default_rng(7)
        obs = np.stack([env.observation_space.sample() for _ in range(200)])
        # Keep magnitudes realistic; a uniform sample of the box is already in
        # range, but jitter it so the test is not evaluating one lattice.
        obs = obs + rng.normal(scale=1e-3, size=obs.shape).astype(np.float32)
        obs = np.clip(obs, env.observation_space.low,
                      env.observation_space.high)

        got, _state = model.predict(obs, deterministic=True)
        want = predictor.predict(obs.astype(np.float64))
        assert np.max(np.abs(got - want)) < 1e-5, np.max(np.abs(got - want))
    finally:
        env.close()


def test_log_std_is_pinned_so_the_behaviour_policy_is_clone_plus_noise(tmp_path):
    from rl.sb3 import init_actor_from_clone

    path = _tiny_squash_clone(tmp_path)
    model, env = _sac_on_window_env()
    try:
        init_actor_from_clone(model, path, init_log_std=math.log(0.1))
        actor = model.policy.actor
        assert torch.allclose(actor.log_std.weight,
                              torch.zeros_like(actor.log_std.weight))
        obs = torch.as_tensor(
            np.stack([env.observation_space.sample() for _ in range(4)]),
            dtype=torch.float32)
        _mean, log_std, _kw = actor.get_action_dist_params(obs)
        assert torch.allclose(log_std,
                              torch.full_like(log_std, math.log(0.1)),
                              atol=1e-6)
    finally:
        env.close()


def test_non_squash_checkpoint_is_rejected(tmp_path):
    from rl.clone import save_clone, train_clone
    from rl.sb3 import init_actor_from_clone

    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(128, 43)), rng.normal(size=(128, 2))
    m, stats, _h = train_clone(X, Y, n_lib=0, hidden=(256, 256), epochs=2,
                               seed=0, device="cpu")           # squash=False
    path = str(tmp_path / "plain.pt")
    save_clone(path, m, stats)

    model, env = _sac_on_window_env()
    try:
        with pytest.raises(ValueError, match="squash"):
            init_actor_from_clone(model, path)
    finally:
        env.close()


def test_dimension_mismatch_is_rejected(tmp_path):
    from rl.sb3 import init_actor_from_clone

    path = _tiny_squash_clone(tmp_path, input_dim=8)   # the 8-D obs clone shape
    model, env = _sac_on_window_env()
    try:
        with pytest.raises(ValueError, match="8"):
            init_actor_from_clone(model, path)
    finally:
        env.close()


def test_a_rejected_transfer_leaves_the_actor_untouched(tmp_path):
    """A half-written actor is worse than a refused one.

    The mismatch is at the LAST layer on purpose: input_dim matches, so a
    check-and-write-per-layer implementation would already have written
    latent_pi[0] and latent_pi[2] before discovering the bad mu. Asserting on
    the EARLIER layers is what makes this test prove total validation rather
    than merely early failure.
    """
    from rl.sb3 import init_actor_from_clone

    path = _tiny_squash_clone(tmp_path, input_dim=43, out_dim=3)
    model, env = _sac_on_window_env()
    try:
        actor = model.policy.actor
        before = [actor.latent_pi[0].weight.detach().clone(),
                  actor.latent_pi[2].weight.detach().clone(),
                  actor.mu.weight.detach().clone(),
                  actor.log_std.bias.detach().clone()]
        with pytest.raises(ValueError):
            init_actor_from_clone(model, path)
        after = [actor.latent_pi[0].weight, actor.latent_pi[2].weight,
                 actor.mu.weight, actor.log_std.bias]
        for i, (b, a) in enumerate(zip(before, after)):
            assert torch.equal(b, a), f"layer {i} was mutated by a refused transfer"
    finally:
        env.close()


@pytest.mark.integration
def test_warm_actor_reproduces_the_clone_over_a_full_episode():
    """Roll the warm-started actor in CloneWindowEnv and ClonePolicy in the bare
    env from the same seed. Per-step actions must agree to 1e-5 and the episode
    verdict must be identical -- divergence compounds, so a per-step tolerance
    that holds for 50 steps is the real invariant."""
    import gymnasium as gym

    from reacher.eval import ClonePolicy
    from reacher.window_env import CloneWindowEnv
    from rl.clone import load_clone
    from rl.sb3 import build_model, init_actor_from_clone

    _skip_if_missing(SQUASH_CLONE)

    win = CloneWindowEnv()
    ref = gym.make("ReacherGoal-v0")
    try:
        model = build_model("sac", win, 3e-4, "cpu", 0, 0, 0.1)
        init_actor_from_clone(model, SQUASH_CLONE)

        obs, info_w = win.reset(seed=21)
        policy = ClonePolicy(load_clone(SQUASH_CLONE, device="cpu"))
        _o, info_r = ref.reset(seed=21)

        warm_best, ref_best = float(info_w["dist"]), float(info_r["dist"])
        warm_reached = ref_reached = False
        for t in range(win.max_steps):
            a_warm, _s = model.predict(obs, deterministic=True)
            a_ref = policy(ref, info_r)
            assert np.max(np.abs(np.asarray(a_warm, dtype=np.float64)
                                 - a_ref)) < 1e-5, f"actions diverged at step {t}"

            obs, _r, _term, trunc_w, info_w = win.step(a_warm)
            _o, _r2, _term2, trunc_r, info_r = ref.step(a_ref)
            warm_best = min(warm_best, float(info_w["dist"]))
            ref_best = min(ref_best, float(info_r["dist"]))
            warm_reached = warm_reached or bool(info_w["reached"])
            ref_reached = ref_reached or bool(info_r["reached"])
            if trunc_w or trunc_r:
                break

        assert warm_reached == ref_reached
        assert warm_best == pytest.approx(ref_best, abs=1e-4)
    finally:
        win.close()
        ref.close()


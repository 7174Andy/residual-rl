"""PandaReachEnv: Gym contract, reward, termination, determinism, DeePC accessors."""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import panda  # noqa: F401  registers the Gym ID
from panda import model as pm
from panda.env import PandaReachEnv


@pytest.fixture
def env():
    e = PandaReachEnv()
    yield e
    e.close()


def test_gym_id_registered():
    e = gym.make("PandaReach-v0")
    assert isinstance(e.unwrapped, PandaReachEnv)
    e.close()


def test_env_checker_passes():
    e = PandaReachEnv()
    check_env(e, skip_render_check=True)
    e.close()


def test_spaces(env):
    assert env.action_space.shape == (7,)
    assert np.allclose(env.action_space.low, -0.2)
    assert np.allclose(env.action_space.high, 0.2)
    assert env.observation_space.shape == (24,)


def test_defaults_match_spec(env):
    assert env.delta_max == 0.2
    assert env.goal_tolerance == 0.05
    assert env.min_start_goal_dist == 0.15
    assert env.max_steps == 150
    assert env.frame_skip == 10
    assert env.reach_bonus == 100.0
    assert np.allclose(env.Q, np.eye(3))
    assert np.allclose(env.R, 1.0e-2 * np.eye(7))
    assert env.dt_ctrl == pytest.approx(0.02)
    assert env.metadata["render_fps"] == 50


def test_dt_ctrl_frame_skip_and_render_fps_stay_in_sync(env):
    """A non-default dt_ctrl must not desync self.dt_ctrl from what
    frame_skip actually rounds to, or from the render_fps derived from it --
    see panda/env.py's dt_ctrl comment for why these three have to agree.
    """
    e = PandaReachEnv(dt_ctrl=0.025)
    try:
        assert e.frame_skip == pm.frame_skip(e.model, 0.025)
        assert e.dt_ctrl == pytest.approx(e.frame_skip * e.model.opt.timestep)
        # render_fps is an int (Gym's metadata convention), so it's the
        # rounded rate, not an exact match to the (generally non-integer)
        # true rate -- what must hold is that it's derived from this
        # instance's own dt_ctrl, not the class-level default of 50.
        assert e.metadata["render_fps"] == round(1.0 / e.dt_ctrl)
        assert e.metadata["render_fps"] != 50
    finally:
        e.close()
    # The default env from the fixture must be untouched (metadata is an
    # instance copy, not a shared mutated class dict).
    assert env.metadata["render_fps"] == 50


def test_reset_is_deterministic_under_seed(env):
    o1, i1 = env.reset(seed=42)
    s1, g1 = env.state.copy(), env.goal.copy()
    o2, i2 = env.reset(seed=42)
    assert np.allclose(o1, o2)
    assert np.allclose(s1, env.state)
    assert np.allclose(g1, env.goal)


def test_reset_different_seeds_differ(env):
    env.reset(seed=1)
    g1 = env.goal.copy()
    env.reset(seed=2)
    assert not np.allclose(g1, env.goal)


def test_reset_state_is_qpos_qvel_concatenated(env):
    env.reset(seed=0)
    assert env.state.shape == (14,)
    assert np.allclose(env.state[:7], env.data.qpos)
    assert np.allclose(env.state[7:], 0.0)  # qvel zeroed on reset


def test_reset_zeroes_last_action_and_step_idx(env):
    env.reset(seed=0)
    assert np.allclose(env.last_action, 0.0)
    assert env.step_idx == 0


def test_reset_start_respects_min_start_goal_dist(env):
    for seed in range(30):
        env.reset(seed=seed)
        assert np.linalg.norm(env.y - env.goal) >= env.min_start_goal_dist


def test_min_start_goal_dist_relaxed_uses_a_tested_candidate(env, monkeypatch):
    """The rejection loop must sample, then check -- not check, then sample --
    or an exhausted loop can silently keep an untested final draw as q0 while
    still reporting `min_dist_relaxed`. Force every draw to fail the distance
    check and confirm: exactly 100 calls to sample_config (not 101), and the
    q0/tip actually used are the last call's return value, i.e. a candidate
    that was itself checked and rejected.
    """
    import panda.env as penv

    calls = []
    real_sample_config = penv.sample_config

    def spy(model, data, rng, lo, hi, tip_site_id):
        result = real_sample_config(model, data, rng, lo, hi, tip_site_id)
        calls.append(result)
        return result

    monkeypatch.setattr(penv, "sample_config", spy)
    env.min_start_goal_dist = 1e6  # impossible to satisfy -> exhausts all 100 attempts
    _, info = env.reset(seed=0, options={"goal": np.array([0.4, 0.0, 0.5])})

    assert len(calls) == 100
    assert info["min_dist_relaxed"] is True
    expected_q0, expected_tip = calls[-1]
    assert np.allclose(env.data.qpos, expected_q0)
    assert np.allclose(env.y, expected_tip)


def test_goals_lie_in_reachable_shell(env):
    lo_r, hi_r = pm.TIP_RADIUS_RANGE
    for seed in range(50):
        env.reset(seed=seed)
        assert lo_r <= float(np.linalg.norm(env.goal)) <= hi_r
        assert env.goal[2] >= pm.MIN_TIP_Z


def test_options_round_trip(env):
    q = 0.5 * (pm.safe_box(env.model)[0] + pm.safe_box(env.model)[1])
    goal = np.array([0.4, 0.2, 0.5])
    env.reset(seed=0, options={"qpos": q, "goal": goal})
    assert np.allclose(env.data.qpos, q)
    assert np.allclose(env.goal, goal)


def test_options_qpos_is_clipped_not_rejected(env):
    lo, hi = pm.safe_box(env.model)
    env.reset(seed=0, options={"qpos": hi + 10.0, "goal": np.array([0.4, 0.0, 0.5])})
    assert np.allclose(env.data.qpos, hi)


def test_safe_box_property_matches_the_model_function(env):
    lo, hi = env.safe_box
    lo_expected, hi_expected = pm.safe_box(env.model)
    assert np.allclose(lo, lo_expected) and np.allclose(hi, hi_expected)
    # Returns copies -- mutating the result must not touch the env's own box.
    lo[0] = 999.0
    assert env.safe_box[0][0] != 999.0


def test_action_is_clipped_and_recorded_post_clip(env):
    env.reset(seed=0)
    env.step(np.full(7, 5.0))
    assert np.allclose(env.last_action, 0.2)
    env.step(np.full(7, -5.0))
    assert np.allclose(env.last_action, -0.2)


def test_info_ctrl_matches_clip_of_qpos_before_plus_action(env):
    """info["ctrl"] is what the plant actually received: clip(qpos_before + u,
    safe_box). It must not be confused with info["action"], which is u itself
    (see panda/env.py's module docstring).
    """
    env.reset(seed=0)
    qpos_before = env.data.qpos.copy()
    u = np.full(7, 0.1)  # within delta_max, so last_action == u exactly
    _, _, _, _, info = env.step(u)
    lo, hi = env.safe_box
    expected = np.clip(qpos_before + u, lo, hi)
    assert np.allclose(info["action"], u)
    assert np.allclose(info["ctrl"], expected)


def test_info_ctrl_reflects_box_clip_when_it_fires(env):
    """Pins the trap Important #2 exists to document: when the box clip
    fires, info["ctrl"] - qpos_before diverges from info["action"] (the raw
    u). A data-collection script that records `action` as `u` on a step like
    this would pair it with a `y` the recorded `u` never produced.
    """
    lo, hi = env.safe_box
    env.reset(seed=0, options={"qpos": hi, "goal": np.array([0.4, 0.0, 0.5])})
    qpos_before = env.data.qpos.copy()
    assert np.allclose(qpos_before, hi)
    u = np.full(7, env.delta_max)  # pushes further past the upper bound
    _, _, _, _, info = env.step(u)
    assert np.allclose(info["ctrl"], hi, atol=1e-9)
    assert not np.allclose(info["ctrl"] - qpos_before, info["action"])


def test_reward_matches_formula(env):
    env.reset(seed=0)
    u = np.full(7, 0.05)
    _, reward, term, _, info = env.step(u)
    err = info["y"] - info["y_ref"]
    expected = -(err @ env.Q @ err + u @ env.R @ u)
    if info["reached"]:
        expected += env.reach_bonus
    assert reward == pytest.approx(expected, abs=1e-9)
    assert term == info["reached"]


def test_tip_is_fresh_after_step(env):
    """mj_step leaves site_xpos pre-integration; env.step must refresh it."""
    env.reset(seed=0)
    before = env.y.copy()
    env.step(np.full(7, 0.2))
    after = env.y.copy()
    assert not np.allclose(before, after)
    # y must equal FK of the *post-step* qpos, computed independently.
    fresh = mujoco.MjData(env.model)
    fresh.qpos[:] = env.data.qpos
    fresh.qvel[:] = env.data.qvel
    mujoco.mj_forward(env.model, fresh)
    assert np.allclose(after, fresh.site_xpos[env.tip_site_id], atol=1e-9)


def test_terminates_exactly_on_reach(env):
    env.reset(seed=0)
    # Place the goal on the current tip: the next step must terminate.
    env.goal = env.y.copy()
    _, reward, term, trunc, info = env.step(np.zeros(7))
    assert info["distance"] < env.goal_tolerance
    assert term is True and trunc is False
    assert reward > 0  # the reach bonus dominates a near-zero stage cost


def test_truncates_at_max_steps_and_never_both(env):
    e = PandaReachEnv(max_steps=5, goal_tolerance=1e-9)
    e.reset(seed=0)
    for k in range(5):
        _, _, term, trunc, info = e.step(np.zeros(7))
        assert not (term and trunc)
    assert trunc is True and term is False
    assert info["step_idx"] == 5
    e.close()


def test_info_keys(env):
    _, info = env.reset(seed=0)
    for key in ("state", "goal", "y", "y_ref", "pos_error", "distance",
                "action", "ctrl", "step_idx", "reached", "min_dist_relaxed",
                "ncon"):
        assert key in info


def test_min_dist_relaxed_is_in_step_info_too(env):
    """min_dist_relaxed used to appear only in reset()'s info, never step()'s
    -- an inconsistent schema across a rollout. It must be the same value
    throughout the episode it was decided in.
    """
    _, reset_info = env.reset(seed=0)
    _, _, _, _, step_info = env.step(np.zeros(7))
    assert "min_dist_relaxed" in step_info
    assert step_info["min_dist_relaxed"] == reset_info["min_dist_relaxed"]


def test_observation_layout(env):
    obs, _ = env.reset(seed=0)
    assert np.allclose(obs[:7], env.data.qpos, atol=1e-6)
    assert np.allclose(obs[7:14], 0.0)
    assert np.allclose(obs[14:17], env.y - env.goal, atol=1e-6)
    assert np.allclose(obs[17:24], 0.0)


def test_y_ref_is_the_goal(env):
    env.reset(seed=0)
    assert np.allclose(env.y_ref, env.goal)


def test_render_returns_rgb_frame():
    e = PandaReachEnv(render_mode="rgb_array")
    e.reset(seed=0)
    frame = e.render()
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    # The arm occupies only a few percent of an otherwise-black frame, so assert
    # *some* lit pixels rather than a brightness average.
    assert (frame.sum(axis=2) > 30).mean() > 0.005
    e.close()


def test_render_none_mode_returns_none(env):
    env.reset(seed=0)
    assert env.render() is None


def test_render_draws_the_goal_marker():
    e = PandaReachEnv(render_mode="rgb_array")
    e.reset(seed=0)
    frame = e.render().astype(int)
    green = (frame[:, :, 1] - frame[:, :, 0] > 25) & (frame[:, :, 1] - frame[:, :, 2] > 25)
    assert green.sum() > 50, "goal marker sphere not visible in the frame"
    e.close()


def test_render_is_repeatable_and_marker_follows_goal():
    e = PandaReachEnv(render_mode="rgb_array")
    e.reset(seed=0)
    a = e.render().copy()
    b = e.render().copy()
    assert np.array_equal(a, b)
    e.goal = e.goal + np.array([0.0, 0.25, 0.0])
    assert not np.array_equal(a, e.render())
    e.close()


def test_q_normalized_maps_the_safe_box_to_pm_one(env):
    """The normalization must be exactly the safe box, not the hardware range.

    Excitation lives inside the safe box, so normalizing by the hardware range
    would leave the q block permanently short of +-1 and quietly reweight it
    against the metre-scale tip block in `lambda_y * ||sigma_y||^2`.
    """
    lo, hi = env.safe_box
    env.reset(seed=0)
    for q, expect in ((lo, -1.0), (hi, 1.0), (0.5 * (lo + hi), 0.0)):
        env.data.qpos[:] = q
        assert env.q_normalized == pytest.approx(np.full(env.nq, expect))
    # And a real configuration lands strictly inside.
    env.reset(seed=3)
    assert np.all(np.abs(env.q_normalized) <= 1.0 + 1e-9)


def test_y_ext_is_tip_then_normalized_q(env):
    """Tip MUST come first: azimuth_key reads y[0], y[1] and must keep working."""
    env.reset(seed=1)
    ext = env.y_ext
    assert ext.shape == (3 + env.nq,)
    assert ext[:3] == pytest.approx(env.y)
    assert ext[3:] == pytest.approx(env.q_normalized)

    from panda.deepc_setup import azimuth_key
    assert azimuth_key(ext) == pytest.approx(azimuth_key(env.y))


def test_y_ref_ext_is_goal_then_zeros(env):
    env.reset(seed=1)
    ref = env.y_ref_ext
    assert ref.shape == (3 + env.nq,)
    assert ref[:3] == pytest.approx(env.goal)
    assert ref[3:] == pytest.approx(np.zeros(env.nq))


def test_extended_output_does_not_change_the_reward(env):
    """`y_ext` is additive: the task, the reward and `y` are untouched.

    This is the guard against the change that was deliberately NOT made -- had `y`
    itself been widened to 10-D, `env.Q` would have had to become diag(I_3, 0_7) in
    the same commit to keep the reward identical, and any slip there would silently
    redefine the task while every reach-rate number kept looking plausible.
    """
    env.reset(seed=5)
    assert env.y.shape == (3,)
    assert env.y_ref.shape == (3,)
    assert env.Q.shape == (3, 3)
    rng = np.random.default_rng(0)
    for _ in range(5):
        u = rng.uniform(-env.delta_max, env.delta_max, env.nq)
        _, reward, _, _, info = env.step(u)
        err = env.y - env.y_ref
        expect = -(err @ env.Q @ err + u @ env.R @ u)
        expect += env.reach_bonus if info["distance"] < env.goal_tolerance else 0.0
        assert reward == pytest.approx(expect)

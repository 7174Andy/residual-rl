import numpy as np
import pytest

CLONE = "data/clone.pt"
LIB = "data/libraries_v0.npz"


@pytest.mark.integration
def test_spaces_and_reset():
    from two_wheel_robot.rl.residual_env import ResidualDeePCEnv

    env = ResidualDeePCEnv(clone_path=CLONE, libraries_path=LIB)
    try:
        assert env.action_space.shape == (2,)
        assert np.allclose(env.action_space.low, -1.0)
        assert np.allclose(env.action_space.high, 1.0)
        # obs = 5 body + 2 u_base = 7 by default, normalized to [-1, 1]
        assert env.observation_space.shape == (7,)
        assert np.allclose(env.observation_space.low, -1.0)
        assert np.allclose(env.observation_space.high, 1.0)

        obs, info = env.reset(seed=0)
        assert isinstance(info, dict)
        assert obs.shape == (7,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert env._u_buf.shape == (env.T_ini, 2)
        assert env._y_buf.shape == (env.T_ini, 3)
        assert env._u_base.shape == (2,)
    finally:
        env.close()


@pytest.mark.integration
def test_obs_without_base_is_5d():
    from two_wheel_robot.rl.residual_env import ResidualDeePCEnv

    env = ResidualDeePCEnv(clone_path=CLONE, libraries_path=LIB, include_base_in_obs=False)
    try:
        assert env.observation_space.shape == (5,)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (5,)
        assert env.observation_space.contains(obs)
    finally:
        env.close()


@pytest.mark.integration
def test_zero_residual_matches_clone_rollout():
    import gymnasium as gym
    import two_wheel_robot.env  # noqa: F401
    from two_wheel_robot.rl.clone import load_clone
    from two_wheel_robot.rl.clone_eval import run_clone_closed_loop
    from two_wheel_robot.rl.deepc_setup import build_canonical_deepc
    from two_wheel_robot.rl.residual_env import ResidualDeePCEnv

    _deepc, info = build_canonical_deepc(libraries_path=LIB)
    predictor = load_clone(CLONE, device="cpu")
    env_c = gym.make("TwoWheelGoal-v0", action_bounds=info["action_bounds"])
    reached_clone, traj_clone = run_clone_closed_loop(predictor, info, env_c, seed=7)
    env_c.close()

    res = ResidualDeePCEnv(clone_path=CLONE, libraries_path=LIB)
    try:
        obs, _ = res.reset(seed=7)
        traj = [res.base.state.copy()]
        term = trunc = False
        last_info: dict = {}
        while not (term or trunc):
            obs, _, term, trunc, last_info = res.step(np.zeros(2, dtype=np.float32))
            traj.append(res.base.state.copy())
    finally:
        res.close()

    traj = np.asarray(traj)
    assert traj.shape == traj_clone.shape
    assert np.allclose(traj, traj_clone, atol=1e-9)
    assert bool(last_info.get("reached", False)) == reached_clone


@pytest.mark.integration
def test_residual_scaling_and_clip():
    from two_wheel_robot.rl.residual_env import ResidualDeePCEnv

    res = ResidualDeePCEnv(clone_path=CLONE, libraries_path=LIB, residual_frac=1.0)
    try:
        res.reset(seed=1)
        u_base = res._u_base.copy()
        res.step(np.ones(2, dtype=np.float32))  # full +residual: u_base + half_range
        applied = res._u_buf[-1]  # buffer's last row holds the applied action
        expected = np.clip(u_base + res.half_range, res.a_low, res.a_high)
        assert np.allclose(applied, expected, atol=1e-9)
    finally:
        res.close()

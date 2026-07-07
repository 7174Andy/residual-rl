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
        assert obs.shape == (7,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert env._u_buf.shape == (env.T_ini, 2)
        assert env._y_buf.shape == (env.T_ini, 3)
        assert env._u_base.shape == (2,)
    finally:
        env.close()

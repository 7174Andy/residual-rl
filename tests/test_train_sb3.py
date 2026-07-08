import numpy as np
import pytest

CLONE = "data/clone.pt"
LIB = "data/libraries_v0.npz"


@pytest.mark.integration
@pytest.mark.parametrize("cls_name", ["TD3", "SAC"])
def test_zero_init_makes_residual_zero(cls_name):
    import stable_baselines3 as sb3

    from two_wheel_robot.rl.train_sb3 import _zero_init_actor, make_residual_env

    Algo = getattr(sb3, cls_name)
    venv = make_residual_env(CLONE, LIB, residual_frac=1.0)
    try:
        model = Algo("MlpPolicy", venv, policy_kwargs=dict(net_arch=[256, 256]),
                     device="cpu", seed=0)
        _zero_init_actor(model)  # TD3: zeros online+target mu; SAC: zeros the mean head
        obs = venv.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert np.allclose(action, 0.0, atol=1e-6)  # deterministic residual starts at 0
    finally:
        venv.close()


@pytest.mark.integration
def test_smoke_train_and_save(tmp_path):
    from two_wheel_robot.rl.train_sb3 import load_residual, train_residual

    out = tmp_path / "residual.zip"
    monitor = tmp_path / "mon"
    model = train_residual(
        clone_path=CLONE, libraries_path=LIB, out_path=str(out),
        total_timesteps=600, device="cpu", seed=0, verbose=0,
        monitor_path=str(monitor),
    )
    assert out.exists()
    # monitor_path persists per-episode returns for the training-return plot
    assert (tmp_path / "mon.monitor.csv").exists()

    loaded = load_residual(str(out), device="cpu")
    obs = np.zeros(model.observation_space.shape, dtype=np.float32)
    a1, _ = loaded.predict(obs, deterministic=True)
    a2, _ = loaded.predict(obs, deterministic=True)
    assert a1.shape == (2,)
    assert np.allclose(a1, a2)  # deterministic predict is reproducible

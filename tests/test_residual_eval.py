import numpy as np
import pytest

CLONE = "data/clone.pt"
LIB = "data/libraries_v0.npz"


@pytest.mark.integration
def test_run_residual_closed_loop_shapes():
    from stable_baselines3 import TD3

    from two_wheel_robot.rl.residual_env import ResidualDeePCEnv
    from two_wheel_robot.rl.residual_eval import run_residual_closed_loop
    from two_wheel_robot.rl.train_sb3 import _zero_init_actor, make_residual_env

    res = ResidualDeePCEnv(clone_path=CLONE, libraries_path=LIB)
    venv = make_residual_env(CLONE, LIB, residual_frac=1.0)
    try:
        model = TD3("MlpPolicy", venv, policy_kwargs=dict(net_arch=[256, 256]),
                    device="cpu", seed=0)
        _zero_init_actor(model)
        reached, traj = run_residual_closed_loop(model, res, seed=3)
        assert isinstance(reached, bool)
        assert traj.ndim == 2 and traj.shape[1] == 3
    finally:
        res.close()
        venv.close()


@pytest.mark.integration
def test_benchmark_keys_and_zero_init_invariant():
    from stable_baselines3 import TD3

    from two_wheel_robot.rl.clone import load_clone
    from two_wheel_robot.rl.deepc_setup import build_canonical_deepc
    from two_wheel_robot.rl.residual_env import ResidualDeePCEnv
    from two_wheel_robot.rl.residual_eval import benchmark
    from two_wheel_robot.rl.train_sb3 import _zero_init_actor, make_residual_env

    deepc, info = build_canonical_deepc(libraries_path=LIB)
    predictor = load_clone(CLONE, device="cpu")
    res = ResidualDeePCEnv(clone_path=CLONE, libraries_path=LIB)
    venv = make_residual_env(CLONE, LIB, residual_frac=1.0)
    try:
        model = TD3("MlpPolicy", venv, policy_kwargs=dict(net_arch=[256, 256]),
                    device="cpu", seed=0)
        _zero_init_actor(model)  # residual == 0 -> residual policy behaves like clone
        rep = benchmark(model, deepc, predictor, res, info, seeds=[0, 1, 2])
    finally:
        res.close()
        venv.close()

    for k in ("n", "deepc_reach_rate", "clone_reach_rate", "residual_reach_rate",
              "deepc_ci", "clone_ci", "residual_ci", "mcnemar_residual_vs_clone",
              "regressions", "rescued", "traj_dev_vs_clone_median"):
        assert k in rep
    # zero residual == clone, so residual outcomes equal clone outcomes exactly
    assert rep["residual_reach"] == rep["clone_reach"]
    assert rep["regressions"] == 0
    assert rep["rescued"] == 0

"""DAWN pieces in rl/sb3.py: the LayerNorm critic and the data-anchored warmup."""
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from reacher.window_env import CloneWindowEnv
from rl.sb3 import build_model, load_policy, seed_replay_buffer


def _model(layernorm: bool):
    return build_model("sac", DummyVecEnv([CloneWindowEnv]), 3e-4, "cpu", 0, 0, 0.1,
                       critic_layernorm=layernorm)


def _n_layernorm(module) -> int:
    return sum(isinstance(m, nn.LayerNorm) for m in module.modules())


def test_layernorm_critic_and_target_survive_save_load(tmp_path):
    m = _model(True)
    assert _n_layernorm(m.critic) == 4            # 2 heads x 2 hidden layers
    assert _n_layernorm(m.critic_target) == 4
    assert _n_layernorm(m.actor) == 0
    opt_params = {id(p) for g in m.critic.optimizer.param_groups for p in g["params"]}
    assert opt_params == {id(p) for p in m.critic.parameters()}
    for a, b in zip(m.critic.parameters(), m.critic_target.parameters()):
        assert torch.equal(a, b)

    m.save(tmp_path / "ln.zip")
    m2 = load_policy(str(tmp_path / "ln.zip"), algo="sac")
    assert _n_layernorm(m2.critic) == 4
    obs = torch.as_tensor(np.stack([m.observation_space.sample() for _ in range(8)]))
    act = torch.rand(8, 2) * 2 - 1
    with torch.no_grad():
        assert torch.allclose(m.critic(obs, act)[0], m2.critic(obs, act)[0])


def test_flags_off_is_plain_sac():
    assert _n_layernorm(_model(False).critic) == 0
    with pytest.raises(ValueError):
        build_model("td3", DummyVecEnv([CloneWindowEnv]), 3e-4, "cpu", 0, 0, 0.1,
                    critic_layernorm=True)


def test_seed_replay_buffer_rows_actions_and_timeouts():
    m = _model(False)
    env = CloneWindowEnv()
    base = lambda o: np.full(2, 0.3)                                  # noqa: E731
    n = seed_replay_buffer(m, env, base, 60, sigma=0.0, seed=0)
    buf = m.replay_buffer
    assert n == 60 and buf.pos == 60 and m.learning_starts == 0
    assert np.allclose(buf.actions[:60, 0], 0.3)                     # sigma 0 = base alone
    ends = np.flatnonzero(buf.dones[:60, 0])
    assert list(ends) == [49]                                        # 50-step episode
    assert buf.timeouts[49, 0] == 1                                  # cut-off, not terminal
    assert np.allclose(buf.next_observations[0, 0], buf.observations[1, 0])


def test_set_actor_lr_survives_sb3_lr_reset():
    from rl.sb3 import set_actor_lr

    m = _model(False)
    set_actor_lr(m, 1e-5)
    m.learn(total_timesteps=150)          # SB3 resets every optimizer's lr each train()
    assert m.actor.optimizer.param_groups[0]["lr"] == 1e-5
    assert m.critic.optimizer.param_groups[0]["lr"] == 3e-4

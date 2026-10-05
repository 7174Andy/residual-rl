import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from panda import scenarios as sc  # noqa: E402
from panda.clone_data import prime_buffers, rollout  # noqa: E402
from panda.clone_features import feature_dim  # noqa: E402
from panda.model import load_model, safe_box  # noqa: E402
from panda.selectdpc import make_select_controller, panda_bank  # noqa: E402


@pytest.fixture(scope="module")
def rig():
    """A tiny bank -- enough for the QP to be well-posed, fast enough for CI."""
    from panda.env import PandaReachEnv
    from panda.task_bank import collect_task_bank, for_select_dpc

    env = PandaReachEnv()
    try:
        payload = for_select_dpc(collect_task_bank(env, n_traj=40, T=40, seed=0))
    finally:
        env.close()
    model, data = load_model()
    bank = panda_bank(payload, T_ini=5, N=12, stride=4)
    return model, data, bank


@pytest.fixture(scope="module")
def scen():
    return sc.load()


def _ctrl(model, bank):
    c = make_select_controller(bank, model, n_cols=50, n_max=1)
    c.carry_prediction = False
    return c


def test_prime_buffers_tiles_the_start():
    q0 = np.arange(7, dtype=float)
    y0 = np.arange(10, dtype=float)
    u_buf, y_buf = prime_buffers(q0, y0, T_ini=5)
    assert u_buf.shape == (5, 7) and y_buf.shape == (5, 10)
    # u primes with the START TARGET, not zeros: u is an absolute q_des here,
    # and a zero row would be the command "go to the origin" -- a real motion.
    assert np.allclose(u_buf, np.tile(q0, (5, 1)))
    assert np.allclose(y_buf, np.tile(y0, (5, 1)))


def test_rollout_shapes_and_full_horizon(rig, scen):
    model, data, bank = rig
    rec = rollout(model, data, _ctrl(model, bank),
                  np.asarray(scen["qpos"][0], float),
                  np.asarray(scen["goal"][0], float),
                  max_steps=6, tol=0.05, T_ini=5)
    assert rec["features"].shape == (6, feature_dim(5))
    assert rec["actions"].shape == (6, 7)
    assert rec["steps"] == 6
    assert 0.0 <= rec["clip_rate"] <= 1.0


def test_rollout_runs_full_horizon_even_after_reaching(rig, scen):
    """The station-keeping regime is exactly what the residual must improve.

    A tol of 1e9 means "reached" on step 1; the episode must still run all its
    steps, or those rows never enter the dataset.
    """
    model, data, bank = rig
    rec = rollout(model, data, _ctrl(model, bank),
                  np.asarray(scen["qpos"][0], float),
                  np.asarray(scen["goal"][0], float),
                  max_steps=4, tol=1e9, T_ini=5)
    assert rec["reached"] is True
    assert rec["steps"] == 4 and rec["features"].shape[0] == 4


def test_labels_are_in_the_safe_box(rig, scen):
    model, data, bank = rig
    lo, hi = safe_box(model)
    rec = rollout(model, data, _ctrl(model, bank),
                  np.asarray(scen["qpos"][0], float),
                  np.asarray(scen["goal"][0], float),
                  max_steps=5, tol=0.05, T_ini=5)
    assert np.all(rec["actions"] >= lo - 1e-9)
    assert np.all(rec["actions"] <= hi + 1e-9)


def test_dagger_corrects_the_experts_buffer(rig, scen):
    """The expert must see the APPLIED target, not the one it proposed.

    SelectDPC.act slides its own past buffer with the target IT computed. Under
    a driving policy that is not what the plant received, so from step 2 onward
    the expert would be answering about a trajectory that never happened -- and
    nothing raises. This pins the correction that prevents it.
    """
    model, data, bank = rig
    lo, hi = safe_box(model)
    ctrl = _ctrl(model, bank)
    seen = []

    def policy(_features, q):
        # A deliberately WRONG driver, far from anything the expert would pick.
        seen.append(q.copy())
        return q + 0.05

    rec = rollout(model, data, ctrl, np.asarray(scen["qpos"][0], float),
                  np.asarray(scen["goal"][0], float),
                  max_steps=4, tol=0.05, T_ini=5, policy=policy)
    assert len(seen) == 4
    expected_applied = np.clip(seen[-1] + 0.05, lo, hi)
    assert np.allclose(ctrl._u_buf[-1], expected_applied, atol=1e-9)
    # Labels are the EXPERT's, so they must differ from the policy's action.
    assert not np.allclose(rec["actions"][-1], expected_applied)


def test_dagger_features_carry_the_applied_target(rig, scen):
    """The clone's own u_ini block must hold the applied target too.

    Training on features whose past disagrees with the plant would teach the
    clone a history it will never see at deployment.
    """
    model, data, bank = rig
    lo, hi = safe_box(model)
    applied = []

    def policy(_features, q):
        applied.append(np.clip(q + 0.05, lo, hi))
        return q + 0.05

    rec = rollout(model, data, _ctrl(model, bank),
                  np.asarray(scen["qpos"][0], float),
                  np.asarray(scen["goal"][0], float),
                  max_steps=4, tol=0.05, T_ini=5, policy=policy)
    # Feature row t holds the buffer BEFORE step t, so row 3's newest u_ini
    # entry is the target applied at step 2.
    assert np.allclose(rec["features"][3][28:35], applied[2], atol=1e-9)

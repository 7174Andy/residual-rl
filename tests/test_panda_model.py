"""panda/model.py: safe box, FK, the delta-control law, and config sampling."""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from panda import model as pm


@pytest.fixture(scope="module")
def md():
    return pm.load_model()


def test_model_shape(md):
    m, _ = md
    assert (m.nq, m.nv, m.nu, m.na) == (7, 7, 7, 0)
    assert m.opt.timestep == pytest.approx(0.002)


def test_frame_skip_is_ten(md):
    m, _ = md
    assert pm.frame_skip(m, 0.02) == 10


def test_frame_skip_rejects_subtimestep_period(md):
    m, _ = md
    with pytest.raises(ValueError, match="dt_ctrl"):
        pm.frame_skip(m, 0.0001)


def test_safe_box_strictly_inside_joint_range(md):
    m, _ = md
    lo, hi = pm.safe_box(m)
    assert lo.shape == hi.shape == (7,)
    assert np.all(lo > m.jnt_range[:, 0])
    assert np.all(hi < m.jnt_range[:, 1])
    assert np.all(hi > lo)
    # 10% trimmed at BOTH ends leaves 80% of each span.
    span = m.jnt_range[:, 1] - m.jnt_range[:, 0]
    assert np.allclose(hi - lo, 0.8 * span)


def test_tip_position_matches_independent_forward(md):
    m, data = md
    tid = pm.tip_id(m)
    data.qpos[:] = m.key_qpos[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(m, data)
    got = pm.tip_position(data, tid)

    fresh = mujoco.MjData(m)
    fresh.qpos[:] = m.key_qpos[0]
    mujoco.mj_forward(m, fresh)
    assert np.allclose(got, fresh.site_xpos[tid], atol=1e-9)
    # `home` keyframe tip, measured during design.
    assert np.allclose(got, [0.5545, 0.0, 0.6245], atol=1e-3)


def test_tip_position_returns_a_copy(md):
    m, data = md
    tid = pm.tip_id(m)
    mujoco.mj_forward(m, data)
    got = pm.tip_position(data, tid)
    got[0] = 999.0
    assert data.site_xpos[tid][0] != 999.0


def test_apply_delta_never_leaves_safe_box(md):
    m, data = md
    lo, hi = pm.safe_box(m)
    data.qpos[:] = 0.5 * (lo + hi)
    # Deltas far larger than delta_max must still land inside the box.
    for u in (np.full(7, 50.0), np.full(7, -50.0), np.zeros(7)):
        ctrl = pm.apply_delta(data, u, lo, hi)
        assert ctrl.shape == (7,)
        assert np.all(ctrl >= lo - 1e-12) and np.all(ctrl <= hi + 1e-12)
        assert np.allclose(data.ctrl, ctrl)


def test_apply_delta_zero_holds_position(md):
    m, data = md
    lo, hi = pm.safe_box(m)
    q = 0.5 * (lo + hi)
    data.qpos[:] = q
    ctrl = pm.apply_delta(data, np.zeros(7), lo, hi)
    assert np.allclose(ctrl, q)


def test_sample_config_respects_constraints(md):
    m, data = md
    lo, hi = pm.safe_box(m)
    tid = pm.tip_id(m)
    rng = np.random.default_rng(0)
    for _ in range(200):
        q, tip = pm.sample_config(m, data, rng, lo, hi, tid)
        assert np.all(q >= lo) and np.all(q <= hi)
        assert tip[2] >= pm.MIN_TIP_Z
        assert data.ncon == 0
        r = float(np.linalg.norm(tip))
        assert pm.TIP_RADIUS_RANGE[0] <= r <= pm.TIP_RADIUS_RANGE[1]


def test_sample_config_is_deterministic_under_seed(md):
    m, data = md
    lo, hi = pm.safe_box(m)
    tid = pm.tip_id(m)
    a = pm.sample_config(m, data, np.random.default_rng(3), lo, hi, tid)
    b = pm.sample_config(m, data, np.random.default_rng(3), lo, hi, tid)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_sample_config_raises_with_diagnosis_when_impossible(md):
    m, data = md
    tid = pm.tip_id(m)
    # A degenerate box pinned at a measured config whose tip sits at z = -0.087,
    # below MIN_TIP_Z, with no self-collision. Every draw must be rejected.
    lo = np.array([-0.373, 1.375, -0.392, -2.333, 1.308, 1.179, 0.305])
    hi = lo.copy()
    with pytest.raises(RuntimeError, match="tip below"):
        pm.sample_config(m, data, np.random.default_rng(0), lo, hi, tid, max_attempts=5)


def test_scene_is_visual_only(md):
    """`load_model(scene=True)` must not perturb a single dynamics quantity.

    The scene exists so rendered frames show the arm against a backdrop instead
    of the void. Everything measured in this repo -- the safe box, the servo
    constants, `data/panda_libraries_v0.npz` -- was measured on the bare model, so
    a scene that changed contacts or actuator gains would silently invalidate all
    of it. Hence a direct comparison rather than trust in `contype=0`.
    """
    bare, bare_d = md
    scene, scene_d = pm.load_model(scene=True)

    assert (scene.nq, scene.nv, scene.nu, scene.na) == (bare.nq, bare.nv, bare.nu, bare.na)
    assert np.array_equal(scene.jnt_range, bare.jnt_range)
    assert np.array_equal(scene.actuator_gainprm, bare.actuator_gainprm)
    assert np.array_equal(scene.actuator_biasprm, bare.actuator_biasprm)
    assert np.array_equal(scene.actuator_ctrlrange, bare.actuator_ctrlrange)
    assert scene.opt.timestep == bare.opt.timestep
    assert np.array_equal(scene.key_qpos, bare.key_qpos)

    # The floor is present, and invisible to the collision system.
    floor = scene.geom(pm.SCENE_FLOOR_GEOM)
    assert floor.contype[0] == 0 and floor.conaffinity[0] == 0
    assert scene.ngeom == bare.ngeom + 1

    # Same pose in, same contacts and same FK out.
    for q in (bare.key_qpos[0], np.zeros(7)):
        for m, d in ((bare, bare_d), (scene, scene_d)):
            d.qpos[:] = q
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
        assert scene_d.ncon == bare_d.ncon
        assert np.array_equal(
            scene_d.site_xpos[pm.tip_id(scene)], bare_d.site_xpos[pm.tip_id(bare)]
        )


def test_scene_model_steps_identically(md):
    """A rollout must be bit-identical with and without the scene.

    Complements the static check rather than subsuming it -- this trajectory never
    descends to z=0, so it would NOT catch a colliding floor (verified: making the
    floor collide fails only the test above). What it does catch is the scene
    perturbing integration itself, since `scene=True` reaches the compiler through
    `from_xml_string` with in-memory assets instead of `from_xml_path`.

    Worth checking because `PandaReachEnv` picks `scene=` from `render_mode`: a
    rendered run and a headless run load different models, and if they diverged at
    all, a recorded video would not be footage of the episode the CSV scored.
    """
    bare, _ = md
    scene, _ = pm.load_model(scene=True)
    rng = np.random.default_rng(0)
    lo, hi = pm.safe_box(bare)
    ctrls = rng.uniform(lo, hi, size=(20, 7))

    traj = []
    for m in (bare, scene):
        d = mujoco.MjData(m)
        d.qpos[:] = m.key_qpos[0]
        mujoco.mj_forward(m, d)
        qs = []
        for c in ctrls:
            d.ctrl[:] = c
            for _ in range(10):
                mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            qs.append(np.concatenate([d.qpos, d.qvel, d.site_xpos[pm.tip_id(m)]]))
        traj.append(np.asarray(qs))

    assert np.array_equal(traj[0], traj[1])

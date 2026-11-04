import mujoco
import numpy as np

from quadruped.checks import run_held, tracking_ok
from quadruped.gait import LEGS, SIDE, TrotGait, leg_fk, leg_ik
from quadruped.model import Go1Sim


def test_loads_and_timing():
    sim = Go1Sim()
    assert sim.model.nu == 12 and sim.frame_skip == 10
    assert abs(sim.model.opt.timestep - 0.002) < 1e-12


def test_actuator_index_by_name():
    sim = Go1Sim()
    for i, leg in enumerate(LEGS):
        for j, part in enumerate(("hip", "thigh", "calf")):
            name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, sim.act_idx[i, j])
            assert name == f"{leg}_{part}"


def test_ik_matches_mujoco_fk():
    sim = Go1Sim()
    sim.reset()
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = np.column_stack([rng.uniform(-0.3, 0.3, 4), rng.uniform(0.4, 1.3, 4), rng.uniform(-2.4, -1.2, 4)])
        sim.data.qpos[7:] = 0  # set joints by actuator->joint mapping, not by assumed qpos order
        for i in range(4):
            for j in range(3):
                jid = sim.model.actuator_trnid[sim.act_idx[i, j], 0]
                sim.data.qpos[sim.model.jnt_qposadr[jid]] = q[i, j]
        mujoco.mj_kinematics(sim.model, sim.data)
        for i in range(4):
            np.testing.assert_allclose(sim.foot_rel_hip()[i], leg_fk(q[i], SIDE[i]), atol=1e-6)


def test_stands_without_falling():
    sim = Go1Sim()
    sim.reset()
    g = TrotGait()
    g.reset()
    stand = np.array([leg_ik(g.nominal[i], SIDE[i]) for i in range(4)])
    for _ in range(100):  # 2 s, held stand pose
        sim.step(stand)
    assert not sim.fell()


def test_reset_is_deterministic():
    def run():
        sim = Go1Sim()
        sim.reset()
        g = TrotGait()
        g.reset()
        for _ in range(100):
            sim.step(g.step(np.array([0.4, 0.0, 0.3])))
        return sim.data.qpos.copy()

    np.testing.assert_array_equal(run(), run())


def test_body_vel_frame_conventions():
    # Trunk yawed +90 deg: world +y is trunk +x, world -x is trunk +y.
    sim = Go1Sim()
    sim.reset()
    sim.data.qpos[3:7] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    sim.data.qvel[0:3] = [-0.3, 0.5, 0.0]  # world frame
    sim.data.qvel[3:6] = [0.0, 0.0, 0.7]   # body frame; body z == world z here
    mujoco.mj_forward(sim.model, sim.data)
    np.testing.assert_allclose(sim.body_vel(), [0.5, 0.3, 0.7], atol=1e-9)
    # Pitched trunk: body-frame angular velocity about trunk x shows up in world z.
    sim.reset()
    th = 0.3  # pitch about +y
    sim.data.qpos[3:7] = [np.cos(th / 2), 0, np.sin(th / 2), 0]
    sim.data.qvel[3:6] = [1.0, 0.0, 0.0]
    mujoco.mj_forward(sim.model, sim.data)
    assert abs(sim.body_vel()[2] - (-np.sin(th))) < 1e-9


def test_tracking_ok_rules():
    assert tracking_ok(np.array([0.5, 0.0, 0.0]), np.array([0.3, 0.02, -0.1]))      # small axes unchecked
    assert not tracking_ok(np.array([0.5, 0.0, 0.0]), np.array([0.2, 0.0, 0.0]))    # >50% short
    assert not tracking_ok(np.array([0.0, 0.0, 0.8]), np.array([0.0, 0.0, -0.8]))   # wrong sign
    assert tracking_ok(np.array([0.1, 0.0, 0.2]), np.array([-0.1, 0.0, 0.0]))       # both below TRACK_MIN


def test_run_held_zero_command_short():
    r = run_held(np.zeros(3), seconds=2.0)
    assert not r["fell"]


def test_fell_counts_knee_contact_not_ankle_brush():
    # Upright trunk above FALL_HEIGHT, legs folded so the knees rest on the floor: fallen.
    sim = Go1Sim()
    sim.reset()
    sim.data.qpos[2] = 0.19
    for i in range(4):
        for j, qj in enumerate((0.0, 0.3, -2.7)):
            jid = sim.model.actuator_trnid[sim.act_idx[i, j], 0]
            sim.data.qpos[sim.model.jnt_qposadr[jid]] = qj
    mujoco.mj_forward(sim.model, sim.data)
    assert sim.trunk_height() > 0.15 and np.all(np.abs(sim.rpy()[:2]) < 0.8)
    assert sim.fell()
    # Trotting forward, the lower-shin capsule brushes the floor next to the foot: not fallen.
    sim.reset()
    g = TrotGait()
    g.reset()
    brushed = False
    for k in range(60):
        sim.step(g.step(np.array([0.5, 0.0, 0.0]) if k >= 25 else np.zeros(3)))
        brushed |= any(sim.floor_geom_id in (c.geom1, c.geom2)
                       and not {c.geom1, c.geom2} & set(sim.foot_geom_ids)
                       for c in sim.data.contact[: sim.data.ncon])
        assert not sim.fell()
    assert brushed  # the case above actually occurred

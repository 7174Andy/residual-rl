import numpy as np
from quadruped.gait import leg_ik, leg_fk, ABD, THIGH, CALF, TrotGait, PHASE_OFFSET, LEGS, Q_LO, Q_HI

def test_ik_fk_round_trip():
    rng = np.random.default_rng(0)
    for side in (1, -1):
        for _ in range(200):
            p = np.array([rng.uniform(-0.15, 0.15),
                          side * ABD + rng.uniform(-0.05, 0.05),
                          rng.uniform(-0.35, -0.18)])
            np.testing.assert_allclose(leg_fk(leg_ik(p, side), side), p, atol=1e-9)

def test_standing_pose_matches_spec():
    q = leg_ik(np.array([0.0, ABD, -0.28]), side=1)
    np.testing.assert_allclose(q, [0.0, 0.854, -1.706], atol=2e-3)
    q_r = leg_ik(np.array([0.0, -ABD, -0.28]), side=-1)
    np.testing.assert_allclose(q_r, [0.0, 0.854, -1.706], atol=2e-3)

def test_ik_unreachable_is_finite():
    q = leg_ik(np.array([0.5, ABD, -0.5]), side=1)   # beyond THIGH + CALF
    assert np.all(np.isfinite(q))

def test_diagonal_pairs_half_cycle_apart():
    g = TrotGait(); g.reset()
    for _ in range(37):
        g.step(np.array([0.5, 0.0, 0.0]))
        fr, fl, rr, rl = (LEGS.index(n) for n in ("FR", "FL", "RR", "RL"))
        assert g.stance[fl] == g.stance[rr] and g.stance[fr] == g.stance[rl]
        assert g.stance[fl] != g.stance[fr]

def test_zero_command_feet_stay_at_nominal():
    g = TrotGait(); g.reset()
    for _ in range(100):
        g.step(np.zeros(3))
        np.testing.assert_allclose(g.feet[:, :2], g.nominal[:, :2], atol=1e-12)
        np.testing.assert_allclose(g.feet[g.stance, 2], -g.height, atol=1e-12)

def test_held_command_stance_slides_by_step_vector():
    g = TrotGait(period=0.4, dt=0.02); g.reset()
    cmd = np.array([0.5, 0.1, 0.0])
    for _ in range(40):                       # 2 cycles: steady state
        g.step(cmd)
    i = LEGS.index("FL")
    while not g.stance[i]: g.step(cmd)        # first stance step
    start = g.feet[i, :2].copy()
    while g.stance[i]: last = g.feet[i, :2].copy(); g.step(cmd)
    # spec: stance goes +d/2 -> -d/2, d = cmd_xy * period/2
    np.testing.assert_allclose(start - last, cmd[:2] * 0.2 - cmd[:2] * 0.02, atol=1e-9)

def test_stance_feet_never_jump():
    g = TrotGait(); g.reset()
    rng = np.random.default_rng(1)
    for _ in range(500):
        cmd = rng.uniform([-0.5, -0.4, -1.5], [1.0, 0.4, 1.5])
        before, was = g.feet.copy(), g.stance.copy()
        g.step(cmd)
        still = was & g.stance
        jump = np.linalg.norm(g.feet[still, :2] - before[still, :2], axis=1)
        assert np.all(jump <= 0.02 * (1.0 + 0.4 + 1.5 * 0.3) + 1e-9)

def test_extreme_command_finite_and_clipped():
    g = TrotGait(); g.reset()
    for _ in range(60):
        q = g.step(np.array([3.0, -2.0, 4.0]))
        assert np.all(np.isfinite(q))
        assert np.all(q >= Q_LO - 1e-12) and np.all(q <= Q_HI + 1e-12)

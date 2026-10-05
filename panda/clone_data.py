"""Clone-data collection for PandaReach: drive, record (features, expert q_des).

This is the ONLY place in the Phase 2/3 pipeline where a QP runs. The frozen
expert costs ~1.8 s/step at the Task 1 labeller settings (~5 s at the frozen
n_max=3, measured on this host), so putting it in an RL loop is out of the
question -- the same obstacle journey 08 named on the unicycle, removed the same
way: clone it once, offline, then train against the clone.

Two collection conventions, inherited rather than invented:

* **`y_t` is recorded before `u_t` is applied**, so `y_{t+1}` is the response to
  `u_t`. Every collection in this repo uses this alignment, and the residual env
  slides its buffer the same way, so the clone is deployed under exactly the
  labelling it was trained on.
* **Episodes run the full horizon, never stopping at first reach.** The dataset
  must contain the station-keeping regime, because that is what the residual is
  being asked to improve. Stopping early would remove those rows entirely.

Labels are `clip(q_des, safe_box)` -- the target the plant actually received.
Labelling the raw pre-clip `q_des` would teach the clone to propose unrealizable
targets whose excess is silently discarded by the servo.
"""
from __future__ import annotations

import numpy as np

from panda.clone_features import featurize
from panda.model import frame_skip, safe_box, tip_id
from panda.qdes import step_qdes, y_ref_for

NQ = 7


def prime_buffers(q0: np.ndarray, y0: np.ndarray, T_ini: int):
    """Initial `(u_buf, y_buf)` for a fresh episode.

    `u` primes with the START TARGET, not zeros. The unicycle and Reacher prime
    `u_ini` with zeros because their inputs are velocities and torques, where
    zero means "do nothing". Here `u` is an ABSOLUTE joint target, so a zero row
    is the command "go to the origin" -- a large, real, and wrong motion, which
    the clone would learn as the normal way an episode opens.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    return np.tile(q0, (T_ini, 1)), np.tile(y0, (T_ini, 1))


def rollout(model, data, ctrl, q0, goal, max_steps: int, tol: float,
            T_ini: int = 5, policy=None, delta_max: float = 0.2,
            noise_sigma: float = 0.0, rng=None) -> dict:
    """One full-horizon episode, recording clone training rows.

    `policy is None`  -- round 0: the EXPERT drives and labels (plain BC).
    `policy` callable -- DAgger round: `policy(features, q) -> q_des (7,)`
    decides where the arm goes; the EXPERT still labels every state visited.

    `noise_sigma > 0` -- DART (Laskey et al. 2017): the APPLIED target is the
    driver's target plus Gaussian noise, while the LABEL stays the expert's
    clean `q_des` at the state actually visited. The expert keeps control, so
    it is perturbed only slightly off its own trajectory and every label is
    produced where it is still competent -- which is the difference from DAgger,
    where the student drives far enough that the expert itself is extrapolating
    (measured here: 36% of clone-visited states inside the validity radius
    against 57% of expert-visited ones). Each perturbed step is a demonstration
    of recovery, which plain collection never contains because the expert never
    deviates.

    Plain behavioral cloning trains only on states the expert visits, so the
    student is accurate exactly where it will never be once it drives itself.
    Measured on the unicycle: disagreement 0.1025 at expert-visited states
    against 0.2815 at its own, a 2.75x gap that collapsed the closed loop.
    DAgger (Ross, Gordon & Bagnell 2011) closes it by aggregating labels on the
    STUDENT's distribution, turning BC's `T^2 * eps` error growth into `T * eps`.

    THE SUBTLETY THAT SILENTLY BREAKS THIS. `SelectDPC.act` slides its own past
    buffer with the target IT computed (`core/selectdpc.py`). When the policy
    drives, that is NOT what the plant received, so from step 2 onward the expert
    would be answering questions about a trajectory that never happened -- and
    nothing raises, it just quietly labels fiction. The expert's `_u_buf[-1]` is
    therefore overwritten with the APPLIED target after every step.

    `SelectDPC` also carries `_tau_prev` across steps, which is a second
    contaminated channel. The Task 1 labeller runs `carry_prediction=False`,
    which clears it each step, so no second correction is needed -- but a caller
    that passes a carrying controller here is on its own.

    Raises `RuntimeError` (propagated from the solver) if the QP fails; the
    caller drops the episode rather than recording a truncated one, which would
    bias the dataset toward states the solver finds easy.
    """
    import mujoco

    lo, hi = safe_box(model)
    tip, fs = tip_id(model), frame_skip(model)
    q0 = np.asarray(q0, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    data.qpos[:] = q0
    data.qvel[:] = 0.0
    data.ctrl[:] = q0
    mujoco.mj_forward(model, data)
    y0 = np.concatenate([q0, data.site_xpos[tip].copy()])

    u_buf, y_buf = prime_buffers(q0, y0, T_ini)
    ctrl.reset(y0, u_initial=q0)
    yref = y_ref_for(goal, NQ)

    feats, acts = [], []
    best = float(np.linalg.norm(y0[NQ:] - goal))
    reached = bool(best < tol)
    clip_hits, n = 0, 0

    for t in range(max_steps):
        n = t + 1
        q = np.asarray(data.qpos).copy()
        y_pre = np.concatenate([q, data.site_xpos[tip].copy()])
        f = featurize(u_buf, y_buf, y_pre, goal, t)
        feats.append(f)

        # The EXPERT labels this state, always -- in round 0 and in DAgger.
        q_des = ctrl.act(y_pre, yref)
        acts.append(np.clip(q_des, lo, hi))

        # ... but in a DAgger round the STUDENT decides where we go next.
        target = q_des if policy is None else np.asarray(
            policy(f, q), dtype=np.float64).reshape(NQ)

        # DART: perturb what the PLANT receives, never what was recorded as the
        # label. `applied` already feeds the buffers and `ctrl._u_buf[-1]` below,
        # so the expert re-plans from where the arm really ended up.
        if noise_sigma > 0.0:
            assert rng is not None, "noise_sigma > 0 requires an rng"
            target = target + rng.normal(0.0, noise_sigma, NQ)

        # How far the commanded target sits from the current joints, against the
        # env's delta cap -- the plant-equivalence statistic (Task 1's gate).
        if float(np.abs(target - q).max()) > delta_max:
            clip_hits += 1

        applied = step_qdes(model, data, target, lo, hi, fs)

        # LOAD-BEARING -- see the docstring. Without this the expert's past
        # diverges from the plant's the moment a policy drives.
        ctrl._u_buf[-1] = applied
        u_buf = np.roll(u_buf, -1, axis=0)
        u_buf[-1] = applied
        y_buf = np.roll(y_buf, -1, axis=0)
        y_buf[-1] = y_pre

        best = min(best, float(np.linalg.norm(data.site_xpos[tip] - goal)))
        reached = reached or bool(best < tol)

    return {"features": np.asarray(feats), "actions": np.asarray(acts),
            "reached": bool(reached), "final": best, "steps": n,
            "clip_rate": clip_hits / max(n, 1)}

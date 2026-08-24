"""The `u = q_des`, `y = [q; p_ee]` DeePC plant, libraries and controller.

This is the anchor-selection plan's interface with one measured change, and it is
NOT the one `panda/env.py` exposes. Three differences from the env, all
load-bearing:

* **Input.** `u_k = q_des,k` -- an *absolute* joint-position target, where
  `PandaReachEnv` takes a delta. The actuators are PD position servos
  (`tau = kp*(ctrl - q) - kd*qdot`), so `ctrl` *is* `q_des` and the plant input is
  faithful by construction, provided the command stays inside the safe box.
  `collect_anchor` clips the excitation to that box for exactly this reason, and
  records the clipped value -- so, unlike the delta interface, there is no
  "recorded u differs from applied ctrl" failure mode to gate on.

  A second consequence matters more than it looks: because the target is
  absolute, the arm's excursion over a horizon does NOT accumulate with N. It is
  bounded by the excitation amplitude. Measured inside a 12-step window at
  `sigma = 0.25`: **0.58-0.73 rad**, against up to 2.40 rad for the delta
  interface at `DELTA_MAX = 0.2`. That keeps the plant in the regime where
  `docs/reference/mujoco-primer.md` section 10 measures ~5-8 mm of curvature
  error rather than ~50 mm.
* **Output.** `y_k = [q_k; p_ee,k]`, 10-D: the joint vector and the tip position.
* **Keying.** Nearest anchor in joint space, not tip azimuth. Since `q` is the
  first 7 components of `y`, the keying quantity is already in the output and
  `AnchorDeePC` reads it straight off -- no `key_fn`, no change to `core/`.

Why `p_ee` and not the plan's scalar `d_g = ||p_ee - p_g||`
-----------------------------------------------------------
The plan specifies a scalar goal-distance output. That was measured against this
same pipeline and it is where the error was coming from:

    region   q RMSE      tip error q alone explains    actual d_g RMSE
      0      0.024 rad             12.2 mm                  57.4 mm
      1      0.020 rad             10.1 mm                  76.2 mm
      2      0.021 rad             10.4 mm                  40.2 mm
      3      0.021 rad             10.5 mm                  51.0 mm

The joint channels predicted fine; the scalar distance channel was 4-6x worse
than the joint error could account for. The error was manufactured by the output
map -- `d_g` compresses three dimensions into one through a square root, and a
2-norm is not differentiable at zero, which is exactly where the controller
finishes. Replacing it with the 3-vector removes both problems and buys a third
thing for free:

**The libraries become goal-free.** `d_g` depends on the goal, so libraries built
on it were valid only for the goal they were collected against, and had to be
rebuilt per episode. `p_ee` does not depend on the goal at all -- the goal enters
only through `y_ref`. One Hankel build serves every goal, and the per-goal
retargeting machinery this module used to carry is gone.

`Q = diag(0_7, I_3)` keeps the tracking cost identical to tip-only: the joint
block informs PREDICTION through the Yp/Yf constraints without entering the
objective, the same trick `deepc_setup(output="ext")` uses.
"""
from __future__ import annotations

import mujoco
import numpy as np

from core.deepc import DeePC
from core.hankel import build_hankel
from panda.model import frame_skip, safe_box, tip_id

# OU excitation around the anchor. No restoring term is needed here (unlike the
# delta interface, where plain OU integrates into the box edge): the command is
# absolute and centred on the anchor, so it cannot drift away from it.
OU_THETA = 0.85
OU_SIGMA = 0.25          # rad, std of the command's excursion from the anchor
DEFAULT_T = 1500

# Weight on the tip block only. The task has no joint-space target.
TIP_WEIGHT = 1.0
# `u` is an absolute joint target, so `u' R u` penalizes distance from q = 0
# rather than control effort -- a genuine wart of the plan's input convention.
# Kept small and non-zero: at this magnitude it acts as regularization with a
# mild pull toward the middle of the safe box (away from the limits), not as a
# meaningful cost term. Raise it and you are asking the arm to prefer q ~ 0.
R_DEFAULT = 1.0e-4


def step_qdes(model, data, q_des, lo, hi, fs, gravity_comp: bool = False) -> np.ndarray:
    """Write `ctrl = clip(q_des, safe_box)`, advance one control step, return ctrl.

    `gravity_comp` offsets the target by `qfrc_bias / kp` before clipping. A
    position servo has no feedforward channel -- the only way to ask it for an
    extra torque `tau` is to lie to it about the target by `tau / kp`, since it
    applies `kp*(ctrl - q)`. `qfrc_bias` is MuJoCo's full bias force (gravity,
    Coriolis, centrifugal), so this cancels the arm's own weight and lets it sit
    where it is commanded instead of sagging below it.

    It matters only at soft gains. At the shipped `kp` the sag is 0.005 rad and
    not worth correcting; at `servo_scale = 0.02` it is 0.27 rad, which is a
    large fraction of the anchor spacing a library is supposed to cover.

    The returned value is the ACTUAL `ctrl`, offset included -- callers that want
    to record the pre-compensation command must keep it themselves. Which of the
    two belongs in a DeePC input is a real choice: `q_des` makes the identified
    plant "arm + gravity compensation", which is what a controller would command,
    and stays smooth in the state, so Willems still applies. Recording `ctrl`
    instead identifies the bare arm but gives the controller a variable it cannot
    set without knowing the state.
    """
    q_des = np.asarray(q_des, dtype=np.float64)
    if gravity_comp:
        q_des = q_des + data.qfrc_bias / model.actuator_gainprm[:, 0]
    ctrl = np.clip(q_des, lo, hi)
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data, nstep=fs)
    # REQUIRED: mj_step returns before the forward pass, so site_xpos would
    # otherwise describe the pre-integration pose (see the MuJoCo primer, sec. 3).
    mujoco.mj_forward(model, data)
    return ctrl


def outputs(q_traj: np.ndarray, tip_traj: np.ndarray) -> np.ndarray:
    """Assemble `y = [q; p_ee]`, shape `(T, nq + 3)`. Goal-independent."""
    return np.hstack([q_traj, tip_traj])


def y_ref_for(goal: np.ndarray, nq: int = 7) -> np.ndarray:
    """`y_ref = [0_nq; goal]`. The joint block is unweighted, so its value is free."""
    return np.concatenate([np.zeros(nq), np.asarray(goal, dtype=np.float64)])


def collect_anchor(
    model, data, anchor: np.ndarray, T: int, rng: np.random.Generator,
    theta: float = OU_THETA, sigma: float = OU_SIGMA, gravity_comp: bool = False,
) -> dict:
    """One local dataset around `anchor` (plan section 6).

    Returns `u` (the applied `q_des`), `q` and `tip`, each length `T`, with `y_t`
    observed *before* `u_t` is applied so `y_{t+1}` is the response to `u_t` --
    the same alignment `panda/data_collection.py` uses.
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    lo, hi = safe_box(model)
    fs, tip = frame_skip(model), tip_id(model)
    anchor = np.clip(np.asarray(anchor, dtype=np.float64), lo, hi)

    data.qpos[:] = anchor
    data.qvel[:] = 0.0
    data.ctrl[:] = anchor
    mujoco.mj_forward(model, data)

    u_traj = np.empty((T, model.nq))
    q_traj = np.empty((T, model.nq))
    tip_traj = np.empty((T, 3))
    e = np.zeros(model.nq)
    n_clip = 0
    for t in range(T):
        e = theta * e + sigma * np.sqrt(1 - theta**2) * rng.standard_normal(model.nq)
        q_traj[t] = data.qpos                       # observed BEFORE u_t
        tip_traj[t] = data.site_xpos[tip]
        cmd = anchor + e
        ctrl = step_qdes(model, data, cmd, lo, hi, fs, gravity_comp=gravity_comp)
        # Under gravity compensation the applied `ctrl` carries the `qfrc_bias/kp`
        # offset, which the controller cannot set without knowing the state -- so
        # what goes in the library is the COMMAND, and the identified plant is
        # "arm + compensation". That is only sound while the post-offset clip
        # stays rare, hence `clip_frac`: check it, do not assume it.
        u_traj[t] = cmd if gravity_comp else ctrl
        n_clip += int(np.any((ctrl <= lo + 1e-12) | (ctrl >= hi - 1e-12)))
    return {"u": u_traj, "q": q_traj, "tip": tip_traj,
            "clip_frac": n_clip / T,
            "sag": float(np.linalg.norm(q_traj - anchor, axis=1).mean())}


def build_libraries(payload: dict, T_ini: int, N: int) -> list:
    """One Hankel library per anchor. Built once -- these do not depend on a goal."""
    n = int(payload["anchors"].shape[0])
    return [
        build_hankel(payload[f"u_{i}"],
                     outputs(payload[f"q_{i}"], payload[f"tip_{i}"]),
                     T_ini=T_ini, N=N)
        for i in range(n)
    ]


class AnchorDeePC(DeePC):
    """DeePC keyed on joint-space distance to the anchors (plan section 7).

    `DeePC._select_index_for` keys on a scalar and picks the nearest anchor *on
    the circle*, which is right for a heading and wrong for a 7-D configuration.
    Overriding just that one method is enough because the past buffer is shared
    across libraries, so a switch costs nothing but a warm-start reset -- and
    because `y = [q; p_ee]` carries `q` in its first `nq` components, the key is
    read straight off the measurement.
    """

    def __init__(self, *args, anchors: np.ndarray, weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchors = np.asarray(anchors, dtype=np.float64)
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.nq = self.anchors.shape[1]

    def _select_index_for(self, y_current: np.ndarray) -> int:
        if self._n_lib == 1:
            return 0
        d = self.anchors - np.asarray(y_current, dtype=np.float64)[: self.nq]
        if self.weights is not None:
            d = d * np.sqrt(self.weights)
        return int(np.argmin((d**2).sum(1)))


def make_controller(
    payload: dict, model, T_ini: int = 5, N: int = 12,
    lambda_g: float = 5e-3, lambda_y: float = 7.5e3, r: float = R_DEFAULT,
    weights=None, solver: str = "SCS", du_max: float | None = None,
) -> tuple[AnchorDeePC, dict]:
    """Build the controller once. It serves every goal -- pass `y_ref_for(goal)`.

    `du_max` bounds |u_j - u_{j-1}| INSIDE the QP, so the optimizer plans a
    trajectory it can actually execute. Contrast clipping the returned action:
    that leaves the plan premised on a move the plant never makes, so the
    predicted trajectory is invalid from its first step. `test_cluster_lti.py`
    measures the envelope this should respect -- superposition holds to ~14-34%
    at command amplitude 0.05 rad, degrading to ~52-61% at 0.25.
    """
    anchors = np.asarray(payload["anchors"], dtype=np.float64)
    libs = build_libraries(payload, T_ini, N)
    nq = anchors.shape[1]
    p_y = nq + 3
    Q = np.zeros((p_y, p_y))
    Q[nq:, nq:] = TIP_WEIGHT * np.eye(3)     # only the tip block is tracked
    lo, hi = safe_box(model)
    deepc = AnchorDeePC(
        libs, anchor_headings=np.zeros(len(libs)), Q=Q, R=r * np.eye(nq),
        T_ini=T_ini, N=N, lambda_g=lambda_g, lambda_y=lambda_y,
        u_bounds=(lo, hi), du_max=du_max, solver=solver, anchors=anchors,
        weights=weights,
    )
    return deepc, {"T_ini": T_ini, "N": N, "p_y": p_y, "nq": nq,
                   "lo": lo, "hi": hi, "Q": Q}


def predict(library, u_ini, y_ini, u_future, lambda_g: float, N: int, p_y: int):
    """Open-loop DeePC prediction (plan section 8), as regularized least squares.

    Solves `min ||g||^2` subject to matching the past window and the known future
    input, in the same Tikhonov form `core/deepc.py` optimizes -- but with no
    control cost and no bounds, since nothing is being chosen here. Returns the
    predicted future output, shape `(N, p_y)`.
    """
    Up, Uf, Yp, Yf = library
    A = np.vstack([Up, Yp, Uf])
    b = np.concatenate([np.ravel(u_ini), np.ravel(y_ini), np.ravel(u_future)])
    g = np.linalg.solve(A.T @ A + lambda_g * np.eye(A.shape[1]), A.T @ b)
    return (Yf @ g).reshape(N, p_y)

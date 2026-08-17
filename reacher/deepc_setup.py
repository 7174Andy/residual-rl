"""DeePC on `Reacher-v5`: collection, libraries, controller.

`u = tau in [-1, 1]^2`, `y = [q; fingertip] in R^4`.

Output shape follows the Panda's `y = [q; p_ee]` for the same two reasons, both
measured there: the joint block makes the state observable through the `Yp`/`Yf`
constraints, and putting the *fingertip* rather than a scalar goal-distance in `y`
keeps the libraries GOAL-FREE -- one Hankel build serves every target, with the
goal entering only through `y_ref`. `Q = diag(0, 0, 1, 1)` so the tracking cost is
fingertip-only; the joint block informs prediction without entering the objective.

Excitation carries a restoring term, and here it is not optional. Under torque the
plant is a double integrator with damping: unlike the Panda's position servo,
nothing pulls the arm back, so plain OU torque random-walks away from the anchor
and the library ends up describing wherever it drifted rather than the anchor's
neighbourhood. `K_RET`/`K_DAMP` form a weak PD toward the anchor -- weak enough
that the excitation still dominates the local response, strong enough to keep the
walk bounded. `collect_anchor` returns the achieved spread so this can be checked
rather than assumed.
"""
from __future__ import annotations

import numpy as np

from core.deepc import DeePC
from core.hankel import build_hankel
from reacher.model import (
    NQ_ARM, config_distance, fingertip, frame_skip, set_state, step_torque, wrap,
)

OU_THETA = 0.85
OU_SIGMA = 0.35        # fraction of the +-1 torque range
K_RET = 0.8            # restoring gain toward the anchor (rad -> torque units)
K_DAMP = 0.15          # velocity damping in the restoring term
DEFAULT_T = 1200

TIP_WEIGHT = 1.0
R_DEFAULT = 1.0e-3     # u IS a torque here, so this is genuine control effort
                       # -- unlike the Panda's q_des, where u'Ru penalized
                       # distance from q = 0 rather than effort.


def outputs(q_traj: np.ndarray, tip_traj: np.ndarray) -> np.ndarray:
    """`y = [q; fingertip]`, shape `(T, 4)`. Goal-independent."""
    return np.hstack([q_traj, tip_traj])


def y_ref_for(goal: np.ndarray) -> np.ndarray:
    """`y_ref = [0, 0, gx, gy]`. The joint block is unweighted, so its value is free."""
    return np.concatenate([np.zeros(NQ_ARM), np.asarray(goal, dtype=np.float64)[:2]])


def collect_anchor(model, data, anchor, T: int, rng, theta: float = OU_THETA,
                   sigma: float = OU_SIGMA, k_ret: float = K_RET,
                   k_damp: float = K_DAMP) -> dict:
    """One local dataset around `anchor`. Returns u, q, tip, and the spread.

    `y_t` is recorded BEFORE `u_t` is applied, so `y_{t+1}` is the response to
    `u_t` -- the alignment every other collection in this repo uses.
    """
    fs = frame_skip(model)
    anchor = np.asarray(anchor, dtype=np.float64)
    set_state(model, data, anchor)

    u_traj = np.empty((T, NQ_ARM))
    q_traj = np.empty((T, NQ_ARM))
    tip_traj = np.empty((T, 2))
    e = np.zeros(NQ_ARM)
    for t in range(T):
        e = theta * e + sigma * np.sqrt(1 - theta**2) * rng.standard_normal(NQ_ARM)
        q = np.asarray(data.qpos[:NQ_ARM]).copy()
        err = np.array([wrap(anchor[0] - q[0]), anchor[1] - q[1]])
        restore = k_ret * err - k_damp * np.asarray(data.qvel[:NQ_ARM])
        q_traj[t] = q
        tip_traj[t] = fingertip(data)
        u_traj[t] = step_torque(model, data, e + restore, fs)
    spread = float(config_distance(anchor, q_traj).max())
    return {"u": u_traj, "q": q_traj, "tip": tip_traj, "spread": spread}


def build_libraries(payload: dict, T_ini: int, N: int) -> list:
    """One Hankel per anchor. Goal-free, so built once for every target."""
    n = int(payload["anchors"].shape[0])
    return [
        build_hankel(payload[f"u_{i}"],
                     outputs(payload[f"q_{i}"], payload[f"tip_{i}"]),
                     T_ini=T_ini, N=N)
        for i in range(n)
    ]


class ReacherDeePC(DeePC):
    """DeePC keyed on wrapped joint-space distance to the anchors.

    `DeePC._select_index_for` keys on a scalar and picks the nearest anchor on the
    circle, which is right for a heading and wrong for a 2-D configuration. The
    key is read straight off `y` because `y = [q; tip]` carries `q` in its first
    two components -- and the distance WRAPS, since joint0 is unlimited.
    """

    def __init__(self, *args, anchors: np.ndarray, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchors = np.asarray(anchors, dtype=np.float64)

    def _select_index_for(self, y_current: np.ndarray) -> int:
        if self._n_lib == 1:
            return 0
        return int(np.argmin(config_distance(np.asarray(y_current)[:NQ_ARM],
                                             self.anchors)))


def make_controller(payload: dict, T_ini: int = 5, N: int = 12,
                    lambda_g: float = 5e-3, lambda_y: float = 7.5e3,
                    r: float = R_DEFAULT, du_max: float | None = None,
                    solver: str = "SCS") -> tuple[ReacherDeePC, dict]:
    """Build the controller once; it serves every target via `y_ref_for(goal)`."""
    anchors = np.asarray(payload["anchors"], dtype=np.float64)
    libs = build_libraries(payload, T_ini, N)
    p_y = NQ_ARM + 2
    Q = np.diag([0.0] * NQ_ARM + [TIP_WEIGHT] * 2)
    deepc = ReacherDeePC(
        libs, anchor_headings=np.zeros(len(libs)), Q=Q, R=r * np.eye(NQ_ARM),
        T_ini=T_ini, N=N, lambda_g=lambda_g, lambda_y=lambda_y,
        u_bounds=(-np.ones(NQ_ARM), np.ones(NQ_ARM)), du_max=du_max,
        solver=solver, anchors=anchors,
    )
    return deepc, {"T_ini": T_ini, "N": N, "p_y": p_y, "Q": Q}


def predict(library, u_ini, y_ini, u_future, lambda_g: float, N: int, p_y: int):
    """Open-loop DeePC prediction as regularized least squares (the QP-free gate)."""
    Up, Uf, Yp, Yf = library
    A = np.vstack([Up, Yp, Uf])
    b = np.concatenate([np.ravel(u_ini), np.ravel(y_ini), np.ravel(u_future)])
    g = np.linalg.solve(A.T @ A + lambda_g * np.eye(A.shape[1]), A.T @ b)
    return (Yf @ g).reshape(N, p_y)

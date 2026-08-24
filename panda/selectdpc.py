"""Select-DPC on PandaReach: the `u = q_des`, `y = [q; p_ee]` adapter.

The algorithm lives in `core/selectdpc.py` (system-agnostic). This module only
assembles the Panda's `(u, y)` trajectories from a collection payload and wires
the controller with the Panda's cost, bounds and rate limit.

Why this and not more anchors
-----------------------------
Measured on the anchor pipeline: libraries are excellent within ~0.5 rad of their
anchor (skill 0.93, cos 0.98) and anti-informative beyond ~2 rad (skill -9.9, with
half the predictions pointing the WRONG WAY). Covering a ~5.7-effective-dimensional
configuration set at 0.5 rad needs ~10^5 trajectories, confirmed by three
independent routes. Select-DPC removes the cells entirely, so there is no "outside
a cell": the columns used at step k are chosen to match the trajectory the robot is
currently on.

Two Panda-specific requirements that Reacher does not have:

* **The rate limit is not optional.** `u` is an ABSOLUTE joint target, so nothing
  in `u_bounds` stops the QP asking for a 2 rad jump in one 20 ms tick -- measured
  median 1.9-3.1 rad before `du_max` existed. Reacher's torque input is natively
  bounded and needs none.
* **Trajectory-space distance mixes units badly.** `tau` stacks `q_des` (rad), `q`
  (rad) and `p_ee` (m), and the tip block is ~10x smaller numerically, so a plain
  norm nearly ignores it. `tip_scale` exposes a correction; the default reproduces
  the paper's plain norm so the faithful result is what gets measured first.

A caveat inherited from the anchor work: an earlier `panda/selectdpc.py` scored
columns against the observed PAST and did not iterate. That is closer to the
Time-Windowed DeePC the paper uses as a baseline than to Select-DPC, and the
"selection ties fixed libraries" result it produced does not test this method.
"""
from __future__ import annotations

import numpy as np

from core.selectdpc import SelectDPC, select_predict, trajectory_bank
from panda.model import safe_box
from panda.qdes import R_DEFAULT, TIP_WEIGHT, outputs

__all__ = ["SelectDPC", "select_predict", "panda_bank", "make_select_controller"]


def panda_bank(payload: dict, T_ini: int, N: int, stride: int = 1) -> dict:
    """Pool a `panda/qdes.py` collection payload into a Select-DPC bank."""
    n = int(payload["anchors"].shape[0])
    u_list = [payload[f"u_{i}"] for i in range(n)]
    y_list = [outputs(payload[f"q_{i}"], payload[f"tip_{i}"]) for i in range(n)]
    return trajectory_bank(u_list, y_list, T_ini, N, stride=stride)


def tau_scale(nq: int, T_ini: int, N: int, tip_scale: float) -> np.ndarray:
    """Per-row weights for the trajectory distance, scaling only the tip block.

    `tau = [u_p; y_p; u_f; y_f]` with `y = [q (nq, rad); p_ee (3, m)]`. Metres are
    ~10x smaller than radians here, so the default plain norm weights tip geometry
    far below joint geometry. Returns all-ones when `tip_scale == 1`.
    """

    y_row = np.concatenate([np.ones(nq), np.full(3, tip_scale)])
    return np.concatenate([
        np.ones(T_ini * nq), np.tile(y_row, T_ini),
        np.ones(N * nq), np.tile(y_row, N),
    ])


def make_select_controller(
    bank: dict, model, T_ini: int = 5, N: int = 12, n_cols: int = 300,
    n_max: int = 3, lambda_g: float = 5e-3, lambda_y: float = 7.5e3,
    r: float = R_DEFAULT, du_max: float | None = 0.02, tip_scale: float = 1.0,
    solver: str = "SCS",
) -> SelectDPC:
    """Select-DPC wired for PandaReach. `du_max` defaults ON -- see module docstring."""
    nq = model.nq
    p_y = nq + 3
    Q = np.zeros((p_y, p_y))
    Q[nq:, nq:] = TIP_WEIGHT * np.eye(3)     # tip-only tracking, as in qdes.py
    lo, hi = safe_box(model)
    scale = None if tip_scale == 1.0 else tau_scale(nq, T_ini, N, tip_scale)
    return SelectDPC(
        bank, anchor_headings=np.zeros(1), Q=Q, R=r * np.eye(nq),
        T_ini=T_ini, N=N, lambda_g=lambda_g, lambda_y=lambda_y,
        u_bounds=(lo, hi), du_max=du_max, solver=solver,
        n_cols=n_cols, n_max=n_max, scale=scale,
    )

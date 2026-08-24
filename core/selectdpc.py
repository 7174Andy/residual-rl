"""Select-DPC: per-timestep Hankel-column selection (arXiv:2503.18845).

Naef, Moffat, Eising & Dorfler, *Choose Wisely: Data-driven Predictive Control for
Nonlinear Systems Using Online Data Selection*. System-agnostic, so it lives here
rather than under `panda/` or `reacher/` -- it operates on Hankel blocks and knows
nothing about either robot. Each system supplies its own `(u, y)` trajectories.

Faithful to Algorithm 1 + Algorithm 2, which differ in two ways from the obvious
"pick columns near the current state" reading:

  1. **Selection is against the open-loop PREDICTION, not the observed past.**
     Algorithm 1 line 3 takes `tau~ <- DPC.getLastPrediction()`, and Algorithm 2
     sorts the data by `||tau_i - tau~||` over the FULL length-L trajectory
     `(u_p, y_p, u_f, y_f)`. The data chosen therefore resembles where the
     controller intends to GO. Scoring on the past alone gives something closer to
     the Time-Windowed DeePC the paper uses as a BASELINE.

  2. **It iterates.** Select, solve, re-select against the new prediction, solve
     again, until convergence or `n_max`. The paper likens this to SQP-MPC's
     sequential linearization: each pass re-linearizes about the trajectory the
     previous pass planned.

Naming follows the paper: `n_cols` is its `N_cols` (how many columns are selected)
and `n_max` its iteration cap. They are separate design parameters and it is easy
to conflate them.

Only norm-based selection (Algorithm 2) is implemented. The paper's second method
embeds the data with Isomap first to dodge the curse of dimensionality in the
`(T_ini + N)(m + p)`-dimensional trajectory space -- worth reaching for on the
Panda (dimension 289) more than on Reacher (102).

Units caveat the paper does not discuss: a trajectory stacks inputs and outputs
whose physical units differ (radians against metres, say), so a raw Euclidean
distance weights the blocks by their arbitrary numeric scale. `scale` exposes a
per-row weighting; `None` reproduces the paper's plain norm.
"""
from __future__ import annotations

import numpy as np

from core.deepc import DeePC
from core.hankel import build_hankel


def trajectory_bank(
    u_list: list, y_list: list, T_ini: int, N: int, stride: int = 1
) -> dict:
    """Pool every trajectory's Hankel columns into one bank.

    Columns must be contiguous windows *within* one trajectory, so the Hankels are
    built per-trajectory and concatenated column-wise. Concatenating the raw
    trajectories first would manufacture columns straddling a discontinuity, which
    are not trajectories of the system at all.

    `tau` is `[Up; Yp; Uf; Yf]` -- the full length-L trajectory per column, which
    is the object Algorithm 2 measures distances between. `origin` and `t0` record
    which source trajectory each column came from and at which sample its window
    starts, so a column can be traced back to the state it was collected at --
    which is how `scripts/measure_selection_distance.py` asks how FAR the selected
    data actually is.

    `stride` subsamples columns to bound memory.
    """
    if len(u_list) != len(y_list):
        raise ValueError(f"{len(u_list)} input vs {len(y_list)} output trajectories")
    blocks: list[list] = [[], [], [], []]
    origin, t0 = [], []
    for i, (u, y) in enumerate(zip(u_list, y_list)):
        lib = build_hankel(u, y, T_ini=T_ini, N=N)
        for b in range(4):
            blocks[b].append(np.ascontiguousarray(lib[b][:, ::stride]))
        n_col = blocks[0][-1].shape[1]
        origin.append(np.full(n_col, i))
        # Hankel column j spans data[j : j+L], so a strided column k starts at k*stride.
        t0.append(np.arange(n_col) * stride)
    Up, Uf, Yp, Yf = (np.hstack(b) for b in blocks)
    return {"Up": Up, "Uf": Uf, "Yp": Yp, "Yf": Yf,
            "tau": np.vstack([Up, Yp, Uf, Yf]),
            "origin": np.concatenate(origin), "t0": np.concatenate(t0),
            "n_traj": len(u_list)}


class SelectDPC(DeePC):
    """DeePC whose columns are re-selected each solver iteration (Algorithm 1).

    Constructed with one dummy library of `n_cols` columns so the cached QP has the
    right shape; each inner iteration overwrites that library in place.

    `act()` is overridden rather than `_select_index_for` because Select-DPC needs
    SEVERAL solves per control step while the base `act()` slides the past buffer on
    every call. The buffer is saved and restored around the inner iterations so all
    of them see the same measured past, then slid exactly once at the end.
    """

    def __init__(self, bank: dict, *args, n_cols: int = 300, n_max: int = 3,
                 tol: float = 1e-3, scale: np.ndarray | None = None,
                 carry_prediction: bool = True, **kwargs):
        total = bank["Up"].shape[1]
        if not 1 <= n_cols <= total:
            raise ValueError(f"n_cols must be in [1, {total}]; got {n_cols}")
        dummy = tuple(bank[k][:, :n_cols].copy() for k in ("Up", "Uf", "Yp", "Yf"))
        super().__init__([dummy], *args, **kwargs)
        self.bank = bank
        self.n_cols = int(n_cols)
        self.n_max = int(n_max)
        self.tol = float(tol)
        self.scale = scale
        # Algorithm 1 selects against the PREVIOUS step's prediction, which makes
        # the controller recurrent: its action depends on the whole episode
        # history, not on the last T_ini steps. That is fine for control and fatal
        # for behavioral cloning -- measured, from identical
        # `(u_ini, y_ini, y_current, y_ref)` the carried state moves the action by
        # a median 0.244, which is 0.81x the action's own magnitude and up to the
        # full width of the input box. No feedforward clone over that window can
        # fit a label that its inputs do not determine.
        #
        # `carry_prediction=False` clears it each step, so `act` becomes a pure
        # function of the past window and the reference. Costs about one reach in
        # forty on Reacher (30/40 against 31/40, paired coin-flip) -- consistent
        # with journey 12's finding that n_max=1's gain comes from SELECTING the
        # right data rather than from Algorithm 1's iteration.
        self.carry_prediction = bool(carry_prediction)
        self._tau_prev: np.ndarray | None = None
        # Diagnostics: how many iterations ran, and how concentrated the pick was.
        self.last_iters: int = 0
        self.last_sel: np.ndarray | None = None
        self.last_n_traj_used: int = -1

    def _select(self, tau: np.ndarray) -> np.ndarray:
        """Algorithm 2: the `n_cols` columns closest to `tau` in trajectory space."""
        d = self.bank["tau"] - tau[:, None]
        if self.scale is not None:
            d = d * self.scale[:, None]
        return np.argpartition(np.einsum("ij,ij->j", d, d), self.n_cols - 1)[: self.n_cols]

    def _warm_tau(self, y_current: np.ndarray) -> np.ndarray:
        """Paper's first-step warm start: hold the measurement, assume zero input."""
        return np.concatenate([
            self._u_buf.flatten(), self._y_buf.flatten(),
            np.zeros(self.N * self.m_u), np.tile(y_current, self.N),
        ])

    def reset(self, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self._tau_prev = None          # a new episode invalidates the prediction

    def act(self, y_current: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        y_current = np.asarray(y_current, dtype=np.float64)
        if not self.carry_prediction:
            self._tau_prev = None
        tau = self._tau_prev if self._tau_prev is not None else self._warm_tau(y_current)
        u_buf, y_buf = self._u_buf.copy(), self._y_buf.copy()
    
        u0 = None
        sel = None
        for it in range(self.n_max):
            sel = self._select(tau)
            self._libraries[0] = tuple(
                np.ascontiguousarray(self.bank[k][:, sel])
                for k in ("Up", "Uf", "Yp", "Yf")
            )
            self._g.value = None       # the columns changed; the warm start is stale
            self._u_buf, self._y_buf = u_buf.copy(), y_buf.copy()
            u0 = super().act(y_current, y_ref)
            g = self._g.value
            Up, Uf, Yp, Yf = self._libraries[0]
            tau_new = np.concatenate([Up @ g, Yp @ g, Uf @ g, Yf @ g])
            moved = np.linalg.norm(tau_new - tau) / max(np.linalg.norm(tau), 1e-12)
            tau = tau_new
            self.last_iters = it + 1
            if moved < self.tol:       # converged: re-selecting stopped moving it
                break

        self.last_sel = sel
        self.last_n_traj_used = int(np.unique(self.bank["origin"][sel]).size)
        self._tau_prev = tau
        # Slide the buffer exactly once for the control step, undoing the inner
        # iterations' slides.
        self._u_buf, self._y_buf = u_buf, y_buf
        self._u_buf = np.roll(self._u_buf, -1, axis=0)
        self._u_buf[-1] = u0
        self._y_buf = np.roll(self._y_buf, -1, axis=0)
        self._y_buf[-1] = y_current
        return u0


def select_predict(bank, u_ini, y_ini, u_future, n_cols, lambda_g, N, p_y,
                   scale=None, tau=None):
    """Open-loop prediction under the selection rule -- the QP-free gate.

    Mirrors each system's `predict()` so the skill/cos metrics apply unchanged.
    `tau` is the trajectory to select against; when omitted it is built from the
    past plus the known future input with the future output held at the last
    measurement, which is the paper's first-step warm start. Passing an actual
    prediction reproduces a later Algorithm 1 iteration.
    """
    u_ini, y_ini, u_future = map(np.ravel, (u_ini, y_ini, u_future))
    if tau is None:
        tau = np.concatenate([u_ini, y_ini, u_future,
                              np.tile(y_ini[-p_y:], N)])
    d = bank["tau"] - tau[:, None]
    if scale is not None:
        d = d * scale[:, None]
    sel = np.argpartition(np.einsum("ij,ij->j", d, d), n_cols - 1)[:n_cols]
    Up, Uf, Yp, Yf = (bank[k][:, sel] for k in ("Up", "Uf", "Yp", "Yf"))
    A = np.vstack([Up, Yp, Uf])
    b = np.concatenate([u_ini, y_ini, u_future])
    g = np.linalg.solve(A.T @ A + lambda_g * np.eye(A.shape[1]), A.T @ b)
    return (Yf @ g).reshape(N, p_y), sel

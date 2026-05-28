"""DeePC: data-EnablEd predictive control (Coulson, Lygeros, Dörfler 2019).

The behavioral-systems-theory-based data-driven predictive controller. Given a
length-T offline trajectory `(u_t, y_t)`, the controller predicts future
outputs as linear combinations of columns of the data Hankel matrices and
solves a regularized QP each step to track a reference.

The QP structure here is adapted from the SUMO Deep-LCC implementation
(`CachedDeepLCCSolver`) — which is itself a DeePC specialization for mixed
traffic — with the disturbance signal removed (no exogenous input in the
two-wheel task). Without `e`, the controller reduces to standard DeePC.

QP (per step) — hybrid regularization following arXiv:2603.07395 (Reg-DDPC):

    min       ‖Yf·g − y_ref_flat‖²_{Q̄}  +  ‖Uf·g‖²_{R̄}
    g, σ_y   + λ_g · ‖g‖_1                  ← L1, induces sparsity in g
             + λ_y · ‖σ_y‖_2²               ← L2 on explicit past-output slack

    s.t. Up · g       = u_ini               (hard past-input constraint)
         Yp · g + σ_y = y_ini               (past-output constraint with slack)
         u_min ≤ Uf·g ≤ u_max               (control bounds, optional)

with `Q̄ = I_N ⊗ Q`, `R̄ = I_N ⊗ R`.

The L1 on `g` is from the original DeePC paper (Coulson et al. 2019, Eq. 8);
the L2 on the slack matches 2603.07395 (the paper informing this repo) — they
state `λ_g = 2` and `λ_y = 3·10⁶` as default values, also used here.

The CVXPY problem is built once at construction with `cp.Parameter`s for
`u_ini`, `y_ini`, `y_ref_flat`; subsequent `act()` calls reuse the compiled
problem (DPP-compliant), so per-step solves are fast.
"""

from __future__ import annotations

from typing import Optional

import cvxpy as cp
import numpy as np


class DeePC:
    """Stateful DeePC controller. See module docstring."""

    def __init__(
        self,
        Up: np.ndarray,
        Uf: np.ndarray,
        Yp: np.ndarray,
        Yf: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        T_ini: int = 5,
        N: int = 12,
        lambda_g: float = 2.0,
        lambda_y: float = 3e6,
        u_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
        solver: Optional[str] = None,
    ):
        if T_ini < 1 or N < 1:
            raise ValueError(f"T_ini, N must be >= 1; got {T_ini}, {N}")
        if Up.shape[0] % T_ini:
            raise ValueError(
                f"Up rows ({Up.shape[0]}) must be divisible by T_ini ({T_ini})"
            )
        if Yp.shape[0] % T_ini:
            raise ValueError(
                f"Yp rows ({Yp.shape[0]}) must be divisible by T_ini ({T_ini})"
            )
        m_u = Up.shape[0] // T_ini
        p_y = Yp.shape[0] // T_ini
        n_cols = Up.shape[1]
        if Uf.shape != (N * m_u, n_cols):
            raise ValueError(
                f"Uf shape {Uf.shape}; expected ({N * m_u}, {n_cols})"
            )
        if Yf.shape != (N * p_y, n_cols):
            raise ValueError(
                f"Yf shape {Yf.shape}; expected ({N * p_y}, {n_cols})"
            )
        if Q.shape != (p_y, p_y):
            raise ValueError(f"Q shape {Q.shape}; expected ({p_y}, {p_y})")
        if R.shape != (m_u, m_u):
            raise ValueError(f"R shape {R.shape}; expected ({m_u}, {m_u})")

        self.Up = np.asarray(Up, dtype=np.float64)
        self.Uf = np.asarray(Uf, dtype=np.float64)
        self.Yp = np.asarray(Yp, dtype=np.float64)
        self.Yf = np.asarray(Yf, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.T_ini = int(T_ini)
        self.N = int(N)
        self.m_u = int(m_u)
        self.p_y = int(p_y)
        self.lambda_g = float(lambda_g)
        self.lambda_y = float(lambda_y)
        self.u_bounds = u_bounds
        self.solver = solver

        # Past trajectory buffer — primed by reset().
        self._u_buf: Optional[np.ndarray] = None  # shape (T_ini, m_u)
        self._y_buf: Optional[np.ndarray] = None  # shape (T_ini, p_y)

        # Build the cached parametric problem.
        self._build_problem()

    # ----- QP construction ----------------------------------------------------

    @staticmethod
    def _psd_sqrt(M: np.ndarray) -> np.ndarray:
        """Symmetric matrix square root of a PSD matrix `M`."""
        # Symmetrize, then eigendecompose. Floor any negative eigenvalues to 0
        # (numerical noise on otherwise-PSD inputs).
        M = 0.5 * (M + M.T)
        w, V = np.linalg.eigh(M)
        w = np.clip(w, 0.0, None)
        return V @ np.diag(np.sqrt(w)) @ V.T

    def _build_problem(self) -> None:
        n_cols = self.Up.shape[1]
        g = cp.Variable(n_cols)
        sigma_y = cp.Variable(self.T_ini * self.p_y)

        u_ini_param = cp.Parameter(self.T_ini * self.m_u)
        y_ini_param = cp.Parameter(self.T_ini * self.p_y)
        y_ref_param = cp.Parameter(self.N * self.p_y)

        # Express ‖x‖²_Q as ‖Q^{1/2} x‖²: keeps the cost DPP-compliant so CVXPY
        # caches the compiled problem (per-step solves stay fast).
        Q_sqrt = self._psd_sqrt(self.Q)
        R_sqrt = self._psd_sqrt(self.R)
        Q_sqrt_bar = np.kron(np.eye(self.N), Q_sqrt)
        R_sqrt_bar = np.kron(np.eye(self.N), R_sqrt)

        u_future = self.Uf @ g
        y_future = self.Yf @ g
        y_err = y_future - y_ref_param

        objective = cp.Minimize(
            cp.sum_squares(Q_sqrt_bar @ y_err)
            + cp.sum_squares(R_sqrt_bar @ u_future)
            + self.lambda_g * cp.norm(g, 1)
            + self.lambda_y * cp.sum_squares(sigma_y)
        )

        constraints: list[cp.Constraint] = [
            self.Up @ g == u_ini_param,
            self.Yp @ g + sigma_y == y_ini_param,
        ]
        if self.u_bounds is not None:
            u_min, u_max = self.u_bounds
            u_min = np.asarray(u_min, dtype=np.float64).reshape(self.m_u)
            u_max = np.asarray(u_max, dtype=np.float64).reshape(self.m_u)
            u_min_full = np.tile(u_min, self.N)
            u_max_full = np.tile(u_max, self.N)
            constraints.append(u_future >= u_min_full)
            constraints.append(u_future <= u_max_full)

        self._g = g
        self._sigma_y = sigma_y
        self._u_ini_param = u_ini_param
        self._y_ini_param = y_ini_param
        self._y_ref_param = y_ref_param
        self._problem = cp.Problem(objective, constraints)

    # ----- API ----------------------------------------------------------------

    def reset(
        self,
        y_initial: np.ndarray,
        u_initial: Optional[np.ndarray] = None,
    ) -> None:
        """Prime the past buffer with `T_ini` copies of `(u_initial, y_initial)`.

        This is a "synthetic past": a constant trajectory where the robot has
        held `y_initial` while applying `u_initial` (zeros by default). The QP's
        soft past-output constraint absorbs the resulting inconsistency.
        """
        y_initial = np.asarray(y_initial, dtype=np.float64).reshape(self.p_y)
        if u_initial is None:
            u_initial = np.zeros(self.m_u, dtype=np.float64)
        else:
            u_initial = np.asarray(u_initial, dtype=np.float64).reshape(self.m_u)
        self._u_buf = np.tile(u_initial, (self.T_ini, 1))
        self._y_buf = np.tile(y_initial, (self.T_ini, 1))

    def act(self, y_current: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        """Solve the QP for the current step and return `u_t` (shape `(m_u,)`).

        Args:
            y_current: current observation, shape `(p_y,)`. Used to update the
                past-output buffer after the QP solve, so the next call's
                `y_ini` includes this.
            y_ref: reference output. Either shape `(p_y,)` (broadcast to all
                `N` future steps) or shape `(N, p_y)` (per-step reference).

        Returns:
            `u_t`: applied control input, shape `(m_u,)`.

        Raises:
            RuntimeError: if `reset()` hasn't been called, or the QP fails to
                converge.
        """
        if self._u_buf is None or self._y_buf is None:
            raise RuntimeError("Call reset() before act().")

        y_current = np.asarray(y_current, dtype=np.float64).reshape(self.p_y)
        y_ref = np.asarray(y_ref, dtype=np.float64)
        if y_ref.ndim == 1:
            if y_ref.shape[0] != self.p_y:
                raise ValueError(
                    f"y_ref shape {y_ref.shape}; expected ({self.p_y},) or ({self.N}, {self.p_y})"
                )
            y_ref_flat = np.tile(y_ref, self.N)
        elif y_ref.ndim == 2:
            if y_ref.shape != (self.N, self.p_y):
                raise ValueError(
                    f"y_ref shape {y_ref.shape}; expected ({self.N}, {self.p_y})"
                )
            y_ref_flat = y_ref.flatten()
        else:
            raise ValueError(f"y_ref must be 1-D or 2-D; got ndim={y_ref.ndim}")

        self._u_ini_param.value = self._u_buf.flatten()
        self._y_ini_param.value = self._y_buf.flatten()
        self._y_ref_param.value = y_ref_flat

        self._problem.solve(solver=self.solver, warm_start=True)
        status = self._problem.status
        if status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"DeePC QP failed: status={status}")

        g_val = self._g.value
        assert g_val is not None
        u_future = self.Uf @ g_val
        u_t = u_future[: self.m_u].copy()
        # Solver tolerance can produce sub-microscopic bound violations; clip to
        # guarantee the returned action lies inside u_bounds.
        if self.u_bounds is not None:
            u_t = np.clip(u_t, self.u_bounds[0], self.u_bounds[1])

        # Slide buffers: drop oldest entry, append the new (u_t, y_current).
        self._u_buf = np.roll(self._u_buf, -1, axis=0)
        self._u_buf[-1] = u_t
        self._y_buf = np.roll(self._y_buf, -1, axis=0)
        self._y_buf[-1] = y_current

        return u_t


class LibrarySwitchingDeePC:
    """Orientation-keyed switcher over multiple `DeePC` instances.

    Each underlying `DeePC` is trained on a separate data library — one per
    initial-heading quadrant in the paper (`arXiv:2603.07395` Appendix D). On
    every `act()`, the wrapper inspects the robot's current heading
    `y_current[2]` and routes the call to the controller whose anchor heading
    is the closest (in shortest angular distance). A single past-`(u, y)`
    buffer is shared across all instances — the robot's actual history is
    independent of which controller predicts the future.

    Selecting by *closest anchor* is equivalent to selecting by *quadrant*
    when anchors are quadrant midpoints (the paper's choice: `π/4, 3π/4,
    5π/4, 7π/4`). Closest-anchor is more general and degrades gracefully if
    the anchors aren't midpoints.
    """

    def __init__(
        self,
        controllers: list[DeePC],
        anchor_headings,
        heading_index: int = 2,
    ):
        if not controllers:
            raise ValueError("need at least one controller")
        anchor_headings = np.asarray(anchor_headings, dtype=np.float64)
        if anchor_headings.shape != (len(controllers),):
            raise ValueError(
                f"anchor_headings shape {anchor_headings.shape}; "
                f"expected ({len(controllers)},)"
            )
        m_u = controllers[0].m_u
        p_y = controllers[0].p_y
        T_ini = controllers[0].T_ini
        for i, c in enumerate(controllers[1:], start=1):
            if (c.m_u, c.p_y, c.T_ini) != (m_u, p_y, T_ini):
                raise ValueError(
                    f"controller {i} disagrees on (m_u, p_y, T_ini): "
                    f"{(c.m_u, c.p_y, c.T_ini)} vs {(m_u, p_y, T_ini)}"
                )
        if not 0 <= heading_index < p_y:
            raise ValueError(
                f"heading_index {heading_index} out of range for p_y={p_y}"
            )

        self.controllers = list(controllers)
        self.anchor_headings = anchor_headings
        self.heading_index = int(heading_index)
        self.m_u = m_u
        self.p_y = p_y
        self.T_ini = T_ini

        # Shared past-(u, y) buffer. Filled by reset().
        self._u_buf: np.ndarray | None = None
        self._y_buf: np.ndarray | None = None
        # Diagnostics: which library was used on the most recent act().
        self.last_library_idx: int = -1

    def reset(self, y_initial, u_initial=None) -> None:
        """Prime the shared buffer (and each underlying controller's buffer)."""
        y_initial = np.asarray(y_initial, dtype=np.float64).reshape(self.p_y)
        if u_initial is None:
            u_initial = np.zeros(self.m_u, dtype=np.float64)
        else:
            u_initial = np.asarray(u_initial, dtype=np.float64).reshape(self.m_u)
        self._u_buf = np.tile(u_initial, (self.T_ini, 1))
        self._y_buf = np.tile(y_initial, (self.T_ini, 1))
        for c in self.controllers:
            c.reset(y_initial, u_initial)

    def _select_index(self, heading: float) -> int:
        """Index of the controller whose anchor is closest to `heading` on the circle."""
        # Wrap-aware shortest signed angular difference; pick smallest |diff|.
        diffs = (heading - self.anchor_headings + np.pi) % (2 * np.pi) - np.pi
        return int(np.argmin(np.abs(diffs)))

    def act(self, y_current, y_ref) -> np.ndarray:
        if self._u_buf is None or self._y_buf is None:
            raise RuntimeError("Call reset() before act().")

        y_arr = np.asarray(y_current, dtype=np.float64).reshape(self.p_y)
        idx = self._select_index(float(y_arr[self.heading_index]))
        self.last_library_idx = idx

        # Inject shared buffer into the selected controller, run its QP, pull
        # the updated buffer back.
        chosen = self.controllers[idx]
        chosen._u_buf = self._u_buf.copy()
        chosen._y_buf = self._y_buf.copy()
        u_t = chosen.act(y_arr, y_ref)
        assert chosen._u_buf is not None and chosen._y_buf is not None
        self._u_buf = chosen._u_buf.copy()
        self._y_buf = chosen._y_buf.copy()

        return u_t

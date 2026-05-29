"""DeePC: data-EnablEd predictive control (Coulson, Lygeros, Dörfler 2019).

The behavioral-systems-theory-based data-driven predictive controller. Given a
length-T offline trajectory `(u_t, y_t)`, the controller predicts future
outputs as linear combinations of columns of the data Hankel matrices and
solves a regularized QP each step to track a reference.

This controller follows the reference paper's (arXiv:2603.07395, Appendix D)
**orientation-keyed library switching** strategy. The unicycle is not globally
Koopman-linearizable, so a single Hankel spanning all headings would mix
incompatible local linearizations; instead we keep one data library per
heading region and, each step, route prediction to the library whose anchor
heading is closest to the robot's current heading.

Crucially this is **one** controller with **one** cached QP. The Hankel
matrices `(Up, Uf, Yp, Yf)` of the *active* library are held in `cp.Parameter`s
and their values are swapped per step — rather than compiling a separate
problem per library. CVXPY's DPP rules permit `sum_squares(F @ g)` with `F` a
parameter, so the compiled problem is reused across solves and library swaps.

QP (per step) — hybrid regularization following arXiv:2603.07395 (Reg-DDPC):

    min       ‖Yf·g − y_ref_flat‖²_{Q̄}  +  ‖Uf·g‖²_{R̄}
    g, σ_y   + λ_g · ‖g‖_1                  ← L1, induces sparsity in g
             + λ_y · ‖σ_y‖_2²               ← L2 on explicit past-output slack

    s.t. Up · g       = u_ini               (hard past-input constraint)
         Yp · g + σ_y = y_ini               (past-output constraint with slack)
         u_min ≤ Uf·g ≤ u_max               (control bounds, optional)

with `Q̄ = I_N ⊗ Q`, `R̄ = I_N ⊗ R`. `Up/Uf/Yp/Yf` are `cp.Parameter`s set to
the selected library each step; `u_ini`, `y_ini`, `y_ref_flat` are also
parameters, so the compiled problem (DPP-compliant) is reused.

The L1 on `g` is from the original DeePC paper (Coulson et al. 2019, Eq. 8);
the L2 on the slack matches 2603.07395, which states `λ_g = 2` and
`λ_y = 3·10⁶` as default values, also used here.

A single past-trajectory buffer `(u_ini, y_ini)` of length `T_ini` is shared
across libraries — the robot's actual history is independent of which library
predicts the future.
"""

from __future__ import annotations

from typing import Optional

import cvxpy as cp
import numpy as np

# A library is a tuple of block-Hankel matrices (Up, Uf, Yp, Yf), e.g. as
# produced by `two_wheel_robot.controllers.hankel.build_hankel`.
Library = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class DeePC:
    """Single parametric DeePC with orientation-keyed swappable libraries.

    See the module docstring. Holds `N_libraries >= 1` data libraries; selects
    one per `act()` by heading (closest anchor) and feeds its Hankels into one
    cached QP. With a single library, selection is trivial (always index 0) and
    `anchor_headings` / `heading_index` are not consulted — this is the
    plain single-library DeePC case.
    """

    def __init__(
        self,
        libraries: list[Library],
        anchor_headings,
        Q: np.ndarray,
        R: np.ndarray,
        T_ini: int = 5,
        N: int = 12,
        lambda_g: float = 2.0,
        lambda_y: float = 3e6,
        u_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
        heading_index: int = 2,
        solver: Optional[str] = None,
    ):
        if T_ini < 1 or N < 1:
            raise ValueError(f"T_ini, N must be >= 1; got {T_ini}, {N}")
        if not libraries:
            raise ValueError("need at least one library")

        # Derive (m_u, p_y, n_cols) from the first library, then require every
        # library to agree — they share one set of Hankel parameters.
        Up0 = np.asarray(libraries[0][0])
        if Up0.shape[0] % T_ini:
            raise ValueError(
                f"Up rows ({Up0.shape[0]}) must be divisible by T_ini ({T_ini})"
            )
        Yp0 = np.asarray(libraries[0][2])
        if Yp0.shape[0] % T_ini:
            raise ValueError(
                f"Yp rows ({Yp0.shape[0]}) must be divisible by T_ini ({T_ini})"
            )
        m_u = Up0.shape[0] // T_ini
        p_y = Yp0.shape[0] // T_ini
        n_cols = Up0.shape[1]

        if Q.shape != (p_y, p_y):
            raise ValueError(f"Q shape {Q.shape}; expected ({p_y}, {p_y})")
        if R.shape != (m_u, m_u):
            raise ValueError(f"R shape {R.shape}; expected ({m_u}, {m_u})")

        stored: list[Library] = []
        for i, lib in enumerate(libraries):
            Up, Uf, Yp, Yf = (np.asarray(a, dtype=np.float64) for a in lib)
            if Up.shape != (T_ini * m_u, n_cols):
                raise ValueError(
                    f"library {i}: Up shape {Up.shape}; "
                    f"expected ({T_ini * m_u}, {n_cols})"
                )
            if Uf.shape != (N * m_u, n_cols):
                raise ValueError(
                    f"library {i}: Uf shape {Uf.shape}; expected ({N * m_u}, {n_cols})"
                )
            if Yp.shape != (T_ini * p_y, n_cols):
                raise ValueError(
                    f"library {i}: Yp shape {Yp.shape}; "
                    f"expected ({T_ini * p_y}, {n_cols})"
                )
            if Yf.shape != (N * p_y, n_cols):
                raise ValueError(
                    f"library {i}: Yf shape {Yf.shape}; expected ({N * p_y}, {n_cols})"
                )
            stored.append((Up, Uf, Yp, Yf))

        anchor_headings = np.asarray(anchor_headings, dtype=np.float64)
        if anchor_headings.shape != (len(stored),):
            raise ValueError(
                f"anchor_headings shape {anchor_headings.shape}; "
                f"expected ({len(stored)},)"
            )
        # heading_index is only consulted when switching between libraries.
        if len(stored) > 1 and not 0 <= heading_index < p_y:
            raise ValueError(
                f"heading_index {heading_index} out of range for p_y={p_y}"
            )

        self._libraries = stored
        self._n_lib = len(stored)
        self.anchor_headings = anchor_headings
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.T_ini = int(T_ini)
        self.N = int(N)
        self.m_u = int(m_u)
        self.p_y = int(p_y)
        self.n_cols = int(n_cols)
        self.lambda_g = float(lambda_g)
        self.lambda_y = float(lambda_y)
        self.u_bounds = u_bounds
        self.heading_index = int(heading_index)
        self.solver = solver

        # Single shared past trajectory buffer — primed by reset().
        self._u_buf: Optional[np.ndarray] = None  # shape (T_ini, m_u)
        self._y_buf: Optional[np.ndarray] = None  # shape (T_ini, p_y)

        # Diagnostics.
        self.last_library_idx: int = -1
        self.last_warm_started: bool = False
        self._prev_idx: int = -1

        self._build_problem()

    # ----- QP construction ----------------------------------------------------

    @staticmethod
    def _psd_sqrt(M: np.ndarray) -> np.ndarray:
        """Symmetric matrix square root of a PSD matrix `M`."""
        M = 0.5 * (M + M.T)
        w, V = np.linalg.eigh(M)
        w = np.clip(w, 0.0, None)
        return V @ np.diag(np.sqrt(w)) @ V.T

    def _build_problem(self) -> None:
        g = cp.Variable(self.n_cols)
        sigma_y = cp.Variable(self.T_ini * self.p_y)

        # Active-library Hankels live in parameters, swapped per step.
        Up = cp.Parameter((self.T_ini * self.m_u, self.n_cols))
        Uf = cp.Parameter((self.N * self.m_u, self.n_cols))
        Yp = cp.Parameter((self.T_ini * self.p_y, self.n_cols))
        Yf = cp.Parameter((self.N * self.p_y, self.n_cols))

        u_ini_param = cp.Parameter(self.T_ini * self.m_u)
        y_ini_param = cp.Parameter(self.T_ini * self.p_y)
        y_ref_param = cp.Parameter(self.N * self.p_y)

        # ‖x‖²_Q as ‖Q^{1/2} x‖² keeps the cost DPP-compliant.
        Q_sqrt_bar = np.kron(np.eye(self.N), self._psd_sqrt(self.Q))
        R_sqrt_bar = np.kron(np.eye(self.N), self._psd_sqrt(self.R))

        u_future = Uf @ g
        y_future = Yf @ g
        y_err = y_future - y_ref_param

        objective = cp.Minimize(
            cp.sum_squares(Q_sqrt_bar @ y_err)
            + cp.sum_squares(R_sqrt_bar @ u_future)
            + self.lambda_g * cp.norm(g, 1)
            + self.lambda_y * cp.sum_squares(sigma_y)
        )

        constraints: list[cp.Constraint] = [
            Up @ g == u_ini_param,
            Yp @ g + sigma_y == y_ini_param,
        ]
        if self.u_bounds is not None:
            u_min, u_max = self.u_bounds
            u_min = np.asarray(u_min, dtype=np.float64).reshape(self.m_u)
            u_max = np.asarray(u_max, dtype=np.float64).reshape(self.m_u)
            constraints.append(u_future >= np.tile(u_min, self.N))
            constraints.append(u_future <= np.tile(u_max, self.N))

        self._g = g
        self._sigma_y = sigma_y
        self._Up_param = Up
        self._Uf_param = Uf
        self._Yp_param = Yp
        self._Yf_param = Yf
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
        """Prime the shared past buffer with `T_ini` copies of `(u_initial, y_initial)`.

        A "synthetic past": a constant trajectory where the robot held
        `y_initial` while applying `u_initial` (zeros by default). The QP's soft
        past-output constraint absorbs the resulting inconsistency.
        """
        y_initial = np.asarray(y_initial, dtype=np.float64).reshape(self.p_y)
        if u_initial is None:
            u_initial = np.zeros(self.m_u, dtype=np.float64)
        else:
            u_initial = np.asarray(u_initial, dtype=np.float64).reshape(self.m_u)
        self._u_buf = np.tile(u_initial, (self.T_ini, 1))
        self._y_buf = np.tile(y_initial, (self.T_ini, 1))
        # A fresh episode invalidates any prior warm-start.
        self._prev_idx = -1
        self._g.value = None

    def _select_index(self, heading: float) -> int:
        """Index of the library whose anchor is closest to `heading` on the circle."""
        # Wrap-aware shortest signed angular difference; pick smallest |diff|.
        # With one library this is trivially 0 regardless of `heading`.
        diffs = (heading - self.anchor_headings + np.pi) % (2 * np.pi) - np.pi
        return int(np.argmin(np.abs(diffs)))

    def act(self, y_current: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        """Solve the QP for the current step and return `u_t` (shape `(m_u,)`).

        Args:
            y_current: current observation, shape `(p_y,)`. Used to select the
                library (by heading) and to update the past-output buffer after
                the solve.
            y_ref: reference output. Either shape `(p_y,)` (broadcast across the
                `N`-step horizon) or shape `(N, p_y)` (per-step reference).

        Returns:
            `u_t`: applied control input, shape `(m_u,)`.

        Raises:
            RuntimeError: if `reset()` hasn't been called, or the QP fails.
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

        # Select the active library by heading (trivially 0 for a single library).
        idx = 0 if self._n_lib == 1 else self._select_index(
            float(y_current[self.heading_index])
        )

        # A warm-start `g` indexes the *previous* library's columns; clear it on
        # a switch so the solve doesn't start from a meaningless point.
        warm = idx == self._prev_idx
        self.last_warm_started = warm
        if not warm:
            self._g.value = None

        Up, Uf, Yp, Yf = self._libraries[idx]
        self._Up_param.value = Up
        self._Uf_param.value = Uf
        self._Yp_param.value = Yp
        self._Yf_param.value = Yf
        self._u_ini_param.value = self._u_buf.flatten()
        self._y_ini_param.value = self._y_buf.flatten()
        self._y_ref_param.value = y_ref_flat

        self._problem.solve(solver=self.solver, warm_start=True)
        status = self._problem.status
        if status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"DeePC QP failed: status={status}")

        g_val = self._g.value
        assert g_val is not None
        u_future = Uf @ g_val
        u_t = u_future[: self.m_u].copy()
        # Solver tolerance can produce sub-microscopic bound violations; clip to
        # guarantee the returned action lies inside u_bounds.
        if self.u_bounds is not None:
            u_t = np.clip(u_t, self.u_bounds[0], self.u_bounds[1])

        # Slide the shared buffer: drop oldest, append (u_t, y_current).
        self._u_buf = np.roll(self._u_buf, -1, axis=0)
        self._u_buf[-1] = u_t
        self._y_buf = np.roll(self._y_buf, -1, axis=0)
        self._y_buf[-1] = y_current

        self._prev_idx = idx
        self.last_library_idx = idx
        return u_t

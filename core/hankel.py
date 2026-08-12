"""Block-Hankel matrix construction for DeePC.

Given an offline trajectory `(u_t, y_t)_{t=0..T-1}` and horizons `T_ini` (past
window) and `N` (future / prediction window), we build the *past* and *future*
block-Hankel matrices the DeePC QP consumes.

For an input series of width `d` (i.e. data shape `(T, d)`), the full Hankel
matrix `H` has `L · d` rows and `T − L + 1` columns, with `L = T_ini + N`:

    H[k·d : (k+1)·d, j] = data[j + k]

Past / future split:

    Up = Hu[: T_ini·m_u, :]            Yp = Hy[: T_ini·p_y, :]
    Uf = Hu[T_ini·m_u :, :]            Yf = Hy[T_ini·p_y :, :]

No disturbance `e` here (the two-wheel goal-reaching task has none); Deep-LCC
in mixed-traffic settings carries an analogous `Ep`/`Ef` pair that we drop.
"""

from __future__ import annotations

import numpy as np


def _block_hankel(data: np.ndarray, L: int) -> np.ndarray:
    """Block-Hankel of `data` of shape `(T, d)` → array of shape `(L·d, T − L + 1)`."""
    T, d = data.shape
    n_cols = T - L + 1
    out = np.empty((L * d, n_cols), dtype=data.dtype)
    for k in range(L):
        out[k * d : (k + 1) * d, :] = data[k : k + n_cols].T
    return out


def build_hankel(
    u: np.ndarray,
    y: np.ndarray,
    T_ini: int = 5,
    N: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build `(Up, Uf, Yp, Yf)` from a single `(u, y)` trajectory.

    Args:
        u: shape `(T, m_u)`. Control inputs.
        y: shape `(T, p_y)`. Outputs.
        T_ini: past window length. Default 5 (paper).
        N: future window length. Default 12 (paper's largest W in Figure 2).

    Returns:
        Up: `(T_ini · m_u, T − L + 1)`
        Uf: `(N · m_u, T − L + 1)`
        Yp: `(T_ini · p_y, T − L + 1)`
        Yf: `(N · p_y, T − L + 1)`
        where `L = T_ini + N`.

    Raises:
        ValueError: if `u.shape[0] != y.shape[0]`, if `T_ini < 1` or `N < 1`,
            or if the trajectory is shorter than `L`.
    """
    if u.ndim != 2 or y.ndim != 2:
        raise ValueError(f"u and y must be 2-D, got shapes {u.shape}, {y.shape}")
    if u.shape[0] != y.shape[0]:
        raise ValueError(
            f"u and y must have the same length T, got {u.shape[0]} vs {y.shape[0]}"
        )
    if T_ini < 1 or N < 1:
        raise ValueError(f"T_ini and N must be >= 1, got T_ini={T_ini}, N={N}")
    T = u.shape[0]
    L = T_ini + N
    if T < L:
        raise ValueError(
            f"trajectory length T={T} must be >= T_ini + N = {L}"
        )

    m_u = u.shape[1]
    p_y = y.shape[1]

    Hu = _block_hankel(u, L)
    Hy = _block_hankel(y, L)

    Up = Hu[: T_ini * m_u, :]
    Uf = Hu[T_ini * m_u :, :]
    Yp = Hy[: T_ini * p_y, :]
    Yf = Hy[T_ini * p_y :, :]

    return Up, Uf, Yp, Yf

"""The bank's `origin`/`t0` must actually locate a column in its source trajectory.

`scripts/measure_selection_distance.py` reads the configuration a selected column
was collected at via `q_{origin[j]}[t0[j]]`. If that mapping is off by a stride or
a trajectory, every distance in the coverage argument is quietly wrong and nothing
raises -- so pin it against the column contents themselves.
"""
from __future__ import annotations

import numpy as np

from core.selectdpc import trajectory_bank


def test_origin_t0_reconstructs_the_column():
    rng = np.random.default_rng(0)
    T_ini, N, stride = 3, 4, 2
    u_list = [rng.standard_normal((40, 2)), rng.standard_normal((25, 2))]
    y_list = [rng.standard_normal((40, 3)), rng.standard_normal((25, 3))]
    bank = trajectory_bank(u_list, y_list, T_ini, N, stride=stride)

    n = bank["Up"].shape[1]
    assert bank["origin"].shape == (n,) and bank["t0"].shape == (n,)
    assert set(np.unique(bank["origin"])) == {0, 1}

    for j in range(n):
        i, t = int(bank["origin"][j]), int(bank["t0"][j])
        u, y = u_list[i], y_list[i]
        assert np.allclose(bank["Up"][:, j], u[t:t + T_ini].ravel())
        assert np.allclose(bank["Uf"][:, j], u[t + T_ini:t + T_ini + N].ravel())
        assert np.allclose(bank["Yp"][:, j], y[t:t + T_ini].ravel())
        assert np.allclose(bank["Yf"][:, j], y[t + T_ini:t + T_ini + N].ravel())

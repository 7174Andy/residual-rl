import numpy as np

from scripts.measure_quadruped_lag import lag_63


def test_lag_63_first_order():
    dt, tau, idx = 0.02, 0.5, 100
    t = np.arange(400) * dt
    v = np.where(np.arange(400) < idx, 0.0, 1.0 - np.exp(-(t - idx * dt) / tau))
    lag, base, final = lag_63(v, idx, dt)
    assert abs(lag - tau) <= 2 * dt and base == 0.0 and final > 0.99

# tests/test_clone_eval.py
"""Fidelity-gate building blocks: Wilson CI, McNemar, regression-by-regime."""
from __future__ import annotations

import numpy as np

from rl.stats import mcnemar_pvalue, wilson_ci
from two_wheel_robot.rl.clone_eval import regression_by_regime, trajectory_deviation


def test_wilson_ci_bounds_are_sane():
    lo, hi = wilson_ci(30, 78)
    assert 0.0 <= lo < 0.385 < hi <= 1.0  # 30/78 ~ 0.385 inside the interval
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_mcnemar_detects_strong_disagreement():
    # 10 seeds where DeePC reaches but clone fails, 0 the other way -> significant.
    assert mcnemar_pvalue(b=10, c=0) < 0.05
    # Balanced disagreement -> not significant.
    assert mcnemar_pvalue(b=5, c=5) > 0.5
    assert mcnemar_pvalue(b=0, c=0) == 1.0


def test_trajectory_deviation_wraps_heading():
    # Identical positions; headings at +0.9π and -0.9π are 0.2π apart on the
    # circle, NOT 1.8π — the deviation must use the wrapped difference.
    a = np.array([[0.0, 0.0, 0.9 * np.pi]])
    b = np.array([[0.0, 0.0, -0.9 * np.pi]])
    dev = trajectory_deviation(a, b)
    assert np.isclose(dev["pos_median"], 0.0)
    assert np.isclose(dev["head_median"], 0.2 * np.pi)


def test_regression_by_regime_separates_groups():
    pred = np.array([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    true = np.array([[1.0, 0.0], [1.5, 0.0], [2.0, 0.0], [2.0, 0.0]])
    regime = np.array(["good", "good", "degenerate", "degenerate"])
    rep = regression_by_regime(pred, true, regime)
    assert set(rep.keys()) == {"good", "degenerate"}
    assert np.isclose(rep["degenerate"]["mae_v"], 0.0)
    assert rep["good"]["mae_v"] > 0.0

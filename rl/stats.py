"""Paired statistics used by every evaluation in the repo.

`wilson_ci` rather than a normal approximation because reach rates land at the
boundaries (0/10, 96/120) where the normal interval is wrong. `mcnemar_pvalue`
is the paired binary test: it counts rescues against regressions on IDENTICAL
scenarios, which is the comparison a reach-rate difference cannot make.
"""
from __future__ import annotations

import math


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_pvalue(b: int, c: int) -> float:
    """McNemar p-value, continuity-corrected, via the exact chi2(1) survival.

    `b` = #(DeePC reach, clone fail), `c` = #(DeePC fail, clone reach). The
    statistic is `chi2(1)`-distributed, whose survival function is
    `erfc(sqrt(stat/2))`.
    """
    n = b + c
    if n == 0:
        return 1.0
    stat = max(0.0, abs(b - c) - 1.0) ** 2 / n
    # stat == 0 (|b - c| <= 1) -> erfc(0) == 1.0; the guard just makes that explicit.
    if stat <= 0.0:
        return 1.0
    return math.erfc(math.sqrt(stat / 2.0))

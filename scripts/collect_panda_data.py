"""Collect the PandaReach-v0 DeePC data libraries and gate on coverage.

    uv run python scripts/collect_panda_data.py

Fails loudly rather than writing a bad library file. Two gates:
  * clip fraction < 1%  -- above that, the recorded u differs from what the plant
    received often enough to corrupt the Hankel
  * rank >= 133 per library -- m_u(T_ini+N) + n_state
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from panda import data_collection as dc
from panda.env import PandaReachEnv

CLIP_GATE = 0.01


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=dc.LIBRARIES_PATH)
    ap.add_argument("--T", type=int, default=dc.DEFAULT_T)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = PandaReachEnv(max_steps=10**9)
    try:
        payload = dc.collect_libraries(env, T=args.T,
                                       rng=np.random.default_rng(args.seed))
        rep = dc.coverage_report(payload)
    finally:
        env.close()

    print(f"anchors q1        {list(dc.PANDA_ANCHOR_Q1)}")
    print(f"anchor azimuths   {[round(np.degrees(a), 1) for a in rep['anchor_azimuths']]} deg")
    print(f"excitation        theta={dc.OU_THETA} sigma={dc.OU_SIGMA_FRAC}*dmax k_ret={dc.K_RET}")
    print(f"clip fraction     {100 * rep['clip_frac']:.2f}%   (gate < {100 * CLIP_GATE:.0f}%)")
    print(f"tip radius        {rep['tip_radius_min']:.3f} .. {rep['tip_radius_max']:.3f} m")
    for lib in rep["libraries"]:
        print(f"  library {lib['index']}  n_cols {lib['n_cols']}  rank {lib['rank']}"
              f"  (floor {dc.RANK_FLOOR})  s133/s1 {lib['s_ratio_133']:.2e}")

    failures = []
    if rep["clip_frac"] >= CLIP_GATE:
        failures.append(f"clip fraction {rep['clip_frac']:.3f} >= {CLIP_GATE}")
    for lib in rep["libraries"]:
        if lib["rank"] < dc.RANK_FLOOR:
            failures.append(f"library {lib['index']} rank {lib['rank']} < {dc.RANK_FLOOR}")
    if failures:
        print("\nGATE FAILED:", "; ".join(failures), file=sys.stderr)
        print("Not writing the library file. Raise T, or lower k_ret, and re-measure.",
              file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **payload)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

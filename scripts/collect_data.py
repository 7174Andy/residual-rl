"""Collect DeePC data libraries.

Layout follows arXiv:2603.07395 Appendix D: 4 trajectories x 1500 steps starting
from headings {pi/4, 3pi/4, 5pi/4, 7pi/4} at the origin. The default PE bounds
are broader than the paper's (v in [0, 20], w in [-pi/2, pi/2] vs v in [10, 20],
w in [-pi/6, pi/6]) so the data includes stopping and pivoting, which
goal-reaching needs. To reproduce the paper's bounds exactly:

    uv run python scripts/collect_data.py --v_min 10 --w_abs_max 0.5236 \\
        --out data/libraries_paper.npz

The chosen bounds are saved inside the .npz under key `sample_bounds`, so
`scripts/run_deepc.py` reconstructs the env with matching action bounds
automatically.
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import numpy as np

import two_wheel_robot.env  # noqa: F401  registers Gym ID
from two_wheel_robot.controllers.data_collection import (
    PAPER_T,
    collect_libraries,
    paper_init_states,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/libraries.npz")
    parser.add_argument("--T", type=int, default=PAPER_T)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--v_min", type=float, default=0.0, help="min tangential vel (paper: 10)")
    parser.add_argument("--v_max", type=float, default=20.0, help="max tangential vel (paper: 20)")
    parser.add_argument(
        "--w_abs_max",
        type=float,
        default=np.pi / 2,
        help="|w| upper bound; w sampled in [-w_abs_max, w_abs_max] (paper: pi/6)",
    )
    args = parser.parse_args()

    sample_bounds = np.array(
        [[args.v_min, args.v_max], [-args.w_abs_max, args.w_abs_max]],
        dtype=np.float64,
    )
    print(
        f"sample bounds: v in [{args.v_min:.3f}, {args.v_max:.3f}], "
        f"w in [{-args.w_abs_max:.3f}, {args.w_abs_max:.3f}]"
    )

    # Construct env with matching action_bounds so PE samples aren't clipped
    # silently — every action we sample is exactly what the dynamics see.
    env = gym.make("TwoWheelGoal-v0", action_bounds=sample_bounds)
    try:
        rng = np.random.default_rng(args.seed)
        libraries = collect_libraries(
            env,
            T=args.T,
            init_states=paper_init_states(),
            rng=rng,
            sample_bounds=sample_bounds,
        )
    finally:
        env.close()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload: dict[str, np.ndarray] = {"sample_bounds": sample_bounds}
    for i, (u, y) in enumerate(libraries):
        payload[f"u_{i}"] = u
        payload[f"y_{i}"] = y
    np.savez(args.out, **payload)  # type: ignore[arg-type]

    print(f"Saved {len(libraries)} libraries to {args.out} (T={args.T}, seed={args.seed})")
    for i, (u, y) in enumerate(libraries):
        print(f"  lib {i}: u {u.shape}, y {y.shape}")


if __name__ == "__main__":
    main()

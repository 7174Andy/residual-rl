"""Record closed-loop DeePC on PandaReach-v0: MP4 + per-scenario cost breakdown.

    # the frozen showcase scenarios at the swept-winning lambdas
    uv run python scripts/record_panda_deepc_video.py \
        --lambda-g 5e-2 --lambda-y 7.5e4

    # specific scenarios, in the order given
    uv run python scripts/record_panda_deepc_video.py --scenario-ids 4 3 1

Episodes come from the frozen scenario set, so the footage is of the same
episodes `scripts/run_panda_deepc.py --mode eval` scores, and the rollout runs
through `eval.run_scenarios` itself -- the video cannot show a trajectory the
results CSV never measured. Nothing is appended to `data/panda_results.csv`:
this is a rendering entrypoint, not an extra measurement.

fps equals the 50 Hz control rate, so playback is real time.
"""
from __future__ import annotations

import argparse

import numpy as np

from core.video_encoding import encode_video
from panda import deepc_setup as ds
from panda import eval as pe
from panda import scenarios as sc
from panda.env import PandaReachEnv


def cost_breakdown(row: dict, r_scalar: float, reach_bonus: float) -> tuple[float, float]:
    """Split a result row's return into `(tracking, control)` cost.

    Inverts the stage cost the env applies,
    `ret = -tracking - control + reach_bonus * reached`, using `effort`
    (`sum u^T u`) for the control half. Reported because the two terms are what
    say whether `R` has any grip on the solution at all -- reach rate alone hides
    a controller that reaches by flailing.
    """
    control = r_scalar * row["effort"]
    tracking = -row["ret"] - control + reach_bonus * bool(row["reached"])
    return tracking, control


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario-ids", type=int, nargs="+", default=list(sc.SHOWCASE_IDS),
                    help=f"frozen scenario ids, in playback order "
                         f"(default: the showcase set {tuple(sc.SHOWCASE_IDS)})")
    ap.add_argument("--lambda-g", type=float, default=ds.LAMBDA_G_DEFAULT)
    ap.add_argument("--lambda-y", type=float, default=ds.LAMBDA_Y_DEFAULT)
    ap.add_argument("--out", default="data/panda_deepc.mp4")
    args = ap.parse_args()

    scen = sc.load()
    print(f"scenarios {sc.SCENARIOS_PATH}  checksum {sc.checksum(scen)}")
    print(f"lambda_g={args.lambda_g:.1e}  lambda_y={args.lambda_y:.1e}  "
          f"ids {args.scenario_ids}\n")

    deepc, info = ds.build_canonical_panda_deepc(
        lambda_g=args.lambda_g, lambda_y=args.lambda_y
    )
    env = PandaReachEnv(render_mode="rgb_array")
    frames: list[np.ndarray] = []
    try:
        rows = pe.run_scenarios(
            env, ds.DeePCPolicy(deepc, info), args.scenario_ids, scen,
            method="deepc_video", frames=frames,
        )
        fps = int(env.metadata["render_fps"])
        r_scalar, reach_bonus = float(env.R[0, 0]), env.reach_bonus
    finally:
        env.close()

    print(f"{'sid':>4} {'reached':>8} {'steps':>6} {'final_m':>8} {'track':>9} "
          f"{'ctrl':>8} {'switches':>9} {'ms/step':>8}")
    tracks, ctrls = [], []
    for r in rows:
        track, ctrl = cost_breakdown(r, r_scalar, reach_bonus)
        tracks.append(track)
        ctrls.append(ctrl)
        print(f"{r['scenario_id']:4d} {str(r['reached']):>8} {r['steps']:6d} "
              f"{r['final_dist']:8.3f} {track:9.2f} {ctrl:8.3f} "
              f"{r['n_switches']:9d} {r['mean_solve_ms']:8.1f}")
    total_track, total_ctrl = sum(tracks), sum(ctrls)
    ratio = f"{total_track / total_ctrl:.0f}x" if total_ctrl else "n/a"
    print(f"\n  reached {sum(r['reached'] for r in rows)}/{len(rows)}   "
          f"tracking {total_track:.2f}   control {total_ctrl:.3f}   ratio {ratio}")
    print("  -- tracking dominating control by orders of magnitude means R is "
          "not shaping the solution; it is a reach-at-any-cost regime.")

    if encode_video(frames, args.out, fps=fps):
        print(f"\nwrote {args.out}  ({len(frames)} frames, "
              f"{len(frames) / fps:.1f}s at {fps} fps)")


if __name__ == "__main__":
    main()

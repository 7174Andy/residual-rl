"""Closed-loop DeePC on PandaReach-v0: lambda sweep, then the full evaluation.

    # phase 3a -- 3x3 grid over the SWEEP subset (~56 min)
    uv run python scripts/run_panda_deepc.py --mode sweep

    # phase 3b -- the winner over all 78 frozen scenarios (~24 min)
    uv run python scripts/run_panda_deepc.py --mode eval --lambda-g 5e-3 --lambda-y 7.5e3

Go/no-go: reach rate >= 20% on the 78 frozen scenarios.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from panda import data_collection as dc
from panda import deepc_setup as ds
from panda import eval as pe
from panda import scenarios as sc
from panda.env import PandaReachEnv

LAMBDA_G_GRID = (5e-4, 5e-3, 5e-2)
LAMBDA_Y_GRID = (7.5e2, 7.5e3, 7.5e4)


def run_one(lambda_g, lambda_y, ids, scen, method, trace_ids=(),
            libraries_path=dc.LIBRARIES_PATH, output="tip"):
    deepc, info = ds.build_canonical_panda_deepc(
        libraries_path=libraries_path, lambda_g=lambda_g, lambda_y=lambda_y,
        output=output,
    )
    env = PandaReachEnv()
    try:
        rows = pe.run_scenarios(env, ds.DeePCPolicy(deepc, info), ids, scen,
                                method=method, trace_ids=trace_ids)
    finally:
        env.close()
    return rows


def summarize(rows):
    reach = float(np.mean([r["reached"] for r in rows]))
    steps = [r["steps"] for r in rows if r["reached"]]
    return {
        "n": len(rows),
        "reach": reach,
        "mean_steps": float(np.mean(steps)) if steps else float("nan"),
        "mean_final_dist": float(np.mean([r["final_dist"] for r in rows])),
        "qp_failures": sum(r["qp_failures"] for r in rows),
        "mean_solve_ms": float(np.mean([r["mean_solve_ms"] for r in rows])),
        "mean_switches": float(np.mean([r["n_switches"] for r in rows])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("sweep", "eval"), required=True)
    ap.add_argument("--lambda-g", type=float, default=ds.LAMBDA_G_DEFAULT)
    ap.add_argument("--lambda-y", type=float, default=ds.LAMBDA_Y_DEFAULT)
    ap.add_argument("--libraries", default=dc.LIBRARIES_PATH)
    ap.add_argument("--output", choices=("tip", "ext"), default="tip",
                    help="DeePC output map: 'tip' (3-D, the recorded baseline) or "
                         "'ext' (10-D tip+normalized q; needs a v1+ libraries file)")
    ap.add_argument("--no-record", action="store_true",
                    help="do not append to data/panda_results.csv")
    args = ap.parse_args()

    scen = sc.load()
    print(f"scenarios {sc.SCENARIOS_PATH}  checksum {sc.checksum(scen)}")

    if args.mode == "sweep":
        ids = list(sc.SWEEP_IDS)
        print(f"\nphase 3a: 3x3 lambda grid over {len(ids)} scenarios, "
              f"output={args.output!r}\n")
        print(f"{'lambda_g':>10} {'lambda_y':>10} {'reach':>7} {'steps':>7} {'ms/step':>8}")
        best = None
        for lg, ly in itertools.product(LAMBDA_G_GRID, LAMBDA_Y_GRID):
            s = summarize(run_one(
                lg, ly, ids, scen,
                method=f"deepc_sweep_{args.output}_g{lg}_y{ly}",
                libraries_path=args.libraries, output=args.output,
            ))
            print(f"{lg:10.1e} {ly:10.1e} {s['reach']:7.2f} {s['mean_steps']:7.1f} "
                  f"{s['mean_solve_ms']:8.1f}")
            if best is None or s["reach"] > best[0]["reach"]:
                best = (s, lg, ly)
        s, lg, ly = best
        print(f"\nbest: lambda_g={lg:.1e} lambda_y={ly:.1e}  reach {s['reach']:.2f}")
        print(f"confirm with: --mode eval --lambda-g {lg} --lambda-y {ly} "
              f"--output {args.output} --libraries {args.libraries}")
        # The lambda grid was chosen against the 3-D tip output. Re-using it for a
        # 10-D output is not a like-for-like sweep: |g|_1 rises with the extra
        # equality rows, so lambda_g*|g|_1 carries more weight against an unchanged
        # tracking term and the grid may sit entirely on the wrong side of the knee.
        if args.output != "tip":
            print(f"\n  NOTE: LAMBDA_G_GRID/LAMBDA_Y_GRID were tuned for output="
                  f"'tip' (p_y=3). At p_y>3 treat this grid as a first pass, and "
                  f"check whether the best point sits on a grid edge.")
        return

    ids = list(sc.EVAL_IDS)
    print(f"\nphase 3b: {len(ids)} scenarios at "
          f"lambda_g={args.lambda_g:.1e} lambda_y={args.lambda_y:.1e} "
          f"output={args.output!r}\n")
    # "deepc" stays the tip-only method name so the rows already in the shared CSV
    # keep their meaning; the ext arm records under a distinct method rather than
    # silently mixing two different output maps under one key.
    method = "deepc" if args.output == "tip" else "deepc_ext"
    rows = run_one(args.lambda_g, args.lambda_y, ids, scen,
                   method=method, trace_ids=sc.SHOWCASE_IDS,
                   libraries_path=args.libraries, output=args.output)
    if not args.no_record:
        pe.append_results(rows)
        print(f"appended {len(rows)} rows to {pe.RESULTS_PATH}")
    s = summarize(rows)
    for k, v in s.items():
        print(f"  {k:16s} {v}")
    # Cross-check: the sweep subset inside the full run must match phase 3a.
    sub = summarize([r for r in rows if r["scenario_id"] in set(sc.SWEEP_IDS)])
    print(f"\n  reach on the SWEEP subset (ids 0-19) within this run: {sub['reach']:.2f}")
    print("  -- compare to phase 3a's number for the same lambda; a material "
          "difference means something is non-deterministic and 3a's choice is void.")
    verdict = "GO" if s["reach"] >= 0.20 else "NO-GO"
    print(f"\n  gate: reach {s['reach']:.2f} vs 0.20  ->  {verdict}")


if __name__ == "__main__":
    main()

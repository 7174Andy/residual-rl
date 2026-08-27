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
import os
import time

import numpy as np

from panda import data_collection as dc
from panda import deepc_setup as ds
from panda import eval as pe
from panda import scenarios as sc
from panda.env import PandaReachEnv

LAMBDA_G_GRID = (5e-4, 5e-3, 5e-2)
LAMBDA_Y_GRID = (7.5e2, 7.5e3, 7.5e4)


def run_one(lambda_g, lambda_y, ids, scen, method, trace_ids=(),
            libraries_path=dc.LIBRARIES_PATH, output="tip", progress=False,
            on_row=None):
    """Roll one (lambda, output, library) configuration over `ids`.

    `progress` drives the scenarios one at a time so a long run reports as it
    goes -- `eval.run_scenarios` is silent, and at T=3000 a 78-scenario arm takes
    hours, where silence is indistinguishable from a hang. Numerically identical
    to one batched call: the only per-call work is the idempotent
    `validate_against_env`, each scenario does its own `reset_to`, and the
    DeePC past-buffer warm-start carries through the shared `policy` either way.

    `on_row(row)` is called as each scenario finishes, so a run that is
    interrupted keeps what it already measured. A 78-scenario arm at T=3000 is
    hours long; recording only at the end means a stop at scenario 77 writes
    nothing, which is exactly how one ext arm was already lost.
    """
    deepc, info = ds.build_canonical_panda_deepc(
        libraries_path=libraries_path, lambda_g=lambda_g, lambda_y=lambda_y,
        output=output,
    )
    env = PandaReachEnv()
    policy = ds.DeePCPolicy(deepc, info)
    try:
        if not progress:
            return pe.run_scenarios(env, policy, ids, scen,
                                    method=method, trace_ids=trace_ids)
        rows: list[dict] = []
        t0 = time.time()
        for n, sid in enumerate(ids, 1):
            fresh = pe.run_scenarios(env, policy, [sid], scen,
                                     method=method, trace_ids=trace_ids)
            rows += fresh
            if on_row is not None:
                for r in fresh:
                    on_row(r)
            hits = sum(r["reached"] for r in rows)
            print(f"  [{output}] {n:3d}/{len(ids)} sid={sid:<3d} "
                  f"reached={hits}/{n}  steps={rows[-1]['steps']:3d}  "
                  f"{time.time() - t0:6.0f}s", flush=True)
    finally:
        env.close()
    return rows


def method_name(output: str, libraries_path: str) -> str:
    """CSV method key, tagged with the library file's version suffix.

    `data/panda_results.csv` is keyed by `(method, scenario_id)` and
    `eval.append_results` appends blindly, so an untagged name silently mixes
    libraries: a T=3000 run recorded as plain "deepc" would sit alongside the
    T=400 rows under one key and every paired test downstream would be comparing
    a mixture to itself. Derived from the path rather than taken as a flag,
    because a flag is a thing you forget exactly once.

    The 78 legacy rows under the bare key "deepc" predate this and are the v0
    (T=400, tip) run; re-running v0 now records as "deepc_v0" instead.
    """
    version = os.path.basename(libraries_path).removesuffix(".npz").rsplit("_", 1)[-1]
    return f"{'deepc' if output == 'tip' else 'deepc_ext'}_{version}"


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
    ap.add_argument("--method", default=None,
                    help="override the CSV method label for --mode eval "
                         "(default: None -> derive from --output/--libraries "
                         "via method_name(), the current behavior)")
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
    # Distinct method per (output map, library version) -- see `method_name`.
    method = args.method if args.method is not None else method_name(args.output, args.libraries)
    print(f"recording as method {method!r}")
    # Resume: `append_results` does not dedupe, so re-running an interrupted arm
    # would leave two rows per scenario under one (method, scenario_id) key and
    # quietly corrupt every paired test that reads it. Skipping what is already
    # recorded makes an interrupted arm resumable and duplicate-free. To
    # re-measure a scenario, delete its row first.
    done: dict[int, dict] = {}
    if not args.no_record and os.path.exists(pe.RESULTS_PATH):
        done = {r["scenario_id"]: r for r in pe.read_results()
                if r["method"] == method}
        if done:
            ids = [i for i in ids if i not in done]
            print(f"resuming: {len(done)} scenarios already recorded under "
                  f"{method!r}, {len(ids)} to go")

    on_row = None if args.no_record else (lambda r: pe.append_results([r]))
    fresh = run_one(args.lambda_g, args.lambda_y, ids, scen,
                    method=method, trace_ids=sc.SHOWCASE_IDS,
                    libraries_path=args.libraries, output=args.output,
                    progress=True, on_row=on_row)
    if not args.no_record:
        print(f"recorded {len(fresh)} rows to {pe.RESULTS_PATH} (incrementally)")
    # Summarize over the whole arm, not just this session's slice.
    rows = list(done.values()) + fresh
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

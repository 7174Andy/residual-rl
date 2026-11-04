"""Score a DeePC config on the frozen quadruped scenario split; resumable, parallel.

    .venv/bin/python scripts/run_quadruped_deepc.py --split tune --out runs/a.csv --prune
Rows are appended to --out as episodes finish; re-running skips ids already there.
The resolved config is kept in `<out>.cfg.json`; resuming with a different config exits.
`--split eval` is capped at 3 looks (data/quadruped_deepc/eval_looks.csv).
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time

from quadruped import scenarios as sc
from quadruped.deepc_setup import (
    DEFAULT_LIBRARIES,
    build_canonical_quadruped_deepc,
    run_episode,
)
from quadruped.env import QuadrupedGoalEnv
from rl.stats import wilson_ci

EVAL_LOOKS_PATH = "data/quadruped_deepc/eval_looks.csv"
MAX_EVAL_LOOKS = 3
FIELDS = ["id", "reached", "fell", "truncated", "qp_fail", "steps", "final_dist", "mean_solve_s"]
_BOOLS = ("reached", "fell", "truncated", "qp_fail")
_W: dict = {}  # per-process worker state


def _init(cfg: dict) -> None:
    _W["ctrl"], _W["info"] = build_canonical_quadruped_deepc(**cfg["model"])
    _W["env"] = QuadrupedGoalEnv()
    _W["scen"] = sc.load(cfg["scenarios"])


def _work(i: int) -> dict:
    return run_episode(_W["ctrl"], _W["info"], _W["env"], _W["scen"], i)


def summarize(rows: list[dict]) -> str:
    n = len(rows)
    k = sum(bool(r["reached"]) for r in rows)
    lo, hi = wilson_ci(k, n) if n else (0.0, 0.0)
    reach = [r["steps"] for r in rows if r["reached"]]
    steps = f"{sum(reach) / len(reach):.1f}" if reach else "n/a"
    ms = 1000 * sum(r["mean_solve_s"] for r in rows) / n if n else float("nan")
    return (
        f"reached {k}/{n} ({100 * k / max(n, 1):.0f}%) Wilson [{lo:.2f}, {hi:.2f}] | "
        f"falls {sum(bool(r['fell']) for r in rows)} | "
        f"trunc {sum(bool(r['truncated']) for r in rows)} | "
        f"qp_fail {sum(bool(r['qp_fail']) for r in rows)} | "
        f"mean steps-to-reach {steps} | mean solve {ms:.1f} ms"
    )


def _load(path: str) -> tuple[list[dict], int]:
    """Parse complete rows; return (rows, byte length of the clean prefix).

    A killed run can leave a partial last line: drop it (no trailing newline, wrong
    field count, or unparsable) so the caller can truncate to the clean prefix.
    """
    if not os.path.exists(path):
        return [], 0
    with open(path, newline="") as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    rows, good = [], 0
    for n, ln in enumerate(lines):
        if not ln.endswith("\n"):
            break
        if n == 0:
            if next(csv.reader([ln])) != FIELDS:
                break
        else:
            try:
                v = next(csv.reader([ln]))
                if len(v) != len(FIELDS):
                    break
                r = dict(zip(FIELDS, v))
                for b in _BOOLS:
                    if r[b] not in ("True", "False"):
                        raise ValueError(b)
                    r[b] = r[b] == "True"
                r["id"], r["steps"] = int(r["id"]), int(float(r["steps"]))
                r["final_dist"], r["mean_solve_s"] = float(r["final_dist"]), float(r["mean_solve_s"])
            except (ValueError, StopIteration):
                break
            rows.append(r)
        good += len(ln.encode())
    return rows, good


def _read(path: str) -> list[dict]:
    return _load(path)[0]


def _run(todo, workers, cfg, w, f):
    def put(r):
        w.writerow({k: r[k] for k in FIELDS})
        f.flush()

    if not todo:
        return
    if workers == 1:
        _init(cfg)
        for i in todo:
            put(_work(i))
    else:
        with mp.Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
            for r in pool.imap_unordered(_work, todo):
                put(r)


def _bad(rows):
    return [r for r in rows if r["fell"] or r["qp_fail"]]


def main(argv=None) -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--libraries", default=DEFAULT_LIBRARIES)
    a.add_argument("--scenarios", default=sc.SCENARIOS_PATH)
    a.add_argument("--split", choices=["tune", "eval"], default="tune")
    a.add_argument("--ids", default=None, help="comma list; overrides --split")
    a.add_argument("--T-ini", type=int, default=20)
    a.add_argument("--N", type=int, default=25)
    a.add_argument("--lambda-g", type=float, default=2.0)
    a.add_argument("--lambda-y", type=float, default=3e6)
    a.add_argument("--affine", action="store_true")
    a.add_argument("--du-scale", type=float, default=25.0, help="0 = no rate limit")
    a.add_argument("--kind", choices=["deepc", "select"], default="deepc")
    a.add_argument("--n-cols", type=int, default=300)
    a.add_argument("--workers", type=int, default=10)
    a.add_argument("--out", required=True)
    a.add_argument("--prune", action="store_true")
    a.add_argument("--force-eval-look", action="store_true")
    args = a.parse_args(argv)

    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
    else:
        ids = list(sc.TUNE_IDS if args.split == "tune" else sc.EVAL_IDS)
    is_look = bool(set(ids) & set(sc.EVAL_IDS))
    if is_look and not args.force_eval_look and os.path.exists(EVAL_LOOKS_PATH):
        looks = max(sum(1 for _ in open(EVAL_LOOKS_PATH)) - 1, 0)
        if looks >= MAX_EVAL_LOOKS:
            sys.exit(f"eval set already looked at {looks} times; pass --force-eval-look")

    env = QuadrupedGoalEnv()
    sc.validate_against_env(env, sc.load(args.scenarios))

    cfg = dict(
        scenarios=args.scenarios,
        model=dict(
            libraries_path=args.libraries, T_ini=args.T_ini, N=args.N,
            lambda_g=args.lambda_g, lambda_y=args.lambda_y, affine=args.affine,
            du_scale=args.du_scale or None, kind=args.kind, n_cols=args.n_cols,
        ),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows0, good = _load(args.out)
    if os.path.exists(args.out):
        with open(args.out, "rb") as f:
            tail = f.read()[good:]
        if b"\n" in tail:  # header mismatch or a bad row mid-file: not a killed run
            sys.exit(f"{args.out}: unreadable header/row at byte {good}; refusing to touch it")
        if tail:
            os.truncate(args.out, good)  # drop a partial trailing line from a killed run
    fresh = good == 0
    cfg_path = args.out + ".cfg.json"
    if not fresh and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            if json.load(f) != cfg:
                sys.exit(f"{args.out} was produced by a different config ({cfg_path}); use a new --out")
    elif not fresh:
        print(f"warning: {cfg_path} missing; cannot check {args.out} came from this config")
    if fresh:
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=1)
    done = {r["id"] for r in rows0}
    pruned = False
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if fresh:
            w.writeheader()
            f.flush()
        if args.prune:
            head = ids[:10]
            _run([i for i in head if i not in done], args.workers, cfg, w, f)
            bad = _bad([r for r in _read(args.out) if r["id"] in head])
            if bad:
                print("PRUNED: " + ", ".join(
                    f"id {r['id']} {'fell' if r['fell'] else 'qp_fail'}" for r in bad))
                pruned = True
            done |= set(head)
        if not pruned:
            _run([i for i in ids if i not in done], args.workers, cfg, w, f)

    rows = [r for r in _read(args.out) if r["id"] in set(ids)]
    print(summarize(rows))
    if is_look:
        os.makedirs(os.path.dirname(os.path.abspath(EVAL_LOOKS_PATH)), exist_ok=True)
        new = not os.path.exists(EVAL_LOOKS_PATH)
        with open(EVAL_LOOKS_PATH, "a", newline="") as lf:
            lw = csv.writer(lf)
            if new:
                lw.writerow(["timestamp", "argv", "k", "n", "pruned", "config"])
            lw.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), " ".join(sys.argv),
                         sum(r["reached"] for r in rows), len(rows), pruned, json.dumps(cfg["model"])])


if __name__ == "__main__":
    main()

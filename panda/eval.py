"""One rollout loop and one results schema, shared by every stage of the pipeline.

This exists so DeePC, the imitation clone and the RL residual cannot end up with
three different definitions of `steps`, `effort` or `reached`. A stage supplies a
`policy(env) -> u` callable and nothing else.

Results are appended to a single CSV keyed by (method, scenario_id). Because the
scenario ids come from the frozen set, paired tests across methods -- McNemar on
`reached`, paired Wilcoxon on `steps` -- are directly available with no further
bookkeeping.
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np

from core.trace_io import write_columns
from panda import scenarios as sc

RESULTS_PATH = "data/panda_results.csv"
TRACES_DIR = "data/panda_traces"

RESULT_COLUMNS = (
    "method", "scenario_id", "scenarios_version",
    "reached", "steps", "final_dist", "min_dist", "ret", "effort", "sat_frac",
    "ncon_steps", "qp_failures", "mean_solve_ms", "n_switches", "library_hist",
)


def run_scenarios(env, policy, scenario_ids, scenarios: dict, method: str,
                  trace_ids=(), trace_dir: str = TRACES_DIR,
                  scenarios_version: str = "v1",
                  frames: list | None = None) -> list[dict]:
    """Roll `policy` over `scenario_ids`. One result dict per scenario.

    `policy(env) -> np.ndarray (7,)`. If it raises, the episode is recorded as a
    failure with `qp_failures` incremented rather than aborting the sweep -- a
    78-scenario run is long and one bad solve should not lose the rest.

    Pass a list as `frames` to also collect `env.render()` output (requires
    `render_mode="rgb_array"`). Recording video through this function rather than
    a separate rollout loop is deliberate: the footage then comes from the same
    code path that produced the measured numbers, so it cannot drift into showing
    something the results CSV never scored.
    """
    sc.validate_against_env(env, scenarios)   # env config must match what the
    #                                           scenarios were recorded under; the
    #                                           checksum protects geometry, this
    #                                           protects context. Once per run.
    trace_ids = set(trace_ids)
    rows: list[dict] = []

    for sid in scenario_ids:
        sc.reset_to(env, scenarios, int(sid))
        goal = env.goal.copy()
        if frames is not None:
            # The start pose at rest, before any action -- otherwise the scenario
            # boundary in the video is invisible and the first frame shown is
            # already one step in.
            frames.append(env.render())
        dists, rewards, efforts, sats, ncon, libs, solves = [], [], [], [], 0, [], []
        qp_failures = 0
        trace: dict[str, list] = {k: [] for k in _TRACE_KEYS}
        reached = False
        steps = 0

        while True:
            t0 = time.perf_counter()
            try:
                u = np.asarray(policy(env), dtype=np.float64).reshape(env.nq)
                # Read the library index only on a successful solve. A
                # DeePC-style policy assigns `last_library_idx` as the last
                # statement of a successful solve, so on a caught exception
                # the attribute would still hold whatever the *previous*
                # successful call left there -- crediting a failed step to a
                # library it never actually used. -1 means "no library used
                # this step", which is what a failure (or a non-finite
                # action, below) actually is.
                lib = int(getattr(policy, "last_library_idx", -1))
                step_failed = 0.0
            except Exception:
                qp_failures += 1
                u = np.zeros(env.nq)
                lib = -1
                step_failed = 1.0
            solves.append((time.perf_counter() - t0) * 1000.0)
            if not np.all(np.isfinite(u)):
                # A non-finite action poisons MuJoCo state for the rest of the
                # episode and would surface only as "did not reach" with a NaN
                # reward. Treat it as a solver failure and hold instead.
                qp_failures += 1
                u = np.zeros(env.nq)
                lib = -1
                step_failed = 1.0

            _, reward, term, trunc, info = env.step(u)
            steps += 1
            if frames is not None:
                frames.append(env.render())

            dists.append(info["distance"])
            rewards.append(float(reward))
            # info["action"]: the clipped *delta* the policy requested --
            # dimensionally what the reward's own u^T R u term penalizes.
            # Not info["ctrl"] (the absolute joint target the actuators
            # received), which differs from the delta whenever the safe-box
            # clip fires -- a substantial fraction of steps near joint
            # limits (see panda/env.py's module docstring). `effort` here is
            # a control-cost proxy, not an estimate of physical work done.
            efforts.append(float(info["action"] @ info["action"]))
            sats.append(bool(np.any(np.abs(info["action"]) >= env.delta_max - 1e-9)))
            ncon += int(info["ncon"] > 0)
            libs.append(lib)

            if int(sid) in trace_ids:
                _append_trace(trace, steps, env, info, goal, reward, lib, step_failed)

            if term:
                reached = True
                break
            if trunc:
                break

        switches = sum(1 for a, b in zip(libs, libs[1:]) if a != b and a >= 0 and b >= 0)
        hist = ";".join(str(libs.count(i)) for i in range(max(max(libs) + 1, 1))) if libs and max(libs) >= 0 else ""
        rows.append({
            "method": method,
            "scenario_id": int(sid),
            "scenarios_version": scenarios_version,
            "reached": bool(reached),
            "steps": int(steps),
            "final_dist": float(dists[-1]),
            "min_dist": float(min(dists)),
            "ret": float(sum(rewards)),
            "effort": float(sum(efforts)),
            "sat_frac": float(np.mean(sats)),
            "ncon_steps": int(ncon),
            "qp_failures": int(qp_failures),
            "mean_solve_ms": float(np.mean(solves)),
            "n_switches": int(switches),
            "library_hist": hist,
        })

        if int(sid) in trace_ids:
            d = os.path.join(trace_dir, method)
            os.makedirs(d, exist_ok=True)
            write_columns(os.path.join(d, f"traj_{int(sid)}.csv"),
                          **{k: np.asarray(v) for k, v in trace.items()})

    return rows


_TRACE_KEYS = (
    ["t"]
    + [f"qpos_{i}" for i in range(7)] + [f"qvel_{i}" for i in range(7)]
    + ["tip_x", "tip_y", "tip_z", "goal_x", "goal_y", "goal_z"]
    + [f"u_{i}" for i in range(7)] + [f"ctrl_{i}" for i in range(7)]
    + ["reward", "distance", "lib_idx", "qp_status"]
)


def _append_trace(trace, t, env, info, goal, reward, lib, qp_status=0.0) -> None:
    q = np.asarray(env.data.qpos); v = np.asarray(env.data.qvel)
    tip = info["y"]; u = info["action"]; ctrl = info["ctrl"]
    trace["t"].append(t)
    for i in range(7):
        trace[f"qpos_{i}"].append(float(q[i]))
        trace[f"qvel_{i}"].append(float(v[i]))
        trace[f"u_{i}"].append(float(u[i]))
        trace[f"ctrl_{i}"].append(float(ctrl[i]))
    for name, val in zip(("tip_x", "tip_y", "tip_z"), tip):
        trace[name].append(float(val))
    for name, val in zip(("goal_x", "goal_y", "goal_z"), goal):
        trace[name].append(float(val))
    trace["reward"].append(float(reward))
    trace["distance"].append(float(info["distance"]))
    trace["lib_idx"].append(float(lib))
    trace["qp_status"].append(float(qp_status))


def append_results(rows, path: str = RESULTS_PATH) -> None:
    """Append rows, writing the header only when creating the file.

    Raises if an existing file's header disagrees with `RESULT_COLUMNS`.
    This file is the one place DeePC, the clone and the residual all agree
    on schema; appending a row set under a different header than what's
    already on disk (e.g. after a column was added or reordered) would
    silently corrupt it -- `csv.DictReader` turns an extra column into a
    stray `None` key with no error until something reads it much later.
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    if not new:
        with open(path, newline="") as fh:
            on_disk = next(csv.reader(fh), [])
        if on_disk != list(RESULT_COLUMNS):
            raise ValueError(
                f"{path} header does not match RESULT_COLUMNS -- refusing to "
                f"append and corrupt the shared results file. "
                f"on disk: {on_disk}; expected: {list(RESULT_COLUMNS)}"
            )
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(RESULT_COLUMNS))
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def read_results(path: str = RESULTS_PATH) -> list[dict]:
    """Read back with `reached`, `steps` and the numeric columns typed."""
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            r["scenario_id"] = int(r["scenario_id"])
            r["reached"] = r["reached"] == "True"
            for k in ("steps", "ncon_steps", "qp_failures", "n_switches"):
                r[k] = int(r[k])
            for k in ("final_dist", "min_dist", "ret", "effort", "sat_frac",
                      "mean_solve_ms"):
                r[k] = float(r[k])
            out.append(r)
    return out

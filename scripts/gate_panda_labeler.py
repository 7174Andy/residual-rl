"""Phase 2 Task 1: pick the labelling expert, and check the two plants agree.

Three arms over the SAME scenario subset, paired:

  frozen    n_max=3, carry_prediction=True   -- panda_expert_v1 exactly as frozen
  cheap     n_max=1, carry_prediction=False  -- the proposed labeller
  gym       frozen config, but stepped through PandaReachEnv's DELTA plant

`frozen` vs `cheap` asks whether the cheap labeller is as good a CONTROLLER.
Measured 2026-08-27: Algorithm 1 never converges early on the Panda (3.00 of 3
iterations every step, ~1.8 s each), so n_max=1 is a ~3x cheaper labeller --
worth 3x on a job whose labelling cost is the schedule's long pole. The
memoryless half is NOT optional: `core/selectdpc.py` carries `_tau_prev` across
steps, which makes the expert recurrent and (its words) "fatal for behavioral
cloning". On the Panda that carried state moves the action by a relative 0.183,
against the reacher's 0.81, so it should be nearly free here -- this measures it.

`frozen` vs `gym` asks whether the expert's 70/78 survives the delta clip the
clone and residual will live under. The expert was scored under
`panda/qdes.py::step_qdes`, which writes `ctrl = clip(q_des, safe_box)` with NO
delta limit; `PandaReachEnv.step` writes `ctrl = clip(q + clip(dq, +-0.2), box)`.
Those are the same plant only while `|q_des - q|_inf <= 0.2`, and that was
already at 0.109 and rising 25 steps into one episode.

All three arms replicate `scripts/run_select_dpc.py::episode` exactly -- in
particular NO correction of the controller's past buffer toward the applied
target. That correction matters for DAgger labelling (see panda/clone_data.py)
but applying it here would mean the `frozen` arm no longer reproduces the
published 70/78, and comparability across arms is the whole point of this gate.

    OMP_NUM_THREADS=1 uv run python scripts/gate_panda_labeler.py --n-scen 24

Run the arms as separate processes to get 3x wall-clock for free; each holds its
own ~830 MB bank, so 3 arms is ~2.5 GB.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import mujoco
import numpy as np

from panda import scenarios as sc
from panda.model import frame_skip, load_model, safe_box, tip_id
from panda.qdes import step_qdes, y_ref_for
from panda.selectdpc import make_select_controller, panda_bank

ARMS = ("frozen", "cheap", "gym")


def make_ctrl(bank, model, arm: str, args):
    """The controller for `arm`. `gym` uses the FROZEN controller settings."""
    n_max, carry = (1, False) if arm == "cheap" else (args.n_max_frozen, True)
    ctrl = make_select_controller(
        bank, model, T_ini=args.T_ini, N=args.N, n_cols=args.n_cols,
        n_max=n_max, du_max=args.du_max)
    ctrl.carry_prediction = carry
    return ctrl


def run(arm: str, ctrl, model, data, q0, goal, args) -> dict:
    """One episode. `arm == "gym"` routes q_des through the env's delta clip."""
    lo, hi = safe_box(model)
    tip, fs = tip_id(model), frame_skip(model)
    data.qpos[:] = q0
    data.qvel[:] = 0.0
    data.ctrl[:] = q0
    mujoco.mj_forward(model, data)
    t0 = data.site_xpos[tip].copy()
    need = float(np.linalg.norm(goal - t0))
    ctrl.reset(np.concatenate([q0, t0]), u_initial=q0)
    yref = y_ref_for(goal, model.nq)

    best, clip_hits, ms, n = need, 0, [], 0
    for t in range(args.steps):
        n = t + 1
        q = np.asarray(data.qpos).copy()
        tic = time.perf_counter()
        try:
            q_des = ctrl.act(np.concatenate([q, data.site_xpos[tip]]), yref)
        except RuntimeError:
            break
        ms.append((time.perf_counter() - tic) * 1000.0)
        # The plant-hazard statistic: how far the commanded target sits from the
        # CURRENT joints, against the env's own +-0.2 delta cap. Recorded for
        # every arm, so `frozen` reports how often the gym plant WOULD differ.
        if float(np.abs(q_des - q).max()) > args.delta_max:
            clip_hits += 1
        if arm == "gym":
            # Exactly PandaReachEnv.step: clip the delta, then the safe box
            # (step_qdes applies the box clip itself).
            q_des = q + np.clip(q_des - q, -args.delta_max, args.delta_max)
        step_qdes(model, data, q_des, lo, hi, fs)
        best = min(best, float(np.linalg.norm(data.site_xpos[tip] - goal)))
        if best < args.tol:
            break
    return {"reached": int(best < args.tol), "final_mm": 1000.0 * best,
            "steps": n, "clip_rate": clip_hits / max(n, 1),
            "mean_step_ms": float(np.mean(ms)) if ms else float("nan")}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--libs", default="data/panda_taskbank_20k_sdpc.npz")
    p.add_argument("--scenarios", default="data/panda_scenarios_v1.npz")
    p.add_argument("--n-scen", type=int, default=24,
                   help="first N of the frozen 78; paired across arms")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--delta-max", type=float, default=0.2)
    p.add_argument("--n-cols", type=int, default=300)
    p.add_argument("--n-max-frozen", type=int, default=3)
    p.add_argument("--du-max", type=float, default=0.02)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--T-ini", type=int, default=5)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--out", default="data/panda_labeler_gate.csv")
    p.add_argument("--checkpoint", default="data/ck_labeler_gate.jsonl")
    args = p.parse_args()

    scen = sc.load(args.scenarios)
    model, data = load_model()
    t0 = time.perf_counter()
    with np.load(args.libs) as z:
        payload = {k: z[k] for k in z.files}
    bank = panda_bank(payload, args.T_ini, args.N, stride=args.stride)
    del payload
    print(f"bank: tau {bank['tau'].shape} ({bank['tau'].nbytes / 1e6:.0f} MB) "
          f"in {time.perf_counter() - t0:.1f}s", flush=True)

    done = set()
    if args.checkpoint and os.path.exists(args.checkpoint):
        with open(args.checkpoint) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["arm"], r["sid"]))
        print(f"resuming: {len(done)} episodes already done", flush=True)

    rows = []
    for arm in args.arms.split(","):
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; pick from {ARMS}")
        todo = [s for s in range(args.n_scen) if (arm, s) not in done]
        if not todo:
            continue
        ctrl = make_ctrl(bank, model, arm, args)
        for sid in todo:
            r = run(arm, ctrl, model, data, np.asarray(scen["qpos"][sid], float),
                    np.asarray(scen["goal"][sid], float), args)
            r.update(arm=arm, sid=sid)
            rows.append(r)
            if args.checkpoint:
                with open(args.checkpoint, "a") as f:
                    f.write(json.dumps(r) + "\n")
            print(f"{arm:7s} s{sid:2d} reached={r['reached']} "
                  f"final={r['final_mm']:7.1f}mm steps={r['steps']:3d} "
                  f"clip={r['clip_rate']:.2f} {r['mean_step_ms']:.0f}ms",
                  flush=True)

    if rows:
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
    print(f"wrote {args.out} (+{len(rows)} rows)")


if __name__ == "__main__":
    main()

"""Select-DPC vs fixed anchors on PandaReach: the gate, then closed loop.

The Panda counterpart of `scripts/run_select_dpc_reacher.py`, using the corrected
algorithm in `core/selectdpc.py` (Algorithm 1 + 2 of arXiv:2503.18845).

Expectation going in, from the anchor work: on Reacher this was worth +5 reaches
because coverage was affordable; here the pooled bank's nearest data sits ~1.98 rad
from a typical episode start, well outside the ~0.5 rad radius where
`scripts/verify_libraries.py` measures libraries to be useful. Select-DPC can only
choose from data that EXISTS, so the prediction is "better than fixed anchors,
still unusable". Worth measuring rather than assuming -- the corrected algorithm
selects against the prediction and iterates, which is a different mechanism from
the version that produced the earlier null result.

Two stages, cheapest-decisive first:

  [GATE]  open-loop, no QP. Held-out configurations drawn the way the ENV draws
          them (uniform `sample_config`), scored with `verify_libraries.py`'s
          skill/cos on identical trajectories for both predictors.
  [LOOP]  closed-loop reaching with the in-QP rate limit, against a random-walk
          control on identical episodes -- which this project learned not to omit.

    uv run python scripts/run_select_dpc.py --gate-only
    uv run python scripts/run_select_dpc.py --libs data/panda_uniform_libs.npz
"""
from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np

from panda.anchors import assign
from panda.model import (
    MIN_TIP_Z, frame_skip, load_model, safe_box, sample_config, tip_id,
)
from panda.qdes import (
    build_libraries, collect_anchor, make_controller, outputs, predict, step_qdes,
    y_ref_for,
)
from panda.selectdpc import make_select_controller, panda_bank, select_predict


def gate(model, data, anchors, bank, libs, rng, args):
    """Score fixed-library vs per-timestep selection on identical trajectories."""
    lo, hi = safe_box(model)
    tip = tip_id(model)
    T_ini, N, nq = args.T_ini, args.N, model.nq
    p_y = nq + 3
    out = {"fixed": [], "select": [], "dist": [], "ntraj": []}
    for _ in range(args.gate_n):
        q0, _ = sample_config(model, data, rng, lo, hi, tip)
        rec = collect_anchor(model, data, q0, T_ini + N + 1, rng, sigma=args.sigma)
        y = outputs(rec["q"], rec["tip"])
        u_ini, y_ini = rec["u"][:T_ini], y[:T_ini]
        u_f, yt = rec["u"][T_ini:T_ini + N], y[T_ini:T_ini + N]
        i = assign(q0, anchors, None)
        out["dist"].append(float(np.linalg.norm(q0 - anchors[i])))

        yh_f = predict(libs[i], u_ini, y_ini, u_f, args.lambda_g, N, p_y)
        yh_s, sel = select_predict(bank, u_ini, y_ini, u_f, args.n_cols,
                                   args.lambda_g, N, p_y)
        out["ntraj"].append(int(np.unique(bank["origin"][sel]).size))
        for key, yh in (("fixed", yh_f), ("select", yh_s)):
            tp, tt, st = yh[:, nq:], yt[:, nq:], y[T_ini - 1, nq:]
            mse_l = np.mean(np.sum((tp - tt) ** 2, axis=1))
            mse_0 = np.mean(np.sum((np.tile(st, (N, 1)) - tt) ** 2, axis=1))
            dp, dt = tp[-1] - st, tt[-1] - st
            den = np.linalg.norm(dp) * np.linalg.norm(dt)
            out[key].append((float(np.sqrt(mse_l)),
                             float(1.0 - mse_l / max(mse_0, 1e-15)),
                             float(dp @ dt / den) if den > 1e-12 else 0.0))
    return out


def _goal_near(model, data, q0, s, rng, lo, hi, tip, tries=40):
    for _ in range(tries):
        d = rng.standard_normal(model.nq)
        qg = np.clip(q0 + s * d / np.linalg.norm(d), lo, hi)
        data.qpos[:] = qg
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if data.ncon == 0 and data.site_xpos[tip][2] > MIN_TIP_Z:
            return data.site_xpos[tip].copy()
    return None


def oracle_step(model, data, goal, tip, dumax, prev_target, _J=[None]):
    """Greedy DLS toward the goal, under the SAME rate constraint as the QP.

    `prev_target` is the previous commanded target, and the step is bounded as
    `|u_j - u_{j-1}| <= dumax` -- a ramp from the last TARGET. An earlier version
    bounded `target - q_measured` instead, which is a different and much slower
    motion budget: measured at du_max=0.02, the ramp achieves 0.0200 rad/step
    against delta-from-measured's 0.0050, a factor of 4. Bounding the ceiling more
    tightly than the controller makes the ceiling meaningless -- and would let a
    controller "beat" it as an artifact.

    Model-based and not a controller under study: it reads the true Jacobian and
    exists only to say whether the task is achievable in the budget at all, after
    a 40-step run turned out to be near-infeasible (oracle 1/12) and therefore
    measured the budget rather than the controllers.
    """
    if _J[0] is None:
        _J[0] = np.zeros((3, model.nv))
    J = _J[0]
    mujoco.mj_jacSite(model, data, J, None, tip)
    e = goal - data.site_xpos[tip]
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), e)
    # Aim from the current pose, then clip the TARGET's movement, matching the QP.
    target = np.asarray(data.qpos) + dq
    step = target - prev_target
    n = np.linalg.norm(step)
    if n > dumax:
        step = step * dumax / n
    return prev_target + step


def episode(model, data, q0, goal, args, ctrl=None, rand_amp=None, oracle=False):
    lo, hi = safe_box(model)
    tip, fs = tip_id(model), frame_skip(model)
    data.qpos[:] = q0
    data.qvel[:] = 0.0
    data.ctrl[:] = q0
    mujoco.mj_forward(model, data)
    t0 = data.site_xpos[tip].copy()
    need = float(np.linalg.norm(goal - t0))
    if ctrl is not None:
        ctrl.reset(np.concatenate([q0, t0]), u_initial=q0)
        yref = y_ref_for(goal, model.nq)
    rw = np.random.default_rng(int(abs(q0[0]) * 1e6) % 2**31)
    best, path, prev, its = need, 0.0, t0.copy(), []
    prev_target = np.asarray(q0, dtype=np.float64).copy()
    for _t in range(args.steps):
        q = np.asarray(data.qpos).copy()
        if oracle:
            u = oracle_step(model, data, goal, tip, args.du_max, prev_target)
            prev_target = u.copy()
        elif ctrl is not None:
            try:
                u = ctrl.act(np.concatenate([q, data.site_xpos[tip]]), yref)
            except RuntimeError:
                break
            its.append(getattr(ctrl, "last_iters", 1))
        else:
            u = q + rw.uniform(-rand_amp, rand_amp, model.nq)
        step_qdes(model, data, u, lo, hi, fs)
        path += float(np.linalg.norm(data.site_xpos[tip] - prev))
        prev = data.site_xpos[tip].copy()
        best = min(best, float(np.linalg.norm(data.site_xpos[tip] - goal)))
        if best < args.tol:
            break
    net = need - best
    return {"reached": best < args.tol, "closed": 100.0 * net / need,
            "final": best, "eff": path / net if net > 1e-4 else float("nan"),
            "iters": float(np.mean(its)) if its else 1.0}


# --- scenario-level parallelism -------------------------------------------------
#
# The QP is the entire cost of this script: MuJoCo runs a control step in 0.050 ms
# while the n_cols=2400 solve takes 5.6 SECONDS -- a factor of 112,000. So batching
# the simulator buys nothing, and the solver is where the time goes. SCS is already
# the fastest option available here (measured: CLARABEL 1.8x slower, OSQP 3.1x), and
# it is single-threaded, so one episode pins one core and leaves the rest idle.
#
# Episodes are independent and `episode()` seeds its RNG from `q0`, so farming them
# across processes is bit-identical to running them in sequence -- this is a
# wall-clock change only. MuJoCo models cannot be pickled, so each worker builds its
# own model, bank and controllers once in the initializer rather than per task.
_W: dict = {}


def _init_worker(libs_path: str, args) -> None:
    model, data = load_model(servo_scale=args.servo_scale)
    with np.load(libs_path) as z:
        payload = {k: z[k] for k in z.files}
    _W.update(model=model, data=data, args=args, payload=payload,
              bank=panda_bank(payload, args.T_ini, args.N, stride=args.stride))


def _ctrl(row):
    """The worker's controller for `row`, built once and reused across episodes.

    Reuse is only safe because `DeePC.reset()` now clears the solver's warm-start
    cache; before that fix a reused controller carried each episode's final solve
    into the next one, which made results depend on evaluation order and is why a
    parallel sweep could not reproduce a serial one. `tests/test_deepc.py` pins it.
    """
    args, model = _W["args"], _W["model"]
    if row not in _W:
        _W[row] = (
            make_select_controller(
                _W["bank"], model, T_ini=args.T_ini, N=args.N, n_cols=args.n_cols,
                n_max=args.n_max, lambda_g=args.lambda_g, du_max=args.du_max,
                tip_scale=args.tip_scale)
            if row == "select" else
            make_controller(_W["payload"], model, T_ini=args.T_ini, N=args.N,
                            lambda_g=args.lambda_g, du_max=args.du_max)[0])
    return _W[row]


def _run_episode(task):
    """One (row, scenario) pair. Returns `(row, index, result)`."""
    row, i, q0, goal = task
    kw = ({"oracle": True} if row == "oracle"
          else {"rand_amp": _W["args"].du_max} if row == "random"
          else {"ctrl": _ctrl(row)})
    return row, i, episode(_W["model"], _W["data"], q0, goal, _W["args"], **kw)


def run_rows(rows, eps, libs_path, args):
    """`{row: [result per episode]}`, computed across `args.jobs` processes.

    Tasks are the full row x episode cross-product in one pool rather than a pool
    per row: `oracle` and `random` are ~4 orders of magnitude cheaper than
    `select`, so per-row pools would leave workers idle waiting on the slow row.

    Every completed episode is appended to `args.checkpoint` (jsonl) the moment
    it lands, and tasks already present there are skipped on startup -- so a
    killed run resumes where it stopped instead of forfeiting hours of QP time.
    A select episode costs up to ~1 h of solver work at n_cols=2400; before this,
    results existed only in memory until the final table printed, and one Ctrl-C
    at hour 7 of an 8-hour benchmark lost all of it. Episodes are independent and
    deterministic given (q0, goal), so a resumed run's numbers are identical to
    an uninterrupted one -- resuming changes wall-clock, not results.
    """
    import json
    import multiprocessing as mp

    out = {row: [None] * len(eps) for row in rows}
    done_keys = set()
    if args.checkpoint and os.path.exists(args.checkpoint):
        with open(args.checkpoint) as fh:
            for line in fh:
                rec = json.loads(line)
                out[rec["row"]][rec["i"]] = rec["result"]
                done_keys.add((rec["row"], rec["i"]))
        print(f"  resuming: {len(done_keys)} episodes already in {args.checkpoint}")

    tasks = [(row, i, q0, goal) for row in rows
             for i, (q0, goal) in enumerate(eps) if (row, i) not in done_keys]
    if not tasks:
        return out
    if args.jobs == 1:
        _init_worker(libs_path, args)
        done = map(_run_episode, tasks)
    else:
        ctx = mp.get_context("spawn")     # fork + MuJoCo/BLAS is not safe
        pool = ctx.Pool(args.jobs, initializer=_init_worker,
                        initargs=(libs_path, args))
        done = pool.imap_unordered(_run_episode, tasks, chunksize=1)
    ck = open(args.checkpoint, "a") if args.checkpoint else None
    n = 0
    for row, i, res in done:
        out[row][i] = res
        if ck:
            ck.write(json.dumps({"row": row, "i": i, "result": res}) + "\n")
            ck.flush()
        n += 1
        print(f"\r  {n}/{len(tasks)} episodes", end="", flush=True)
    print()
    if ck:
        ck.close()
    if args.jobs != 1:
        pool.close()
        pool.join()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--libs", default="data/panda_uniform_libs.npz",
                   help="pooled bank; uniform anchors match how episodes start")
    p.add_argument("--n-cols", type=int, default=300, help="the paper's N_cols")
    p.add_argument("--n-max", type=int, default=3)
    p.add_argument("--tip-scale", type=float, default=1.0,
                   help="1.0 = the paper's plain norm; >1 up-weights the tip block")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--gate-n", type=int, default=40)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--gate-only", action="store_true")
    p.add_argument("--skip-fixed", action="store_true",
                   help="skip the 1484-column baseline (~9 h on 78 scenarios)")
    p.add_argument("--goal-dist", type=float, default=0.4)
    p.add_argument("--du-max", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--tol", type=float, default=0.02)
    p.add_argument("--T-ini", type=int, default=5)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--lambda-g", type=float, default=5e-3)
    p.add_argument("--scenarios", default=None,
                   help="frozen scenario npz (qpos, goal) -- the repo's canonical "
                        "Panda benchmark, instead of freshly sampled near-goals")
    p.add_argument("--checkpoint", default=None,
                   help="jsonl file for per-episode results; if it exists, "
                        "episodes recorded there are skipped (resume). Use a "
                        "DIFFERENT path per configuration -- the file does not "
                        "record n_cols/libs, so resuming a changed config from "
                        "an old checkpoint silently mixes results.")
    p.add_argument("--jobs", type=int, default=0,
                   help="worker processes for the episode sweep; 0 = auto "
                        "(cpu_count-2, capped at 10), 1 = serial. Results are "
                        "identical either way -- episodes are independent and "
                        "seeded from q0, so this is wall-clock only.")
    p.add_argument("--servo-scale", type=float, default=1.0,
                   help="PD servo gain multiplier; MUST match the gains --libs "
                        "was collected at")
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()
    if args.jobs <= 0:
        args.jobs = max(1, min(10, (os.cpu_count() or 2) - 2))

    model, data = load_model(servo_scale=args.servo_scale)
    with np.load(args.libs) as z:
        payload = {k: z[k] for k in z.files}
    anchors = payload["anchors"]
    print(f"pooling {len(anchors)} trajectories from {args.libs}, "
          f"stride={args.stride} ...")
    bank = panda_bank(payload, args.T_ini, args.N, stride=args.stride)
    floor = model.nq * (args.T_ini + args.N) + 2 * model.nv
    print(f"  bank: {bank['Up'].shape[1]} columns, tau dim {bank['tau'].shape[0]}")
    print(f"  N_cols={args.n_cols} (rank floor {floor}), n_max={args.n_max}, "
          f"tip_scale={args.tip_scale}")
    libs = build_libraries(payload, args.T_ini, args.N)

    rng = np.random.default_rng(args.seed)
    g = gate(model, data, anchors, bank, libs, rng, args)
    print(f"\n[GATE] {args.gate_n} held-out configs drawn as the ENV draws them "
          f"(uniform), median:")
    print(f"  {'':<12}{'tip RMSE':>11}{'skill':>9}{'cos':>8}{'cos>0.5':>10}{'cos<0':>8}")
    for key, label in (("fixed", f"K={len(anchors)} fixed"), ("select", "Select-DPC")):
        a = np.array(g[key])
        print(f"  {label:<12}{np.median(a[:, 0]) * 1e3:>9.0f} mm"
              f"{np.median(a[:, 1]):>9.2f}{np.median(a[:, 2]):>8.2f}"
              f"{100 * np.mean(a[:, 2] > 0.5):>9.0f}%{100 * np.mean(a[:, 2] < 0):>7.0f}%")
    print(f"  median distance to nearest fixed anchor: {np.median(g['dist']):.2f} rad")
    print(f"  Select-DPC drew from {np.median(g['ntraj']):.0f} distinct trajectories "
          f"(of {len(anchors)})")
    if args.gate_only:
        return

    print(f"\n[LOOP] {args.episodes} episodes, du_max={args.du_max}, "
          f"{args.goal_dist} rad goals, {args.steps} steps")
    lo, hi = safe_box(model)
    tip = tip_id(model)
    if args.scenarios:
        # The frozen benchmark: whole-task goals (median 735 mm tip distance),
        # NOT the short near-goals `--goal-dist` generates. Feasibility at the
        # chosen du_max/steps must be checked against the oracle row -- a 40-step
        # run on near-goals was already retracted for measuring the budget.
        with np.load(args.scenarios) as zz:
            eps = [(zz["qpos"][i], zz["goal"][i]) for i in range(len(zz["qpos"]))]
        if args.episodes and args.episodes < len(eps):
            eps = eps[: args.episodes]
        print(f"  using {len(eps)} frozen scenarios from {args.scenarios}")
    else:
        rng = np.random.default_rng(args.seed + 500)
        eps = []
        while len(eps) < args.episodes:
            q0, _ = sample_config(model, data, rng, lo, hi, tip)
            goal = _goal_near(model, data, q0, args.goal_dist, rng, lo, hi, tip)
            if goal is not None:
                eps.append((q0, goal))

    rows = ["oracle", "select"] + ([] if args.skip_fixed else ["fixed"]) + ["random"]
    labels = {"oracle": "DLS oracle", "select": "Select-DPC", "random": "random",
              "fixed": f"K={len(anchors)} fixed"}
    print(f"  {args.jobs} worker process(es) over {len(rows)} rows x {len(eps)} "
          f"episodes")
    out = run_rows(rows, eps, args.libs, args)

    print(f"  {'':<14}{'reached':>9}{'median closed':>15}{'median final':>14}"
          f"{'path/net':>10}{'iters':>8}")
    for row in rows:
        res = out[row]
        print(f"  {labels[row]:<14}{sum(r['reached'] for r in res):>4}/{len(res):<4}"
              f"{np.median([r['closed'] for r in res]):>13.1f}%"
              f"{np.median([r['final'] for r in res]) * 1e3:>11.1f} mm"
              f"{np.nanmedian([r['eff'] for r in res]):>10.1f}"
              f"{np.mean([r['iters'] for r in res]):>8.2f}", flush=True)


if __name__ == "__main__":
    main()

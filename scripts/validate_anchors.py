"""Stage 3 of the anchor-selection plan: validate, then split only what fails.

Covers plan sections 8-11 (the validation phase). Three parts, in the order the
plan insists on -- geometry does not decide the anchor count, validation does:

  [8] Open-loop prediction error per region, on trajectories not used to build
      the library. Reported as the plan's `E_i`, and split by channel -- joints in
      rad, tip in mm. `E_i` alone sums rad^2 with m^2, so it cannot say which half
      is failing; the split is what identified the plan's scalar `d_g` output as
      the culprit and motivated `y = [q; p_ee]`.
  [9] Closed-loop reaching on held-out goals: success rate, final distance, steps
      to reach, joint-limit violations, command smoothness, library switches.
 [10] A split recommendation -- which single region to divide, per the plan's
      "do not automatically double all anchors".

    uv run python scripts/validate_anchors.py --libs data/panda_anchors_k4_libs.npz
    uv run python scripts/validate_anchors.py --libs ... --episodes 40 --tol 0.02
"""
from __future__ import annotations

import argparse

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


def prediction_error(model, data, payload, region, rng, args, n_traj, libs):
    """Plan section 8's `E_i` for one region, on freshly collected trajectories.

    Also splits the error by channel: the joint block in rad and the tip block in
    mm. `E_i` alone sums rad² and m² and so cannot say which half is failing --
    that split is what identified the plan's scalar `d_g` output as the culprit.
    """
    a = payload["anchors"][region]
    lo, hi = safe_box(model)
    T_ini, N = args.T_ini, args.N
    errs, q_errs, t_errs = [], [], []
    nq = model.nq
    for _ in range(n_traj):
        # A test point inside the region, NOT the anchor itself: the plan asks for
        # error "throughout its assigned region", and error at the anchor is the
        # noise floor rather than the quantity of interest.
        q0 = np.clip(a + rng.normal(0, args.test_spread, nq), lo, hi)
        rec = collect_anchor(model, data, q0, T_ini + N + 1, rng, sigma=args.sigma)
        y = outputs(rec["q"], rec["tip"])
        yh = predict(libs[region], rec["u"][:T_ini], y[:T_ini],
                     rec["u"][T_ini:T_ini + N], args.lambda_g, N, y.shape[1])
        yt = y[T_ini:T_ini + N]
        errs.append(float(np.mean(np.sum((yh - yt) ** 2, axis=1))))
        q_errs.append(float(np.sqrt(np.mean((yh[:, :nq] - yt[:, :nq]) ** 2))))
        t_errs.append(float(np.sqrt(np.mean(
            np.sum((yh[:, nq:] - yt[:, nq:]) ** 2, axis=1)))))
    return float(np.mean(errs)), float(np.mean(q_errs)), float(np.mean(t_errs))


def run_episode(model, data, deepc, info, goal, q0, args):
    """One closed-loop reaching episode. Returns the plan section 9 metrics."""
    lo, hi = info["lo"], info["hi"]
    fs, tip = frame_skip(model), tip_id(model)
    # No retargeting: with y = [q; p_ee] the libraries are goal-free and the goal
    # enters only here, through the reference.
    y_ref = y_ref_for(goal, info["nq"])

    data.qpos[:] = q0
    data.qvel[:] = 0.0
    data.ctrl[:] = q0
    mujoco.mj_forward(model, data)
    deepc.reset(np.concatenate([q0, data.site_xpos[tip]]), u_initial=q0)

    switches, viol, prev, us = 0, 0, -1, []
    best = float(np.linalg.norm(data.site_xpos[tip] - goal))
    for t in range(args.max_steps):
        y = np.concatenate([np.asarray(data.qpos), np.asarray(data.site_xpos[tip])])
        try:
            u = deepc.act(y, y_ref)
        except RuntimeError:
            return {"reached": False, "steps": t, "final": best, "switches": switches,
                    "viol": viol, "smooth": float("nan"), "failed": True}
        if deepc.last_library_idx != prev and prev >= 0:
            switches += 1
        prev = deepc.last_library_idx
        # Rate limit on the applied command. `u` is an ABSOLUTE joint target, so
        # nothing in `u_bounds` (the safe box) stops the QP asking for a jump
        # across the workspace in one 20 ms tick -- measured median 1.9-3.1 rad.
        # This is the job DELTA_MAX did for the delta interface. It keeps the
        # command inside the amplitude where `test_cluster_lti.py` measures
        # superposition holding to ~14-34% rather than ~52-61%.
        # SUPERSEDED for real use by `core.deepc.DeePC(du_max=...)`, which puts the
        # bound INSIDE the QP so the optimizer plans a feasible trajectory. This
        # post-hoc path is retained only so `--rate-sweep` can measure clip-after
        # against constrain-inside on identical episodes; applied here the plan is
        # still premised on a move the plant never makes.
        if args.rate_limit is not None:
            u = np.clip(u, np.asarray(data.qpos) - args.rate_limit,
                        np.asarray(data.qpos) + args.rate_limit)
        us.append(u.copy())
        step_qdes(model, data, u, lo, hi, fs)
        # A safe-box EXCURSION of qpos, not a jnt_range violation -- the command is
        # clipped unconditionally, but momentum can still carry a joint past it.
        viol += int(np.any(np.asarray(data.qpos) < lo - 1e-9)
                    or np.any(np.asarray(data.qpos) > hi + 1e-9))
        d = float(np.linalg.norm(data.site_xpos[tip] - goal))
        best = min(best, d)
        if d < args.tol:
            U = np.array(us)
            return {"reached": True, "steps": t + 1, "final": d, "switches": switches,
                    "viol": viol, "failed": False,
                    "smooth": float(np.abs(np.diff(U, axis=0)).mean()) if len(U) > 1 else 0.0}
    U = np.array(us)
    return {"reached": False, "steps": args.max_steps, "final": best,
            "switches": switches, "viol": viol, "failed": False,
            "smooth": float(np.abs(np.diff(U, axis=0)).mean()) if len(U) > 1 else 0.0}


def _goal_at_joint_distance(model, data, q0, s, rng, lo, hi, tip, tries=40):
    """FK goal of a valid configuration `s` rad away from `q0` in joint space."""
    for _ in range(tries):
        d = rng.standard_normal(model.nq)
        q_g = np.clip(q0 + s * d / np.linalg.norm(d), lo, hi)
        data.qpos[:] = q_g
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if data.ncon == 0 and data.site_xpos[tip][2] > MIN_TIP_Z:
            return q_g, data.site_xpos[tip].copy()
    return None, None


def rate_sweep(model, data, deepc, info, rng, args) -> None:
    """Confirm the rate limit on PAIRED episodes.

    Every rate limit sees the identical `(start, goal)` list, so the comparison is
    within-pair. That matters here: single episodes swung 1-71% under the same
    setting during the exploratory run, variance far larger than the effect being
    looked for. Paired differences remove the episode-to-episode term entirely,
    which is what makes n=8 informative rather than suggestive.
    """
    lo, hi = safe_box(model)
    tip = tip_id(model)
    eps = []
    while len(eps) < args.rate_eps:
        q0, _ = sample_config(model, data, rng, lo, hi, tip)
        _, goal = _goal_at_joint_distance(model, data, q0, args.rate_scale,
                                          rng, lo, hi, tip)
        if goal is not None:
            eps.append((q0, goal))

    saved_steps, saved_rl = args.max_steps, args.rate_limit
    args.max_steps = args.rate_steps
    print(f"\n[rate] {len(eps)} PAIRED episodes, goal {args.rate_scale} rad away, "
          f"max {args.rate_steps} steps, tol={args.tol * 1e3:.0f} mm")
    results = {}
    for rl in args.rate_limits:
        args.rate_limit = None if rl <= 0 else rl
        closed, hits = [], 0
        for q0, goal in eps:
            data.qpos[:] = q0
            mujoco.mj_forward(model, data)
            need = float(np.linalg.norm(goal - data.site_xpos[tip]))
            m = run_episode(model, data, deepc, info, goal, q0, args)
            closed.append(100.0 * (need - m["final"]) / need)
            hits += int(m["reached"])
        results[rl] = np.array(closed)
        lbl = "none" if rl <= 0 else f"{rl:g} rad/step"
        print(f"  {lbl:>14}: reached {hits}/{len(eps)}   "
              f"median {np.median(closed):5.1f}% closed   "
              f"per-episode " + " ".join(f"{c:5.1f}" for c in closed), flush=True)
    args.max_steps, args.rate_limit = saved_steps, saved_rl

    base = results.get(0.0)
    if base is not None:
        print("\n  paired vs no rate limit (sign test on per-episode differences):")
        for rl, v in results.items():
            if rl <= 0:
                continue
            d = v - base
            wins = int((d > 0).sum())
            print(f"    {rl:g} rad/step: better on {wins}/{len(d)} episodes, "
                  f"median gain {np.median(d):+5.1f} pp")


def near_goal_sweep(model, data, deepc, info, rng, args) -> None:
    """Is the binding failure reachability, or accuracy?

    Sweeps how far the goal is placed from the start IN JOINT SPACE. The library
    around each anchor was excited at `sigma` rad, so `Uf @ g` -- a combination of
    its columns -- can only express commands within roughly that neighbourhood.
    If success collapses once the required travel exceeds that radius while near
    goals succeed, the controller works and is range-limited. If even the nearest
    goals fail, something is broken and no amount of anchor tuning is relevant.
    """
    lo, hi = safe_box(model)
    tip = tip_id(model)
    print(f"\n[near-goal] required travel vs success, {args.near_eps} episodes/scale, "
          f"max {args.near_steps} steps, tol={args.tol * 1e3:.0f} mm")
    print(f"  library excitation sigma = {args.sigma} rad -- expect the cliff near here")
    print(f"\n  {'travel (rad)':>13}{'reached':>9}{'tip travel needed':>19}"
          f"{'median final':>14}{'tip moved':>12}")
    saved = args.max_steps
    args.max_steps = args.near_steps
    for s in args.near_scales:
        hits, finals, needed, moved = 0, [], [], []
        for _ in range(args.near_eps):
            q0, tip0 = sample_config(model, data, rng, lo, hi, tip)
            _, goal = _goal_at_joint_distance(model, data, q0, s, rng, lo, hi, tip)
            if goal is None:
                continue
            needed.append(float(np.linalg.norm(goal - tip0)))
            m = run_episode(model, data, deepc, info, goal, q0, args)
            hits += int(m["reached"])
            finals.append(m["final"])
            moved.append(needed[-1] - m["final"])
        if not finals:
            continue
        print(f"  {s:>13.2f}{hits:>4}/{len(finals):<4}"
              f"{np.median(needed) * 1e3:>16.0f} mm"
              f"{np.median(finals) * 1e3:>11.0f} mm{np.median(moved) * 1e3:>9.0f} mm")
    args.max_steps = saved


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--libs", default="data/panda_anchors_k4_libs.npz")
    p.add_argument("--episodes", type=int, default=30, help="held-out closed-loop goals")
    p.add_argument("--pred-traj", type=int, default=8, help="held-out traj per region")
    p.add_argument("--tol", type=float, default=0.02,
                   help="reach threshold, m (plan section 9 uses 0.02)")
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--T-ini", type=int, default=5)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--lambda-g", type=float, default=5e-3)
    p.add_argument("--lambda-y", type=float, default=7.5e3)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--test-spread", type=float, default=0.15,
                   help="rad, std of the test-point offset from the anchor")
    p.add_argument("--skip-closed-loop", action="store_true")
    p.add_argument("--rate-limit", type=float, default=None,
                   help="clip the applied command to q +- this many rad/step")
    p.add_argument("--rate-sweep", action="store_true",
                   help="paired confirmation of the rate limit")
    p.add_argument("--rate-limits", type=float, nargs="+",
                   default=[0.0, 0.10, 0.05, 0.02],
                   help="rate limits to compare; 0 means none")
    p.add_argument("--rate-eps", type=int, default=8, help="paired episodes")
    p.add_argument("--rate-scale", type=float, default=0.3,
                   help="joint-space distance from start to goal, rad")
    p.add_argument("--rate-steps", type=int, default=30)
    p.add_argument("--near-goal", action="store_true",
                   help="run the reachability sweep instead of the full section-9 eval")
    p.add_argument("--near-scales", type=float, nargs="+",
                   default=[0.1, 0.2, 0.4, 0.8, 1.6],
                   help="joint-space distances from start to goal, rad")
    p.add_argument("--near-eps", type=int, default=2, help="episodes per scale")
    p.add_argument("--near-steps", type=int, default=40, help="max steps per episode")
    p.add_argument("--seed", type=int, default=1, help="held out from collection's seed")
    args = p.parse_args()

    model, data = load_model()
    rng = np.random.default_rng(args.seed)
    with np.load(args.libs) as z:
        payload = {k: z[k] for k in z.files}
    anchors = payload["anchors"]
    w = payload["weights"] if payload["weights"].size else None
    k = len(anchors)
    print(f"validating {k} anchors from {args.libs}")

    # Built once -- goal-free under y = [q; p_ee], so the same libraries serve the
    # prediction test and every closed-loop episode.
    libs = build_libraries(payload, args.T_ini, args.N)

    if args.near_goal or args.rate_sweep:
        deepc, info = make_controller(payload, model, T_ini=args.T_ini, N=args.N,
                                      lambda_g=args.lambda_g,
                                      lambda_y=args.lambda_y, weights=w)
        (rate_sweep if args.rate_sweep else near_goal_sweep)(
            model, data, deepc, info, rng, args)
        return

    print(f"\n[8] open-loop prediction, {args.pred_traj} held-out trajectories/region")
    print(f"  {'region':>7}{'E_i':>14}{'q RMSE (rad)':>15}{'tip RMSE (mm)':>16}")
    E, Qe, Tp = np.zeros(k), np.zeros(k), np.zeros(k)
    for j in range(k):
        E[j], Qe[j], Tp[j] = prediction_error(model, data, payload, j, rng, args,
                                              args.pred_traj, libs)
        print(f"  {j:>7}{E[j]:>14.5f}{Qe[j]:>15.4f}{Tp[j] * 1e3:>16.1f}")
    worst = int(np.argmax(Tp))

    if not args.skip_closed_loop:
        print(f"\n[9] closed-loop reaching, {args.episodes} held-out goals, "
              f"tol={args.tol * 1e3:.0f} mm, max {args.max_steps} steps")
        deepc, info = make_controller(payload, model, T_ini=args.T_ini,
                                      N=args.N, lambda_g=args.lambda_g,
                                      lambda_y=args.lambda_y, weights=w)
        lo, hi = safe_box(model)
        tip = tip_id(model)
        rows, region_hits = [], {j: [] for j in range(k)}
        for e in range(args.episodes):
            q0, _ = sample_config(model, data, rng, lo, hi, tip)
            qg, goal = sample_config(model, data, rng, lo, hi, tip)
            m = run_episode(model, data, deepc, info, goal, q0, args)
            rows.append(m)
            region_hits[assign(qg, anchors, w)].append(m["reached"])
            print(f"  ep {e:>3}  {'REACH' if m['reached'] else ' miss'}  "
                  f"final {m['final'] * 1e3:>7.1f} mm  steps {m['steps']:>4}  "
                  f"switch {m['switches']:>3}"
                  + ("  QP-FAIL" if m["failed"] else ""))
        R = [r for r in rows if not r["failed"]]
        n_ok = sum(r["reached"] for r in rows)
        print(f"\n  success rate       : {n_ok}/{len(rows)} = {n_ok / len(rows) * 100:.0f}%")
        print(f"  median final dist  : {np.median([r['final'] for r in rows]) * 1e3:.1f} mm")
        if n_ok:
            print(f"  median steps (hit) : {np.median([r['steps'] for r in rows if r['reached']]):.0f}")
        print(f"  safe-box excursions: {sum(r['viol'] for r in rows)}")
        print(f"  library switches   : {np.mean([r['switches'] for r in rows]):.1f} / episode")
        if R:
            print(f"  command smoothness : {np.nanmean([r['smooth'] for r in R]):.4f} rad/step")
        print(f"  QP failures        : {sum(r['failed'] for r in rows)}")
        print("\n  by goal region:")
        for j in range(k):
            h = region_hits[j]
            print(f"    region {j}: {sum(h)}/{len(h)}" if h else f"    region {j}: no goals drawn")
        by_region = {j: (sum(h) / len(h) if h else 1.0) for j, h in region_hits.items()}
        worst = min(range(k), key=lambda j: (by_region[j], -float(Tp[j])))

    print(f"\n[10] recommendation: split region {worst} "
          f"(worst tip RMSE {Tp[worst] * 1e3:.1f} mm"
          + ("" if args.skip_closed_loop else ", lowest regional reach rate") + ")")
    print("  The plan splits ONE region rather than doubling K. panda.anchors.split_region")
    print(f"  replaces that anchor with two and leaves the other {k - 1} libraries reusable:")
    print("    from panda.anchors import split_region")
    print(f"    new = split_region(Q, labels, anchors, {worst}, rng)   # Q/labels from the anchors npz")
    print(f"  Then re-run collect_anchor_libraries.py and this script at K={k + 1}.")


if __name__ == "__main__":
    main()

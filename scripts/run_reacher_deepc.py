"""Local-library DeePC on Reacher-v5: the same pipeline, at a tractable budget.

The Panda work established a controller that steers well (path/net 1.0) inside
~0.5 rad of an anchor and is anti-informative beyond ~2, and that covering its
~5.7-dimensional configuration set at that radius needs ~10^5 trajectories --
confirmed by three independent routes. Reacher asks the same question where the
answer is affordable: 2 joints, no redundancy, no gravity term, direct torque.

Three stages, cheapest-decisive first, each with the control that the Panda work
learned not to omit.

  [GATE]  skill / cos vs distance from the anchor (no QP). Same metrics as
          `scripts/verify_libraries.py`.
  [LOOP]  closed-loop reaching from uniform starts to `Reacher-v5`'s own goal
          distribution, against a random-torque control on identical episodes.
  path/net is reported throughout: a controller that steers runs 1-2x.

    uv run python scripts/run_reacher_deepc.py --grid 6 5
    uv run python scripts/run_reacher_deepc.py --grid 8 7 --episodes 20
"""
from __future__ import annotations

import argparse

import numpy as np

from reacher.deepc_setup import (
    anchor_grid, build_libraries, collect_anchor, make_controller, outputs,
    predict, y_ref_for,
)
from reacher.model import (
    NQ_ARM, config_distance, fingertip, frame_skip, load_model, safe_box,
    sample_config, sample_goal, set_state, step_torque,
)


def gate(model, data, libs, anchors, rng, args) -> None:
    """Prediction skill/cos as a function of distance from the assigned anchor."""
    fs = frame_skip(model)
    lo, hi = safe_box(model)
    T_ini, N, p_y = args.T_ini, args.N, NQ_ARM + 2
    print(f"\n[GATE] {args.gate_n} trajectories per radius, sigma={args.sigma}")
    print(f"  {'radius':>8}{'tip RMSE':>11}{'skill':>9}{'cos':>8}{'cos>0.5':>10}{'cos<0':>8}")
    for radius in args.radii:
        S, C, R = [], [], []
        for _ in range(args.gate_n):
            i = int(rng.integers(len(anchors)))
            a = anchors[i]
            if radius == 0.0:
                q0 = a.copy()
            else:
                v = rng.standard_normal(NQ_ARM)
                q0 = a + radius * v / np.linalg.norm(v)
                q0[1] = np.clip(q0[1], lo[1], hi[1])
            rec = collect_anchor(model, data, q0, T_ini + N + 1, rng, sigma=args.sigma)
            y = outputs(rec["q"], rec["tip"])
            yh = predict(libs[i], rec["u"][:T_ini], y[:T_ini],
                         rec["u"][T_ini:T_ini + N], args.lambda_g, N, p_y)
            yt = y[T_ini:T_ini + N]
            tp, tt, st = yh[:, NQ_ARM:], yt[:, NQ_ARM:], y[T_ini - 1, NQ_ARM:]
            mse_l = np.mean(np.sum((tp - tt) ** 2, axis=1))
            mse_0 = np.mean(np.sum((np.tile(st, (N, 1)) - tt) ** 2, axis=1))
            R.append(np.sqrt(mse_l))
            S.append(1.0 - mse_l / max(mse_0, 1e-15))
            dp, dt = tp[-1] - st, tt[-1] - st
            den = np.linalg.norm(dp) * np.linalg.norm(dt)
            C.append(float(dp @ dt / den) if den > 1e-12 else 0.0)
        S, C, R = map(np.array, (S, C, R))
        print(f"  {radius:>8.2f}{np.median(R) * 1e3:>9.1f} mm{np.median(S):>9.2f}"
              f"{np.median(C):>8.2f}{100 * np.mean(C > 0.5):>9.0f}%"
              f"{100 * np.mean(C < 0):>7.0f}%")
    _ = fs


def episode(model, data, q0, goal, args, ctrl=None, rand_amp=None) -> dict:
    fs = frame_skip(model)
    set_state(model, data, q0, goal)
    t0 = fingertip(data)
    need = float(np.linalg.norm(goal - t0))
    if ctrl is not None:
        ctrl.reset(np.concatenate([q0, t0]), u_initial=np.zeros(NQ_ARM))
        yref = y_ref_for(goal)
    rw = np.random.default_rng(int(abs(q0[0]) * 1e6) % 2**31)
    best, path, prev, sw, prev_i = need, 0.0, t0.copy(), 0, -1
    for _t in range(args.steps):
        if ctrl is not None:
            y = np.concatenate([np.asarray(data.qpos[:NQ_ARM]), fingertip(data)])
            try:
                u = ctrl.act(y, yref)
            except RuntimeError:
                break
            if ctrl.last_library_idx != prev_i and prev_i >= 0:
                sw += 1
            prev_i = ctrl.last_library_idx
        else:
            u = rw.uniform(-rand_amp, rand_amp, NQ_ARM)
        step_torque(model, data, u, fs)
        path += float(np.linalg.norm(fingertip(data) - prev))
        prev = fingertip(data)
        best = min(best, float(np.linalg.norm(fingertip(data) - goal)))
        if best < args.tol:
            break
    net = need - best
    return {"reached": best < args.tol, "closed": 100.0 * net / need,
            "final": best, "eff": path / net if net > 1e-5 else float("nan"),
            "switches": sw}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, nargs=2, default=[6, 5],
                   help="anchors along (q0, q1)")
    p.add_argument("--T", type=int, default=1200)
    p.add_argument("--gate-n", type=int, default=12)
    p.add_argument("--radii", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps", type=int, default=50, help="Reacher-v5 uses 50")
    p.add_argument("--tol", type=float, default=0.01, help="reach threshold, m")
    p.add_argument("--du-max", type=float, default=None)
    p.add_argument("--T-ini", type=int, default=5)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--lambda-g", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, data = load_model()
    rng = np.random.default_rng(args.seed)
    anchors = anchor_grid(model, *args.grid)
    print(f"Reacher-v5: {len(anchors)} anchors on a {args.grid[0]}x{args.grid[1]} "
          f"(q0, q1) grid, T={args.T}")
    worst = max(float(config_distance(a, anchors[np.arange(len(anchors)) != i]).min())
                for i, a in enumerate(anchors))
    print(f"  anchor spacing (max nearest-neighbour): {worst:.2f} rad")

    payload = {"anchors": anchors}
    spreads = []
    for i, a in enumerate(anchors):
        rec = collect_anchor(model, data, a, args.T, rng)
        payload[f"u_{i}"], payload[f"q_{i}"], payload[f"tip_{i}"] = (
            rec["u"], rec["q"], rec["tip"])
        spreads.append(rec["spread"])
    print(f"  collection spread around anchors: median {np.median(spreads):.2f} rad, "
          f"max {max(spreads):.2f}")
    libs = build_libraries(payload, args.T_ini, args.N)
    M = np.vstack(libs[0])
    print(f"  Hankel: {M.shape[1]} columns, {M.shape[0]} rows, "
          f"rank {np.linalg.matrix_rank(M)} (floor "
          f"{NQ_ARM * (args.T_ini + args.N) + 2 * NQ_ARM})")

    gate(model, data, libs, anchors, rng, args)

    print(f"\n[LOOP] {args.episodes} episodes, {args.steps} steps, "
          f"tol={args.tol * 1e3:.0f} mm, du_max={args.du_max}")
    deepc, _ = make_controller(payload, T_ini=args.T_ini, N=args.N,
                               lambda_g=args.lambda_g, du_max=args.du_max)
    rng = np.random.default_rng(args.seed + 99)
    eps = [(sample_config(model, data, rng)[0], sample_goal(rng))
           for _ in range(args.episodes)]
    print(f"  {'':<12}{'reached':>10}{'median closed':>15}{'median final':>14}"
          f"{'path/net':>10}{'switch/ep':>11}")
    for label, kw in (("DeePC", dict(ctrl=deepc)), ("random", dict(rand_amp=1.0))):
        res = [episode(model, data, q0, g, args, **kw) for q0, g in eps]
        hits = sum(r["reached"] for r in res)
        print(f"  {label:<12}{hits:>4}/{len(res):<5}"
              f"{np.median([r['closed'] for r in res]):>13.1f}%"
              f"{np.median([r['final'] for r in res]) * 1e3:>11.1f} mm"
              f"{np.nanmedian([r['eff'] for r in res]):>10.1f}"
              f"{np.mean([r['switches'] for r in res]):>11.1f}", flush=True)


if __name__ == "__main__":
    main()

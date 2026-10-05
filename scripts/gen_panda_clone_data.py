"""Collect clone training data from panda_expert_v1 (Phase 2).

Round 0 (expert drives, plain BC by construction):

    OMP_NUM_THREADS=1 uv run python scripts/gen_panda_clone_data.py \
      --out data/pcd_r0.npz --episodes 100 --jobs 4 \
      --checkpoint data/ck_pcd_r0.jsonl

DAgger round (the clone drives, the expert labels):

    OMP_NUM_THREADS=1 uv run python scripts/gen_panda_clone_data.py \
      --out data/pcd_r1.npz --episodes 100 --jobs 4 --round 1 \
      --clone data/panda_clone_r0.pt --checkpoint data/ck_pcd_r1.jsonl

Episode `(q0, goal)` pairs are drawn from a round-offset RNG the way
`PandaReachEnv.reset` draws them, and are DISJOINT from the frozen 78, which
stay held out for evaluation.

This is the schedule's long pole: ~150 steps x 100 episodes x ~1.8 s/step is
~7.5 worker-hours per round, so `--jobs` and `--checkpoint` are both load
bearing. Each worker holds its own ~0.9 GB bank; Phase 1 lost zero work across
three host-memory kills because the checkpoint resumes exactly.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from panda.clone_data import rollout
from panda.model import load_model, safe_box, sample_config, tip_id
from panda.selectdpc import make_select_controller, panda_bank

_W: dict = {}


def episode_specs(model, data, rng, n: int, min_dist: float, lo, hi, tip,
                  goal_box=None):
    """`n` `(q0, goal)` pairs, drawn as `PandaReachEnv.reset` draws them.

    Goal first, by forward kinematics from a random valid configuration -- that
    is what guarantees every goal is reachable, which a Cartesian box sample
    would not. With `goal_box` (3, 2), out-of-box goals are rejected, as in
    `PandaReachEnv(goal_box=...)`. Then a start at least `min_dist` away.
    """
    out = []
    for _ in range(n):
        while True:
            _q, goal = sample_config(model, data, rng, lo, hi, tip)
            if goal_box is None or np.all(
                    (goal >= goal_box[:, 0]) & (goal <= goal_box[:, 1])):
                break
        for _ in range(100):
            q0, t0 = sample_config(model, data, rng, lo, hi, tip)
            if np.linalg.norm(t0 - goal) >= min_dist:
                break
        out.append((q0, goal))
    return out


def _init_worker(libs: str, clone_path: str | None, args) -> None:
    """Per-process setup. MuJoCo models cannot be pickled, so each worker builds
    its own model, bank and controller once here rather than per task."""
    model, data = load_model()
    with np.load(libs) as z:
        payload = {k: z[k] for k in z.files}
    bank = panda_bank(payload, args.T_ini, args.N, stride=args.stride)
    del payload
    ctrl = make_select_controller(
        bank, model, T_ini=args.T_ini, N=args.N, n_cols=args.n_cols,
        n_max=args.n_max, du_max=args.du_max)
    # Memoryless by default: a feedforward clone cannot fit a recurrent label.
    ctrl.carry_prediction = args.carry
    predictor = None
    if clone_path:
        from rl.clone import load_clone      # torch stays out of panda/
        predictor = load_clone(clone_path)
    _W.update(model=model, data=data, ctrl=ctrl, args=args, predictor=predictor)


def _run(task):
    """One episode in a worker. `task = (idx, q0, goal)`."""
    idx, q0, goal = task
    args, model, data = _W["args"], _W["model"], _W["data"]
    predictor = _W["predictor"]
    lo, hi = safe_box(model)

    policy = None
    if predictor is not None:
        # Same convention as validate_panda_clone.ClonePolicy: a delta clone
        # regresses q_des - q, so q is added back. Read as absolute it would
        # silently drive every episode toward the joint origin.
        target = predictor.stats.get("target", "absolute")
        if target not in ("absolute", "delta"):
            raise ValueError(f"DAgger driver does not support target={target!r}")

        def policy(features, q, _p=predictor, _lo=lo, _hi=hi, _d=target == "delta"):
            return np.clip((q if _d else 0.0) + _p.predict(features), _lo, _hi)

    # Per-episode stream so DART noise is reproducible and independent of how
    # the episodes happen to be distributed across workers.
    rng = np.random.default_rng(args.seed + 7_919 * args.round + idx)
    try:
        rec = rollout(model, data, _W["ctrl"], np.asarray(q0), np.asarray(goal),
                      max_steps=args.steps, tol=args.tol, T_ini=args.T_ini,
                      policy=policy, delta_max=args.delta_max,
                      noise_sigma=args.noise_sigma, rng=rng)
    except RuntimeError:
        return idx, None
    return idx, rec


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--libs", default="data/panda_taskbank_20k_sdpc.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--round", type=int, default=0,
                   help="offsets the episode RNG so rounds do not repeat states")
    p.add_argument("--clone", default=None,
                   help="checkpoint to DRIVE with; omit for round 0")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--delta-max", type=float, default=0.2)
    p.add_argument("--noise-sigma", type=float, default=0.0,
                   help="DART (Laskey et al. 2017): Gaussian noise added to the "
                        "APPLIED target while the label stays the expert's clean "
                        "q_des. Perturbs the expert slightly off its own path so "
                        "collection contains recovery, which plain collection "
                        "never does. 0 disables it (the default, so every "
                        "existing command is unchanged).")
    p.add_argument("--min-dist", type=float, default=0.25)
    p.add_argument("--goal-box", type=float, nargs=6, default=None,
                   metavar=("XLO", "XHI", "YLO", "YHI", "ZLO", "ZHI"),
                   help="restrict goals to this box (m); starts stay random")
    p.add_argument("--n-cols", type=int, default=300)
    p.add_argument("--n-max", type=int, default=1, help="Task 1 labeller value")
    p.add_argument("--carry", action="store_true",
                   help="keep Select-DPC's recurrent state; OFF by default "
                        "because a feedforward clone cannot fit a recurrent label")
    p.add_argument("--du-max", type=float, default=0.02)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--T-ini", type=int, default=5)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default="clone_panda")
    args = p.parse_args()

    model, data = load_model()
    lo, hi = safe_box(model)
    tip = tip_id(model)
    # Round-offset RNG: rounds must not revisit the same states, and none of
    # these may collide with the frozen 78 (drawn from their own seeds).
    rng = np.random.default_rng(args.seed + 1_000_003 * args.round)
    goal_box = None if args.goal_box is None else np.reshape(args.goal_box, (3, 2))
    specs = episode_specs(model, data, rng, args.episodes, args.min_dist,
                          lo, hi, tip, goal_box)
    tasks = [(i, q0, goal) for i, (q0, goal) in enumerate(specs)]

    shard = f"{args.out}.parts"
    os.makedirs(shard, exist_ok=True)
    done_idx: set = set()
    if args.checkpoint and os.path.exists(args.checkpoint):
        with open(args.checkpoint) as f:
            done_idx = {json.loads(line)["idx"] for line in f}
        print(f"resuming: {len(done_idx)}/{args.episodes} episodes already done",
              flush=True)
    todo = [t for t in tasks if t[0] not in done_idx]

    stats = []
    if todo:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker,
                                 initargs=(args.libs, args.clone, args)) as ex:
            for idx, rec in ex.map(_run, todo):
                if rec is None:
                    print(f"episode {idx}: QP failed, dropped", flush=True)
                    continue
                np.savez(os.path.join(shard, f"{idx}.npz"),
                         features=rec["features"], actions=rec["actions"])
                row = {"idx": idx, "reached": int(rec["reached"]),
                       "final_mm": 1000.0 * rec["final"], "steps": rec["steps"],
                       "clip_rate": rec["clip_rate"]}
                stats.append(row)
                if args.checkpoint:
                    with open(args.checkpoint, "a") as f:
                        f.write(json.dumps(row) + "\n")
                print(f"episode {idx}: reached={row['reached']} "
                      f"final={row['final_mm']:7.1f}mm "
                      f"clip={row['clip_rate']:.2f}", flush=True)

    feats, acts = [], []
    for i in range(args.episodes):
        f = os.path.join(shard, f"{i}.npz")
        if not os.path.exists(f):
            continue
        with np.load(f) as z:
            feats.append(z["features"])
            acts.append(z["actions"])
    if not feats:
        raise RuntimeError(f"every one of {args.episodes} episodes failed")

    X, Y = np.vstack(feats), np.vstack(acts)
    meta = {"episodes": args.episodes, "n_collected": len(feats),
            "round": args.round, "clone": args.clone, "n_max": args.n_max,
            "carry": args.carry, "n_cols": args.n_cols, "stride": args.stride,
            "T_ini": args.T_ini, "N": args.N, "seed": args.seed,
            "steps": args.steps, "libs": args.libs, "du_max": args.du_max,
            "noise_sigma": args.noise_sigma, "goal_box": args.goal_box}
    np.savez(args.out, features=X, actions=Y, meta=json.dumps(meta))
    print(f"wrote {args.out}: {X.shape[0]} rows from {len(feats)} episodes")

    if args.wandb_project and stats:
        from rl.wb import init_run
        run = init_run(args.wandb_project, args.wandb_name, config=meta,
                       tags=["panda", "clone", f"round{args.round}"],
                       group=args.wandb_group)
        if run is not None:
            run.log({
                "panda/collect_reach": float(np.mean([s["reached"] for s in stats])),
                "panda/collect_clip_rate": float(np.mean([s["clip_rate"] for s in stats])),
                "panda/collect_final_mm": float(np.mean([s["final_mm"] for s in stats])),
                "panda/collect_episodes": len(stats),
            })
            run.finish()


if __name__ == "__main__":
    main()

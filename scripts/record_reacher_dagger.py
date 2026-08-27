"""Paired videos: behavioral cloning vs DAgger, same clone architecture, same scenarios.

The only difference between the two arms is WHERE the training states came from.
`data/reacher_clone_ml.pt` is round 0 -- 10,000 rows collected by rolling the
Select-DPC expert and labelling with it, i.e. the collection every dataset in this
repo used before journey 13. `data/dagger_clone_r3.pt` adds 15,000 rows collected
by rolling the CLONE and labelling with the expert. Same net, same features, same
epochs; 22/40 vs 27/40 on the gate.

Early stopping is OFF, so every clip runs the full 50 steps -- the BC clone's
signature failure is arriving and then wandering off, which a clip that stops at
first contact cannot show. The readout carries both `now` and `best`.

Scenarios are picked by outcome from the frozen 120, not by index:

    rescue      DAgger reaches, BC misses     -- what on-policy data bought
    both        both reach                    -- the precision difference
    bc_drift    widest BC best->final gap     -- arrives, then leaves
    regression  BC reaches, DAgger misses     -- 14 of 120; DAgger is not free
    neither     both miss                     -- what DAgger does not fix

    uv run python scripts/record_reacher_dagger.py --scan 120
"""
from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import numpy as np

import reacher  # noqa: F401  registers the Gym ID
from core.video_encoding import encode_video
from reacher.eval import ClonePolicy, run_episode
from rl.clone import load_clone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record_reacher_residual import record  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", default="data/reacher_scenarios_v1.npz")
    p.add_argument("--bc-clone", default="data/reacher_clone_ml.pt",
                   help="round-0 clone: expert-driven data only, no DAgger")
    p.add_argument("--dagger-clone", default="data/dagger_clone_r3.pt")
    p.add_argument("--scan", type=int, default=120)
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--size", type=int, default=720)
    p.add_argument("--out-dir", default="videos/reacher_dagger")
    p.add_argument("--episode", type=int, default=None,
                   help="render this scenario index only and skip the scan")
    args = p.parse_args()

    with np.load(args.scenarios) as z:
        eps = [(z["qpos"][i], z["goal"][i]) for i in range(args.scan)]

    arms = [("BC clone (no DAgger)", "bc", load_clone(args.bc_clone, device="cpu")),
            ("DAgger clone (round 3)", "dagger",
             load_clone(args.dagger_clone, device="cpu"))]

    if args.episode is not None:
        picks = [(args.episode, f"ep{args.episode}")]
    else:
        scan = gym.make("ReacherGoal-v0")
        # A fresh ClonePolicy per episode: it carries the (u, y) window across
        # steps, so reusing one across scenarios leaks the previous episode's past.
        rows = {tag: [run_episode(scan, ClonePolicy(pred), q, g) for q, g in eps]
                for _lab, tag, pred in arms}
        scan.close()
        bc, dg = rows["bc"], rows["dagger"]
        n = len(eps)
        print(f"scanned {n}: BC {sum(r['reached'] for r in bc)}/{n}   "
              f"DAgger {sum(r['reached'] for r in dg)}/{n}")

        picks = []
        resc = [i for i in range(n) if dg[i]["reached"] and not bc[i]["reached"]]
        regr = [i for i in range(n) if bc[i]["reached"] and not dg[i]["reached"]]
        both = [i for i in range(n) if dg[i]["reached"] and bc[i]["reached"]]
        nei = [i for i in range(n) if not dg[i]["reached"] and not bc[i]["reached"]]
        def pick(idxs, key, why):
            i = max(idxs, key=key)
            picks.append((i, f"{why}_ep{i}"))

        if resc:
            pick(resc, lambda i: bc[i]["final"], "rescue")
        drift = lambda i: bc[i]["final"] - bc[i]["best"]  # noqa: E731
        if both:
            pick(both, drift, "both")
        pick(range(n), drift, "bc_drift")
        if regr:
            pick(regr, lambda i: dg[i]["final"], "regression")
        if nei:
            pick(nei, lambda i: dg[i]["final"], "neither")
        print(f"  {len(resc)} DAgger rescues, {len(regr)} regressions {regr[:10]}, "
              f"{len(both)} both, {len(nei)} neither")
        # The widest-drift pick often IS the both-reach pick; rendering it twice
        # writes two identical clips under different names.
        seen = set()
        picks = [(i, why) for i, why in picks
                 if not (i in seen or seen.add(i))]

    os.makedirs(args.out_dir, exist_ok=True)
    env = gym.make("ReacherGoal-v0", render_mode="rgb_array", render_size=args.size)
    for idx, why in picks:
        q, g = eps[idx]
        out = []
        for label, tag, pred in arms:
            frames, summ = record(env, ClonePolicy(pred), q, g, label, args.tol)
            path = os.path.join(args.out_dir, f"{why}_{tag}.mp4")
            encode_video(frames, path, args.fps)
            out.append((label, summ, path))
        print(f"\nscenario #{idx} ({why}), need {out[0][1]['need'] * 1e3:.0f} mm")
        for label, s, path in out:
            print(f"  {label:<24} best {s['best'] * 1e3:5.1f}mm  "
                  f"final {s['final'] * 1e3:5.1f}mm  -> {path}")
    env.close()


if __name__ == "__main__":
    main()

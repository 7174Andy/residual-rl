"""One-off: upload this week's reacher runs to Weights & Biases.

Each training run becomes a W&B run: the Monitor CSV supplies the per-episode
return curve (x = cumulative env steps), and the matching checkpoint-sweep CSV
(when one exists) is attached as a table. Requires `wandb login` first; set
WANDB_MODE=offline to smoke-test without an account (sync later).

    uv run python scripts/backfill_wandb.py --project two-wheel-exp
"""
from __future__ import annotations

import argparse
import csv
import os

D = "data/reacher_ckpt_seeds"

# (run name, monitor csv, sweep csv or None, sweep arm filter, tags)
MANIFEST = [
    *[(f"res_f1_s{i}", f"{D}/res_s{i}.monitor.csv",
       f"{D}/reacher_crossover_s{i}.csv", "residual",
       ["reacher", "residual", "frac1.0", f"seed{i}"]) for i in (1, 2, 3, 4)],
    *[(f"van_s{i}", f"{D}/van_s{i}.monitor.csv",
       f"{D}/reacher_crossover_s{i}.csv", "vanilla",
       ["reacher", "vanilla", f"seed{i}"]) for i in (1, 2, 3, 4)],
    *[(f"res_f2_s{i}", f"{D}/resf2_s{i}.monitor.csv",
       f"{D}/resf2_s{i}.csv", "residual_f2",
       ["reacher", "residual", "frac2.0", f"seed{i}"]) for i in range(5)],
    ("res_f1_s0", "data/reacher_residual_400k.monitor.csv", None, None,
     ["reacher", "residual", "frac1.0", "seed0"]),
    ("van_s0", "data/reacher_vanilla_400k.monitor.csv",
     "docs/reference/reacher_crossover.csv", "vanilla",
     ["reacher", "vanilla", "seed0"]),
]

# Unicycle: 5-seed TD3 sweep (frac1/frac2/vanilla) + the SAC 2x2 cells.
# Their deployed-checkpoint metrics live in ONE csv (checkpoint_sweep_all.csv,
# 4 arms x 5 seeds x 20 ckpts), uploaded once as a summary-table run.
UNICYCLE = [
    *[(f"uni_res_f1_s{i}", f"data/seedsweep/res_f1_s{i}_mon.monitor.csv",
       ["unicycle", "residual", "td3", "frac1.0", f"seed{i}"])
      for i in range(5)],
    *[(f"uni_res_f2_s{i}", f"data/seedsweep/res_f2_s{i}_mon.monitor.csv",
       ["unicycle", "residual", "td3", "frac2.0", f"seed{i}"])
      for i in range(5)],
    *[(f"uni_van_td3_s{i}", f"data/seedsweep/van_s{i}_mon.monitor.csv",
       ["unicycle", "vanilla", "td3", f"seed{i}"]) for i in range(5)],
    *[(f"uni_res_sac_s{i}", f"data/sacsweep/res_s{i}_mon.monitor.csv",
       ["unicycle", "residual", "sac", "frac2.0", f"seed{i}"])
      for i in range(5)],
    *[(f"uni_van_sac_s{i}", f"data/sacsweep_van/van_s{i}_mon.monitor.csv",
       ["unicycle", "vanilla", "sac", f"seed{i}"]) for i in range(5)],
]
UNICYCLE_SWEEP_CSV = "data/checkpoint_sweep_all.csv"


def episodes(monitor_csv):
    with open(monitor_csv) as f:
        next(f)  # json meta line
        yield from csv.DictReader(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="two-wheel-exp")
    p.add_argument("--systems", nargs="+", default=["reacher", "unicycle"],
                   choices=["reacher", "unicycle"],
                   help="which backfill sections to upload")
    args = p.parse_args()

    import wandb

    def upload_monitor(run, mon, prefix):
        steps = 0
        for ep in episodes(mon):
            steps += int(float(ep["l"]))
            run.log({f"{prefix}/episode_return": float(ep["r"]),
                     f"{prefix}/episode_length": int(float(ep["l"]))},
                    step=steps)
        return steps

    if "unicycle" in args.systems:
        for name, mon, tags in UNICYCLE:
            if not os.path.exists(mon):
                print(f"skip {name}: {mon} missing")
                continue
            run = wandb.init(project=args.project, name=name, tags=tags,
                             group="unicycle-sweeps-backfill", reinit=True,
                             config={"source": "backfill", "monitor": mon})
            steps = upload_monitor(run, mon, "unicycle")
            run.finish()
            print(f"backfilled {name}: {steps} steps")
        if os.path.exists(UNICYCLE_SWEEP_CSV):
            with open(UNICYCLE_SWEEP_CSV) as f:
                r = csv.reader(f)
                cols = next(r)
                rows = list(r)
            run = wandb.init(project=args.project, name="uni_checkpoint_sweep_all",
                             tags=["unicycle", "sweep-summary"],
                             group="unicycle-sweeps-backfill", reinit=True,
                             config={"source": UNICYCLE_SWEEP_CSV})
            run.log({"checkpoint_sweep_all": wandb.Table(columns=cols, data=rows)})
            run.finish()
            print(f"backfilled uni_checkpoint_sweep_all: {len(rows)} rows")

    if "reacher" not in args.systems:
        return
    for name, mon, sweep, arm, tags in MANIFEST:
        if not os.path.exists(mon):
            print(f"skip {name}: {mon} missing")
            continue
        run = wandb.init(project=args.project, name=name, tags=tags,
                         group="reacher-5seed-backfill", reinit=True,
                         config={"source": "backfill", "monitor": mon})
        steps = upload_monitor(run, mon, "reacher")
        if sweep and os.path.exists(sweep):
            with open(sweep) as f:
                next(f)
                rows = [[r["steps"], r["reached"], r["n"], r["best_med_mm"],
                         r["final_med_mm"]]
                        for r in csv.DictReader(f) if r["arm"] == arm]
            run.log({"checkpoint_sweep": wandb.Table(
                columns=["steps", "reached", "n", "best_med_mm",
                         "final_med_mm"], data=rows)})
        run.finish()
        print(f"backfilled {name}: {steps} steps")


if __name__ == "__main__":
    main()

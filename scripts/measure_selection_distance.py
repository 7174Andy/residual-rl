"""How far away is the data Select-DPC selects? The measurement the argument was missing.

Journey 11 established a ~0.5 rad validity radius and measured the Panda's median
distance to the nearest *anchor* at 1.98 rad. Select-DPC does not use anchors -- it
picks Hankel *columns* out of a pooled bank, and the collection trajectories wander
from their anchors under OU excitation. So "1.98 rad from an anchor" does not by
itself say how far the selected data is, and the whole coverage argument rests on
that number being large. This script measures it directly.

For each held-out configuration, drawn the way each env draws its episode starts:

  d_sample  distance to the nearest COLLECTED SAMPLE, anywhere in the collection.
            The absolute floor: no selection rule, no Hankel windowing and no
            stride can put data closer than this.
  d_anchor  distance to the nearest anchor          (the journey-11 number)
  d_bank    distance to the nearest column in the ENTIRE bank -- the floor for a
            selector that must pick whole length-L windows
  d_sel     distance to the columns Algorithm 2 actually picks (median over them)
  valid%    share of picked columns inside the 0.5 rad validity radius

Column -> configuration via the bank's `origin`/`t0`: column j starts its past
window at sample `t0[j]` of trajectory `origin[j]`, whose configuration is known.
A column is a length-L window rather than a point -- on the Panda it traverses a
median 0.58 rad -- so `d_bank` locates it by its first sample and `d_sample`
exists to give the same argument a bound that carries no windowing convention at
all. Distance is joint-space (wrapped on Reacher's periodic joint0), the same
metric journey 11's validity curve is parameterised by.

    uv run python scripts/measure_selection_distance.py --system both
    uv run python scripts/measure_selection_distance.py --system panda --n 40
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from core.selectdpc import select_predict  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, CRITICAL, GOOD = "#0b0b0b", "#52514e", "#b8b7b2", "#d03b3b", "#0ca30c"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "lines.linewidth": 2.0,
})

VALID_RAD = 0.5          # journey 11 / 12: skill > 0.7 inside, < 0 beyond ~1 rad

# The validity curves both panels are read against (journey 12, "The validity
# radius is ~0.5 rad on BOTH arms"). Transcribed, not re-measured here.
RAD_P = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
SKILL_PANDA = [0.93, 0.88, 0.72, 0.14, -9.93, -23.64]
RAD_R = [0.0, 0.25, 0.5, 1.0, 2.0]
SKILL_REACH = [0.94, 0.91, 0.84, -0.02, -6.06]


def column_configs(payload: dict, bank: dict) -> tuple[np.ndarray, np.ndarray]:
    """`(per-column window-start configs, every collected config)`."""
    q = [payload[f"q_{i}"] for i in range(bank["n_traj"])]
    cols = np.array([q[i][t] for i, t in zip(bank["origin"], bank["t0"])])
    return cols, np.vstack(q)


def score(yh: np.ndarray, y_true: np.ndarray, y_last: np.ndarray, nq: int) -> tuple:
    """`verify_libraries.py`'s tip skill and direction cosine, on the tip block only.

    skill compares against a "the tip does not move" predictor, so skill < 0 means
    the prediction is worse than assuming the arm is frozen.
    """
    tp, tt, st = yh[:, nq:], y_true[:, nq:], y_last[nq:]
    mse_l = np.mean(np.sum((tp - tt) ** 2, axis=1))
    mse_0 = np.mean(np.sum((np.tile(st, (len(tt), 1)) - tt) ** 2, axis=1))
    dp, dt = tp[-1] - st, tt[-1] - st
    den = np.linalg.norm(dp) * np.linalg.norm(dt)
    return (float(1.0 - mse_l / max(mse_0, 1e-15)),
            float(dp @ dt / den) if den > 1e-12 else 0.0)


def run_panda(args) -> dict:
    from panda.model import load_model, safe_box, sample_config, tip_id
    from panda.qdes import collect_anchor, outputs
    from panda.selectdpc import panda_bank

    model, data = load_model(servo_scale=args.servo_scale)
    with np.load(args.libs) as z:
        payload = {k: z[k] for k in z.files}
    bank = panda_bank(payload, args.T_ini, args.N, stride=args.stride)
    anchors, nq = payload["anchors"], model.nq
    Qcol, Qall = column_configs(payload, bank)
    lo, hi = safe_box(model)
    tip = tip_id(model)
    rng = np.random.default_rng(args.seed)

    def dist(q, Q):
        return np.linalg.norm(Q - q, axis=1)

    rows = []
    for _ in range(args.n):
        q0, _ = sample_config(model, data, rng, lo, hi, tip)
        rec = collect_anchor(model, data, q0, args.T_ini + args.N + 1, rng,
                             sigma=args.sigma)
        rows.append(_one(q0, rec, outputs(rec["q"], rec["tip"]), bank, Qcol, Qall,
                         anchors, nq, dist, args))
    return {"name": "Panda (7-DoF)", "rows": rows, "n_col": bank["Up"].shape[1],
            "tau_dim": bank["tau"].shape[0], "n_anchor": len(anchors),
            "n_sample": len(Qall), "valid_rad": args.valid_rad}


def run_reacher(args) -> dict:
    from reacher.deepc_setup import anchor_grid, collect_anchor, outputs
    from reacher.model import NQ_ARM, config_distance, load_model, sample_config
    from reacher.selectdpc import trajectory_bank

    model, data = load_model()
    anchors = anchor_grid(model, *args.grid)
    rng = np.random.default_rng(args.seed)
    payload = {"anchors": anchors}
    for i, a in enumerate(anchors):
        rec = collect_anchor(model, data, a, args.T, rng)
        payload[f"u_{i}"], payload[f"q_{i}"], payload[f"tip_{i}"] = (
            rec["u"], rec["q"], rec["tip"])
    bank = trajectory_bank(payload, args.T_ini, args.N, stride=args.stride)
    Qcol, Qall = column_configs(payload, bank)

    rows = []
    for _ in range(args.n):
        q0, _ = sample_config(model, data, rng)
        rec = collect_anchor(model, data, q0, args.T_ini + args.N + 1, rng)
        rows.append(_one(q0, rec, outputs(rec["q"], rec["tip"]), bank, Qcol, Qall,
                         anchors, NQ_ARM, config_distance, args))
    return {"name": "Reacher (2-DoF)", "rows": rows, "valid_rad": args.valid_rad,
            "n_col": bank["Up"].shape[1],
            "tau_dim": bank["tau"].shape[0], "n_anchor": len(anchors),
            "n_sample": len(Qall)}


def _one(q0, rec, y, bank, Qcol, Qall, anchors, nq, dist, args) -> dict:
    """One held-out configuration: run the selection rule, then measure it."""
    T_ini, N = args.T_ini, args.N
    p_y = y.shape[1]
    u_ini, y_ini = rec["u"][:T_ini], y[:T_ini]
    u_f, y_true = rec["u"][T_ini:T_ini + N], y[T_ini:T_ini + N]
    yh, sel = select_predict(bank, u_ini, y_ini, u_f, args.n_cols, args.lambda_g,
                             N, p_y)
    skill, cos = score(yh, y_true, y[T_ini - 1], nq)
    d_all = dist(q0, Qcol)
    d_sel = d_all[sel]
    return {"d_sample": float(dist(q0, Qall).min()),
            "d_anchor": float(dist(q0, anchors).min()),
            "d_bank": float(d_all.min()),
            "d_sel_med": float(np.median(d_sel)),
            "d_sel_min": float(d_sel.min()),
            "valid_frac": float(np.mean(d_sel < args.valid_rad)),
            "n_traj": int(np.unique(bank["origin"][sel]).size),
            "skill": skill, "cos": cos}


def summarise(res: dict) -> None:
    r = res["rows"]

    def med(k):
        return float(np.median([x[k] for x in r]))

    print(f"\n{res['name']}  --  {res['n_col']} columns from {res['n_anchor']} "
          f"trajectories ({res['n_sample']} samples), tau dim {res['tau_dim']}")
    print(f"  {'nearest anchor':>22}{med('d_anchor'):>9.2f} rad")
    print(f"  {'nearest sample':>22}{med('d_sample'):>9.2f} rad   "
          f"<- the floor: nothing in the collection is closer")
    print(f"  {'nearest bank column':>22}{med('d_bank'):>9.2f} rad   "
          f"<- the floor for whole-window selection")
    print(f"  {'selected (median)':>22}{med('d_sel_med'):>9.2f} rad")
    print(f"  {'selected (closest)':>22}{med('d_sel_min'):>9.2f} rad")
    label = f"inside {res['valid_rad']:.2f} rad"
    print(f"  {label:>22}{100 * med('valid_frac'):>8.0f}%    "
          f"of the selected columns")
    print(f"  {'skill':>22}{med('skill'):>9.2f}      cos {med('cos'):>5.2f}"
          f"   ({sum(x['skill'] < 0 for x in r)}/{len(r)} negative)")


def figure(results: list, out: str) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 4.1))
    colours = {"Panda (7-DoF)": SERIES[1], "Reacher (2-DoF)": SERIES[0]}

    # A: the selected distance against the validity curve it cannot move ------
    ax[0].axhline(0, color=CRITICAL, lw=1.2, ls="--", zorder=1)
    ax[0].axvspan(0, VALID_RAD, color="#e8f3e8", zorder=0)
    ax[0].plot(RAD_P, SKILL_PANDA, color=SERIES[1], lw=1.4, ls=":", zorder=2,
               label="Panda validity curve")
    ax[0].plot(RAD_R, SKILL_REACH, color=SERIES[0], lw=1.4, ls=":", zorder=2,
               label="Reacher validity curve")
    for res in results:
        c = colours[res["name"]]
        ax[0].scatter([x["d_sel_med"] for x in res["rows"]],
                      [x["skill"] for x in res["rows"]], s=26, color=c,
                      edgecolors="#fcfcfb", linewidths=0.8, zorder=3,
                      label=f"{res['name']} Select-DPC")
    ax[0].set_ylim(-12, 2.5)
    ax[0].text(0.24, 1.6, "usable", ha="center", fontsize=8, color=GOOD)
    ax[0].set_xlabel("distance to the SELECTED data (rad)")
    ax[0].set_ylabel("prediction skill")
    ax[0].set_title("A · Selection moves you along\nthe curve, not off it",
                    loc="left", color=INK)
    ax[0].legend(frameon=False, fontsize=7.5, loc="lower left")

    # B: the floor -- nearest column anywhere in the bank --------------------
    for res in results:
        c = colours[res["name"]]
        d = np.sort([x["d_sample"] for x in res["rows"]])
        ax[1].step(d, np.arange(1, len(d) + 1) / len(d), where="post", color=c,
                   zorder=3, label=res["name"])
        d2 = np.sort([x["d_sel_med"] for x in res["rows"]])
        ax[1].step(d2, np.arange(1, len(d2) + 1) / len(d2), where="post", color=c,
                   lw=1.2, ls="--", zorder=3)
    ax[1].axvline(VALID_RAD, color=CRITICAL, lw=1.2, ls="--", zorder=2)
    ax[1].text(VALID_RAD * 1.08, 0.06, "validity radius", color=CRITICAL, fontsize=8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("distance (rad)   solid: nearest sample collected\n"
                     "dashed: median selected column")
    ax[1].set_ylabel("fraction of held-out configurations")
    ax[1].set_title("B · Even the BEST possible pick\nis outside on the Panda",
                    loc="left", color=INK)
    ax[1].legend(frameon=False, fontsize=8, loc="upper left")

    # C: share of the selection that is inside the validity radius -----------
    y = np.arange(len(results))
    frac = [100 * float(np.median([x["valid_frac"] for x in r["rows"]]))
            for r in results]
    ax[2].barh(y, frac, 0.5, color=[colours[r["name"]] for r in results], zorder=3)
    for i, (r, f) in enumerate(zip(results, frac)):
        skill = float(np.median([x["skill"] for x in r["rows"]]))
        ax[2].annotate(f"{f:.0f}%   median skill {skill:+.2f}", xy=(f, i),
                       xytext=(9, 0), textcoords="offset points", va="center",
                       fontsize=8, color=CRITICAL if f < 1 else INK2)
        if f < 1:
            ax[2].plot([0], [i], marker="|", ms=16, mew=3,
                       color=colours[r["name"]], zorder=4)
    ax[2].set_yticks(y)
    ax[2].set_yticklabels([r["name"] for r in results], fontsize=8)
    ax[2].invert_yaxis()
    ax[2].set_xlim(0, 135)
    ax[2].set_xlabel("selected columns inside 0.5 rad (%)")
    ax[2].set_title("C · What the algorithm had\nto choose from", loc="left",
                    color=INK)

    for a in ax:
        a.grid(True, zorder=0)
        a.set_axisbelow(True)
    fig.suptitle("Select-DPC selects; it does not collect — the 7-DoF bank has "
                 "nothing near enough to pick",
                 x=0.005, ha="left", color=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"\nwrote {out}")

    csv = out.replace(".png", ".csv")
    keys = ("d_sample", "d_anchor", "d_bank", "d_sel_med", "d_sel_min",
            "valid_frac", "n_traj", "skill", "cos")
    with open(csv, "w") as fh:
        fh.write("# scripts/measure_selection_distance.py\n")
        fh.write("system," + ",".join(keys) + "\n")
        for res in results:
            for x in res["rows"]:
                fh.write(res["name"].split()[0] + ","
                         + ",".join(f"{x[k]:.4f}" for k in keys) + "\n")
    print(f"wrote {csv}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", default="both", choices=["panda", "reacher", "both"])
    p.add_argument("--libs", default="data/panda_uniform_libs.npz",
                   help="Panda bank; uniform anchors match how episodes start")
    p.add_argument("--grid", type=int, nargs=2, default=[6, 5],
                   help="Reacher anchor grid, as run_select_dpc_reacher.py uses")
    p.add_argument("--T", type=int, default=1200, help="Reacher samples per anchor")
    p.add_argument("--n", type=int, default=40, help="held-out configurations")
    p.add_argument("--n-cols", type=int, default=300, help="the paper's N_cols")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--T-ini", type=int, default=5)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--sigma", type=float, default=0.25, help="Panda excitation")
    p.add_argument("--lambda-g", type=float, default=5e-3)
    p.add_argument("--out", default="docs/reference/panda_selection_distance.png")
    p.add_argument("--servo-scale", type=float, default=1.0,
                   help="Panda PD servo gain multiplier; MUST match the gains "
                        "--libs was collected at")
    p.add_argument("--valid-rad", type=float, default=VALID_RAD,
                   help="validity radius the valid%% column is scored against; "
                        "0.5 is the stiff-servo number, ~1.05 the retuned one")
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    results = []
    if args.system in ("panda", "both"):
        results.append(run_panda(args))
        summarise(results[-1])
    if args.system in ("reacher", "both"):
        results.append(run_reacher(args))
        summarise(results[-1])
    figure(results, args.out)


if __name__ == "__main__":
    main()

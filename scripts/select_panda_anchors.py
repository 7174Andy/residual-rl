"""Stage 1 of the anchor-selection plan: pick anchors from task-relevant configs.

Covers plan sections 2-5 (offline steps 1-7): sample the goals the task draws
from, take the configuration that reaches each, cluster them, and check that the
resulting anchors are genuinely different arm shapes rather than one shape
rotated about the base.

`--ik` selects plan section 3's inverse-kinematics stage, and it is the single
most consequential flag here: it decides which slice of the 7-D configuration
space the robot is asked to live in, and therefore whether there is any structure
for k-medoids to find. `--ik none` keeps the FK-sampled configuration and yields a
uniform box; `--ik random` is the null control that should reproduce it.

    uv run python scripts/select_panda_anchors.py --K 4 --ik home
    uv run python scripts/select_panda_anchors.py --K 8 --samples 2000 --ik previous
"""
from __future__ import annotations

import argparse

import numpy as np

from panda.anchors import sample_task_configs, select_anchors
from panda.model import load_model

DEFAULT_OUT = "data/panda_anchors_k4.npz"


def report(res: dict, Q: np.ndarray) -> None:
    k = len(res["anchors"])
    div = res["diversity"]
    print(f"\nk-medoids over {len(Q)} task configurations, K={k}")
    print(f"  total within-cluster distance: {res['cost']:.2f} rad")
    print(f"\n  {'region':>7}{'members':>9}{'radius (rad)':>15}   anchor q")
    for j in range(k):
        print(f"  {j:>7}{res['counts'][j]:>9}{res['radius'][j]:>15.3f}   "
              f"{np.round(res['anchors'][j], 3)}")

    ev = res["pca"]["explained"]
    print(f"\n  silhouette (full 7-D):  {res['silhouette']:.3f}")
    print(f"  PCA variance per PC:    {np.round(ev, 3)}")
    print(f"  PC1+PC2 capture {np.cumsum(ev)[1] * 100:.0f}% -- read the 2-D scatter "
          "with that in mind")
    # Participation ratio of the variance spectrum: 1 if one direction carries
    # everything, 7 if the cloud is perfectly isotropic.
    print(f"  effective dimensionality: {1 / (ev**2).sum():.2f} / 7")
    if res["silhouette"] < 0.15:
        print("  NOTE: silhouette near 0 -- the clusters touch. k-medoids partitioned a\n"
              "        continuum rather than finding separated groups, so anchor identity\n"
              "        is a tiling choice, not a discovered structure.")

    print("\nPlan section 5 -- are these different arm shapes?")
    print(f"  per-joint std across anchors: {np.round(div['per_joint_std'], 3)}")
    print(f"  effective joints varying:     {div['effective_joints']:.2f} / 7")
    print(f"  share of variance in joint 1: {div['q1_variance_share'] * 100:.1f}%")
    if div["effective_joints"] < 2.0:
        print("  WARNING: anchors lie along ~one direction -- the plan's section-5\n"
              "           failure case. They are near-duplicates as local models.")
    if div["q1_variance_share"] > 0.5:
        print("  WARNING: most spread is in joint 1, whose axis is gravity-aligned.\n"
              "           The arm's joint-space dynamics are EXACTLY invariant to it\n"
              "           (measured 6.7e-16 m), so that spread buys no model fidelity.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--K", type=int, default=4, help="number of anchors (plan starts at 4)")
    p.add_argument("--samples", type=int, default=1000,
                   help="task goals to sample (plan suggests 500-2000)")
    p.add_argument("--weights", type=float, nargs=7, default=None,
                   help="per-joint weights w_1..w_7 for the distance metric (section 7)")
    p.add_argument("--ik", choices=["none", "home", "previous", "random"],
                   default="none",
                   help="how to choose the configuration for each goal (plan section 3); "
                        "'none' keeps the FK-sampled one, 'random' is the null control")
    p.add_argument("--n-init", type=int, default=10, help="k-medoids restarts")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, data = load_model()
    rng = np.random.default_rng(args.seed)
    w = None if args.weights is None else np.asarray(args.weights, dtype=np.float64)

    print(f"sampling {args.samples} task-relevant (config, goal) pairs, "
          f"ik={args.ik} ...")
    Q, P = sample_task_configs(model, data, rng, args.samples,
                               ik=None if args.ik == "none" else args.ik)
    print(f"  tip radius over the goal set: "
          f"[{np.linalg.norm(P, axis=1).min():.3f}, {np.linalg.norm(P, axis=1).max():.3f}] m")

    res = select_anchors(Q, args.K, rng, w=w, n_init=args.n_init)
    report(res, Q)

    np.savez(args.out, anchors=res["anchors"], Q=Q, P=P, labels=res["labels"],
             medoid_idx=res["medoid_idx"], counts=res["counts"],
             radius=res["radius"], seed=args.seed,
             weights=np.array([] if w is None else w), ik=args.ik)
    print(f"\nwrote {args.out}  (anchors, Q_task, goals, labels)")
    print(f"next: uv run python scripts/collect_anchor_libraries.py --anchors {args.out}")


if __name__ == "__main__":
    main()

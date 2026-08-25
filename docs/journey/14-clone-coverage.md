# 14. Clone coverage, measured — why BC "worked" on the unicycle

## Decision

The unicycle never avoided the covariate-shift failure that forced DAgger on
the Reacher ([13](13-reacher-residual.md)) — its hybrid clone dataset
**pre-empted** it. Measured by ablation: retrain the same clone on each slice
of `data/clone_dataset.npz` alone, and the reacher's failure signature
reproduces almost exactly. Consequence for the Panda: **DAgger (or an
equivalent hybrid collection) from round zero, no plain-BC attempt.**

## Context

Journey 13 diagnosed the reacher clone's failure as distribution shift:
disagreement with the expert was 0.1025 at expert-visited states against
0.2815 at the clone's own — **2.75x** — and DAgger, which changes only the
data distribution, fixed the gate. That left an obvious question with no
measurement behind it: why did plain supervised cloning pass the gate on the
unicycle ([07](07-imitation-learning.md))?

A correction first, because the natural reading of journey 07's table is
wrong: the unicycle dataset's `onpolicy` slice (42.9%) is **expert-driven**
data — canonical DeePC rolling its own closed loop — i.e. exactly the kind of
data the reacher's failing BC clone was trained on. It is *not* clone-queried
DAgger data. So "the unicycle had on-policy data" cannot by itself explain
the escape; the dataset's other 57% (random-pose synthetic + degenerate
stall states) had to be doing work. Whether it actually was is the
measurement below.

## The measurement

`scripts/measure_clone_disagreement.py` rolls the same 15 held-out seeds
(base 4104626029) two ways and reports the median `||u_clone − u_expert||`
under each distribution — the exact protocol of journey 13's 2.75x. The
ablations retrain the identical architecture on single slices (the dataset's
`regime` column tags every row); closed-loop gates are
`scripts/validate_clone.py`, 78 seeds. Figure:
`docs/reference/clone_coverage.png` (`scripts/plot_clone_coverage.py`).

| clone trained on | in-dist fit | at expert states | at own states | ratio | closed loop, 78 seeds |
| --- | --- | --- | --- | --- | --- |
| expert rollouts only (15,023) | **best** (MAE v 0.355) | 0.64 | 1.71 | **2.68x** | **16/78 — worse than DeePC's 30/78, McNemar p = 0.0056** |
| hybrid, all slices (shipped) | mid | 1.06 | 1.34 | 1.27x | 30/78 = expert parity, high agreement (journey 07's gate) |
| synthetic + degenerate only (20,000) | worst (val MSE 0.116) | 2.35 | 1.69 | 0.72x | 33/78 reach — but agreement 0.705, traj dev median 1.583 |

Three readings, one per row:

1. **Give the unicycle reacher-style data and it fails the reacher's way.**
   Expert-rollout-only is the best fit in-distribution and the worst
   controller: its error blows up 2.68x on its own induced states (the
   reacher measured 2.75x), and the closed loop collapses to roughly half
   the expert's reach rate. The failure follows the **data**, not the
   system.
2. **The hybrid's escape is real but partial.** 1.27x is not 1.0x — the
   shift exists, merely small enough that the closed loop tolerates it.
   Coverage bought tolerance, not immunity.
3. **Breadth without the expert's distribution is not coverage either.**
   The synthetic-only clone is uniformly bad (its *training* distribution —
   random-action pasts — matches neither rollout distribution), yet it
   *reaches* 33/78. Marginal reach parity while disagreeing with the expert
   on 23/78 seeds and deviating 1.6 units in trajectory: an accidental
   controller, and exactly why journey 07's gate demands paired agreement
   and trajectory fidelity, never reach rate alone.

## Considered

- **"Geometric coverage" as the metric.** Rejected, per journey 13's own
  retraction: the reacher's clone-visited states sat only 1.13x further from
  the training set in feature space while the error grew 2.75x. Density is
  the wrong test; disagreement under the deployed distribution is the right
  one, and it is what this entry measures.
- **Explaining the escape structurally** (smoother expert, coarser
  tolerance, 3-D fully-observed state). Falsified as the *sufficient*
  explanation by row 1: with expert-only data the unicycle fails anyway.
  Structure sets the size of the tolerance band (1.27x passed here; the
  reacher's knife-edge would likely not have passed it), but the binding
  variable is the data.

## Outcome

- `scripts/measure_clone_disagreement.py`, `scripts/plot_clone_coverage.py`,
  `docs/reference/clone_coverage.png`; raw distances in
  `data/disagree_{hybrid,onpolicyonly,noonpolicy}.npz`, ablation clones and
  filtered datasets in `data/clone_{onpolicyonly,noonpolicy}.pt` /
  `data/clone_dataset_{onpolicyonly,noonpolicy}.npz`.
- Journey 07's dataset design is retroactively vindicated as the load-bearing
  choice, and its "on-policy" naming clarified (expert-driven, not
  clone-driven).
- For the Panda: the expert is slower (~190 ms/QP) and the state is
  higher-dimensional, so post-hoc rescue is expensive — collect on-policy
  (DAgger) or hybrid from the start, and give the clone the full `q` in its
  input (tip-only input makes imitation one-to-many, journey 11's
  self-motion measurement).

## Caveats

- 15 episodes per distribution (~2,200 states each) for the disagreement
  medians; single training run per ablation cell. The effect sizes (2.68x,
  30→16 with p = 0.0056) dwarf that noise, but the exact ratios should not
  be quoted past two digits.
- The disagreement metric mixes `v` (range 20) and `w` (range π) in one
  Euclidean norm, like journey 13 mixed its torque channels; ratios are
  unit-free, absolute values are not comparable across systems.
- The synthetic-only row's 33/78 reach uses the same broadened action bounds
  as every arm; nothing suggests it generalizes — its trajectory deviation
  says it is not tracking the expert even when it reaches.

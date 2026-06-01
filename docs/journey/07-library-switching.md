# 07. Library switching — local linearization via 4 quadrants

## Decision

**Pre-collect four data libraries, one per heading quadrant, and hand all four to a single `DeePC` controller.** Each step, the controller selects the library whose anchor heading is closest to the robot's current heading, swaps its Hankel matrices into one cached QP, and solves. The robot keeps one controller; only the *data library* feeding the predictor changes.

## Context

The previous entry [Why one library isn't enough](06-single-library-fails.md) explained the structural problem: the unicycle's bilinear `cos(δ)·v` term cannot be represented by a single linear behavioral predictor across all orientations. The fix is piecewise linearization — fit a separate local model in each "operating region" and route the controller through whichever region you're in.

The paper does exactly this. From [arXiv:2603.07395](https://arxiv.org/abs/2603.07395) Appendix D:

> *"For each prediction horizon W, we construct 4 corresponding data libraries Hd with Tini = 5, each approximately representing the local behavior of the nonlinear system within orientation intervals [0, π/2), [π/2, π), [π, 3π/2), and [3π/2, 2π). During online tracking, the data library associated with the robot's current orientation is selected to improve tracking performance."*

So four libraries cover the heading circle, each anchored at the midpoint of its quadrant.

## Why local linearization works

Within a fixed heading region — say `δ ≈ π/4` — both `cos(δ)` and `sin(δ)` are nearly constant (`≈ 0.71` each). The dynamics behave locally like:

$$
\Delta x \approx \Delta t \cdot 0.71 \cdot v, \qquad \Delta y \approx \Delta t \cdot 0.71 \cdot v
$$

This is **linear in `v`** with fixed slope. A linear behavioral predictor trained on data collected while the robot was near heading `π/4` will correctly capture this slope and produce confident, non-degenerate predictions.

By collecting one library *starting from each of four orientations* (the paper's `π/4, 3π/4, 5π/4, 7π/4`), each library's data is concentrated in a roughly π/2-wide heading region around its anchor. At runtime we always have **the right library for the current heading**.

## Considered

1. **One library** — failed structurally (previous entry).
2. **Many libraries** — diminishing returns past 4; the local linearity holds well over π/2-wide quadrants. 4 matches the paper.
3. **Continuous gain scheduling** — interpolate between libraries based on heading. More work; the paper's discrete switch is enough.
4. **Lift the state via Koopman (`sin δ`, `cos δ` in `y`)** — alternative path that would let one library suffice. Bigger refactor (env's `y` becomes dim 4, `Q` becomes 4×4). Sidelined for now in favor of switching.

## Implementation

Switching lives **inside `DeePC`** — there is no separate wrapper class. One controller holds:

- A list of N pre-collected libraries (each a `(Up, Uf, Yp, Yf)` Hankel tuple) plus their anchor headings.
- The active library's Hankels in `cp.Parameter`s, so swapping libraries means writing new parameter values into **one compiled QP** — no recompile per library.
- **One shared past-`(u, y)` buffer** — the robot's actual history is the same regardless of which library predicts the future.

On each `act(y_current, y_ref)`:

1. `idx = argmin_i |wrap(y_current[heading_index] − anchor_i)|` (trivially 0 with a single library).
2. If `idx` changed since last step, clear the warm-start `g` — its columns indexed the old library and are meaningless under the new one.
3. Write `libraries[idx]`'s `(Up, Uf, Yp, Yf)` into the Hankel parameters and solve.
4. Slide the shared buffer with the applied `(u_t, y_current)`.

The four libraries come from offline rollouts started at the paper's four init headings (`controllers/data_collection.PAPER_INIT_HEADINGS`); each rollout's data concentrates around one quadrant anchor. `scripts/run_deepc.py` builds all four by default (`--single_library N` passes a one-element list for the single-library contrast) and prints **per-episode library usage** so you can see when switching actually triggered:

```text
episode 1: REACHED   after  94 steps  final_dist=0.46
  library usage: [0, 0, 17, 77]    # Q3 → Q4 mid-flight
```

## Outcome

Closed-loop test on broad-`w` data with library switching enabled (all four libraries loaded). Reproduce with `uv run python scripts/run_deepc.py --headless --episodes 4 --seed 0` (data: `data/libraries_v0.npz`, `T_ini=5`, `N=12`):

| Episode | Outcome | Steps | Final dist |
|---|---|---|---|
| 0 | truncated | 200 | 16.95 |
| 1 | **REACHED** | 94 | **0.46** |
| 2 | **REACHED** | 35 | **0.46** |
| 3 | **REACHED** | 115 | **0.48** |

Success rate **3/4** on this seed (mean steps-to-reach 81). This is the first time the DeePC controller actually navigated the robot to the goal. Compare to single-library across the same seeds, where `v` was stuck at zero in all four episodes.

Seed 0 is a lucky draw, though. Over a larger random sample (`--episodes 100 --random`, base seed `4104626029`, n=78 before the run was stopped) the aggregate is more sobering:

- **Reach rate ≈ 39%** (30/78); the rest truncate at 200 steps.
- **QP failures: 0/78** — the solver did not fail once across the whole sample (nor on seed 0).
- Steps-to-reach (successes): mean ≈ 103, range 7–174. Final distance: mean ≈ 4.0, max ≈ 15.8.

So switching makes goal-reaching *possible* (~39% vs single-library's 0%), but it is far from reliable — the dominant failure is the truncation/spin mode below, not the QP.

## Caveats

- Switching only helps if the robot **actually crosses quadrant boundaries** during an episode. With narrow `w` data the turn rate is too low and switching becomes a no-op.
- **Truncation/spin is the dominant failure** (~61% of random episodes). Episode 0 above is typical: 200 steps, ends 16.95 from goal despite cycling all four libraries — `w` saturates positive while `v` stays near zero, so the robot spins in place rather than driving out. Diagnosing these would benefit from logging the QP's `σ_y` and the controller's predicted vs actual trajectory.
- **QP solver failures did not appear** in this measurement (0 across 82 episodes: 78 random + 4 at seed 0), so they are not the bottleneck here — the earlier "occasional `user_limit` failure" concern is not borne out on this build/data. The QP is still ill-conditioned at the paper's `λ_y ≈ 3e6` (see `deepc.py`), so a numerical breakdown remains *possible* on other data/solvers (SCS is chosen for robustness; CLARABEL can break down), but it is not what limits reaching.

## Code

`two_wheel_robot/controllers/deepc.py::DeePC` — library switching is built into the controller (`libraries`, `anchor_headings`, `heading_index`; `last_library_idx` diagnostic). See [library switching reference](../controllers/library-switching.md).

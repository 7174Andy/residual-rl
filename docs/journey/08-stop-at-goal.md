# 08. Stopping at the goal — the deceleration / overshoot problem

!!! note "Status: partially resolved (2026-05-29)"
The original **overshoot** problem below is resolved by widening the data
envelope (see _Resolution_). Doing so surfaced a **new** failure — _over-braking_
near the goal — whose proposed fix (RL + DeePC) is still open. Read top-to-bottom
as a sequence: overshoot → fix → new problem → next direction.

## Problem (one line)

With forward-only speed bounds (`v_min > 0`), the controller drives the robot to
the goal's neighborhood but **cannot decelerate to land inside the tolerance** —
it skims past the goal circle and misses by a small margin, then loops back.

## Context

After [03 action bounds](03-action-bounds.md) and the bounds sweep, `data/libraries.npz`
is collected with **hybrid** PE bounds `v ∈ [10, 20]`, `w ∈ [±π/2]` (paper-style
forward-only `v`, but a fast turn rate). This removes the spin-in-place stall from
[06](06-single-library-fails.md): forcing `v ≥ 10` makes "do nothing" infeasible.

But the paper's task is continuous heart-curve _tracking_ — the robot never stops.
Ours is point-to-point _reaching_, which requires stopping within `goal_tolerance`
(`0.5`). Forcing `v ≥ 10` means the robot moves a minimum of `v_min · Δt = 10 · 0.025
= 0.25` units **every** step and can never slow down. On a tangential approach it
steps _over_ the 0.5-radius circle instead of landing in it.

## What it looks like

Closed-loop run on the hybrid data (6 seeds, `Q_heading=2`, bearing reference):

| seeds                                                                      | result |
| -------------------------------------------------------------------------- | ------ |
| reach the goal                                                             | 2 / 6  |
| **near-miss overshoot** (reached `min_dist` 0.68 / 0.88, then sailed past) | 2 / 6  |
| never converge                                                             | 2 / 6  |

Two observed symptoms, one root cause:

1. **`v` barely changes; only `w` does** — with `v_min=10` and a tiny control weight
   (`R = 1.3e-3`), cruising near minimum speed and steering with `w` is cost-optimal.
   Not a bug; it's the _cause_ of the next symptom.
2. **Misses the goal by ~1 cm** — the constant ~0.25 units/step stride can't be
   shortened, so the robot overshoots the tolerance circle.

## Resolution — widen the data envelope to `v ∈ [0, 20]`

The overshoot is a **stabilizability** fact, not a tuning issue. For the goal `p*`
to be an equilibrium we need `p_{t+1} = p_t`, i.e. `Δt · v_t = 0 ⇒ v_t = 0`. With
`v_min = 10 > 0`, **no admissible input makes the goal a fixed point** — the robot
is a Dubins vehicle with a forward-speed floor and physically cannot stop. Worse,
its minimum turning radius is `r_min = v_min / w_max = 10 / (π/2) = 20/π ≈ 6.4`
units — ~13× the `0.5` tolerance — so a near-miss can only be corrected by a wide
loop, not a tight retry.

Fix: recollect with the floor removed. `data/libraries_v0.npz` uses
`v ∈ [0, 20]`, `w ∈ [±π/2]` (T=1500, 4 paper headings, seed 42), so the data span
now contains stopping/slow motion and DeePC can represent deceleration.

Closed-loop on the new data (**seed 42, 3 episodes**):

| metric              | old `v ∈ [10, 20]` | new `v ∈ [0, 20]`        |
| ------------------- | ------------------ | ------------------------ |
| success rate        | 1 / 3              | **2 / 3**                |
| `v` min observed    | +10.0 (floored)    | **+0.0 — it brakes**     |
| `v` mean            | ~11.5              | 3.2 – 7.4                |
| worst `final_dist`  | 8 – 20 (spirals)   | **2.75 (lingers)**       |

With `v_min = 0` the goal is a genuine equilibrium, the controller brakes, and a
miss now _lingers near_ the goal instead of spiraling across the box.

<figure markdown>
  <video controls loop muted playsinline width="480">
    <source src="../videos/successful-landing.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <figcaption>
    Successful reach (<code>libraries_v0.npz</code>, seed 42). The robot now
    decelerates on approach and lands inside the 0.5 tolerance (final_dist 0.42,
    129 steps) — the deceleration the old <code>v ∈ [10, 20]</code> data could not
    represent. Reproduce: <code>uv run python scripts/run_deepc.py --libraries
    data/libraries_v0.npz --seed 42 --episodes 1 --record docs/journey/videos</code>.
  </figcaption>
</figure>

## New problem — over-braking / pirouette at the goal

The remaining failure (seed 42, ep 1: truncated at `final_dist = 2.75`, `w`
saturated near `+1.1`) is the **opposite** of overshoot: the robot now slows down
_too much_ near the goal and gets stuck in a near-stationary spin.

<figure markdown>
  <video controls loop muted playsinline width="480">
    <source src="../videos/over-braking-pirouette.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <figcaption>
    Over-braking failure (<code>libraries_v0.npz</code>, seed 43). The robot
    decelerates near the goal and locks into a stationary pirouette, never closing
    the last ~2.8 units. Reproduce: <code>uv run python scripts/run_deepc.py
    --libraries data/libraries_v0.npz --seed 43 --episodes 1
    --record docs/journey/videos</code>.
  </figcaption>
</figure>

This is an **objective mismatch**, and `R` is not the cause — the position-vs-effort
trade-off prefers speed `v* ≈ Q·d·Δt·N / R ≈ 230·d`, which saturates `v_max` at
`d = 2.75`, i.e. it _wants_ to push hard. The brake comes from the **heading term**:

1. **Bearing reference is ill-conditioned as `d → 0`.** With `y_ref` heading =
   `atan2(g_y−y, g_x−x)`, its position-sensitivity is `‖∇ atan2‖ = 1/d` — the
   heading target _spins_ near the goal.
2. **Heading is the heaviest-weighted coordinate** (`Q = diag(1,1,2)`), so the QP
   spends effort chasing that spinning target instead of translating in.
3. **The position pull vanishes** as `2Q·d → 0`. Heading screams (`∝ 1/d`) just as
   drive-in goes quiet (`∝ d`) → optimum tips to _rotate, don't translate_ → `v→0`.
   Stationary robot ⇒ `d` fixed ⇒ bearing keeps demanding rotation ⇒ **pirouette**.
4. **No terminal incentive in the QP.** The env's `reach_bonus = 100` is _not_ in
   the controller's objective, and `N = 12` (~3–6 units lookahead) is too myopic to
   value committing to land. A low-effort stop-and-turn is a valid QP solution.

Note termination is **position-only** (heading is irrelevant to "reached"), yet
the controller is penalized for heading error the task doesn't reward — so the
heading tracking near the goal is actively counterproductive.

A cheap, untested config fix exists: `--Q_heading 0` or `--no_bearing_ref` (or
gate heading off inside a small radius). Try this _before_ reaching for RL.

## Direction — combine RL with DeePC

For a robust _general_ policy (and if the config fix plateaus), warm-start RL from
DeePC. RL optimizes the **true** env reward (including `reach_bonus`), closing the
exact gap that causes the stall. Two candidate architectures:

- **Residual RL** — `u = u_DeePC + u_RL(obs)`, residual zero-initialized so initial
  behavior ≈ DeePC; RL learns a near-goal correction. Conceptually clean, but keeps
  the QP _in the training loop_ (a solve per step → expensive).
- **BC warm-start → fine-tune** — clone DeePC into a fast neural policy, then
  PPO/SAC on env reward. Avoids QP-in-loop during RL, but risks _importing the
  expert's flaw_ (cloning DeePC near the goal clones the stall) — must down-weight
  near-goal demos. The repo's body-frame observation is already designed for this.

(Scaffolding note: no `rl/` dir and `stable_baselines3` not yet installed — this is
greenfield. Item 3 of the repo plan.)

## Open questions

**1. What metrics beyond success rate?** Success rate alone hides _how_ a run fails
(overshoot vs stall vs diverge) and how efficient the successes are. Candidates:

| group       | metric                                    | catches                          |
| ----------- | ----------------------------------------- | -------------------------------- |
| success     | success rate; success rate 95% CI         | headline + variance across seeds |
| proximity   | **min distance** ever reached; final dist | near-miss (0.68/0.88) vs stall (2.75) |
| efficiency  | steps-to-reach; **path-length ratio** (actual ÷ straight-line, optimal = 1) | looping / pirouetting |
| effort      | Σ‖u‖² (energy); Σ‖Δu‖ (smoothness/jerk)    | chattering, wasted control       |
| diagnostic  | `w`-saturation fraction; mean `v` in final approach | spin-stall vs overshoot |
| settling    | dwell time inside tolerance               | can-it-stay (≈0 if `v_min>0`)    |
| failure tax | overshoot / stall / diverge / wall-pin %  | _why_ it failed, not just that   |
| compute     | QP failure rate; solve time; warm-start hit rate | controller health           |

`path-length ratio` and `min distance` are the highest-value additions: together
they distinguish all three observed failure modes that success rate collapses.

**2. Will residual RL improve the current controller?** _Hypothesis: yes for the
near-goal landing and general robustness_ — the residual is ~0 in the far field
(DeePC already navigates well), small to learn, starts at the DeePC baseline (no
regression risk), and is trained on the true reward, so it directly targets the
terminal-maneuver gap. _Genuine uncertainties_: (a) QP-in-the-loop training cost
may be prohibitive; (b) the reach bonus is sparse — may need reward shaping; (c) if
the trivial `--Q_heading 0` fix already removes the stall, the _marginal_ gain of
residual RL over the config fix could be small. **The real question is not "does RL
help?" but "does residual RL beat the much cheaper heading-reference fix?"** — which
requires an ablation: `DeePC + heading-fix` vs `DeePC + residual-RL` vs both,
scored on the metrics above.

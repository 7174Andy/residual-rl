# 07. Stopping at the goal — the deceleration / overshoot problem

Overshoot → fixed by widening the data envelope → two residual failures surface: a
near-goal pirouette and, dominant, a far-field `v`-collapse. Reach rate over random
seeds is **~39%**, and the collapse traces to a hallucinating predictor, not missing data.

## Problem (one line)

With forward-only speed bounds (`v_min > 0`), the controller drives the robot to
the goal's neighborhood but **cannot decelerate to land inside the tolerance** —
it skims past the goal circle and misses by a small margin, then loops back.

## Context

After the action-bounds sweep, `data/libraries.npz`
is collected with **hybrid** PE bounds `v ∈ [10, 20]`, `w ∈ [±π/2]` (paper-style
forward-only `v`, but a fast turn rate). This removes the spin-in-place stall from
[05](05-library-switching.md): forcing `v ≥ 10` makes "do nothing" infeasible.

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

<figure markdown>
  <video controls loop muted playsinline width="480">
    <source src="../videos/cannot-stop-overshoot.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <figcaption>
    Can't-stop overshoot (<code>libraries.npz</code>, `v ∈ [10, 20]`, seed 42).
    `v` stays floored at 10 (never brakes), so the robot skims the goal and orbits
    at the ~6.4-unit minimum turning radius without landing. Reproduce:
    <code>uv run python scripts/run_deepc.py --libraries data/libraries.npz
    --seed 42 --episodes 3 --record docs/journey/videos</code> (episode 1).
  </figcaption>
</figure>

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

| metric             | old `v ∈ [10, 20]` | new `v ∈ [0, 20]`    |
| ------------------ | ------------------ | -------------------- |
| success rate       | 1 / 3              | **2 / 3**            |
| `v` min observed   | +10.0 (floored)    | **+0.0 — it brakes** |
| `v` mean           | ~11.5              | 3.2 – 7.4            |
| worst `final_dist` | 8 – 20 (spirals)   | **2.75 (lingers)**   |

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

## Larger-sample reality check (2026-06-01)

The numbers above ("2 / 3", "a miss now lingers near the goal at 2.75") come from
**n = 3 on seed 42** — optimistic. A broader random sweep (`scripts/run_deepc.py
--episodes 100 --random`, base seed `4104626029`, n = 78 before the slow run was
stopped; `libraries_v0.npz`) reframes the picture:

- **Reach rate ≈ 39 %** (30 / 78); seed 0's 3 / 4 is a lucky draw.
- **QP failures: 0 / 78** — the controller is numerically healthy, so the solver is
  _not_ the bottleneck (the "compute" metric row below can be deprioritized).
- The 48 truncated episodes end a **median of 5.38 units from the goal** (max 15.77),
  with median mean-`v` ≈ **1.66** (successes run `v̄` ≈ 4–7).

That splits the residual failure into **two populations**, not the single near-goal
mode the section above describes:

| population                       | count (of 48 truncations) | signature                                             | matches                 |
| -------------------------------- | ------------------------- | ----------------------------------------------------- | ----------------------- |
| near-goal pirouette / over-brake | 15 (final_dist < 3.0)     | reaches the neighborhood, then stalls / spins         | the "New problem" above |
| **far-field `v`-collapse**       | **33 (final_dist ≥ 3.0)** | never drives in; `v̄` stays ≈ 1.6 across the whole run | **dominant; new**       |

So the Resolution's "miss now lingers near the goal" claim holds only for a minority
of seeds. On most failing seeds the robot **never commits to forward motion** and
truncates far away — the same `v → 0` family as
[05](05-library-switching.md), now seed-dependent rather than universal.

<figure markdown>
  <video controls loop muted playsinline width="480">
    <source src="../videos/far-field-v-collapse-stall.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <figcaption>
    Far-field `v`-collapse — pure stall (`libraries_v0.npz`, seed 4104626047).
    `v` never leaves ~0 (`v̄ = 0.00`), so the robot sits near the start while the
    goal is 14.6 units away; it truncates at `final_dist 14.61` having made no net
    progress. Nothing to do with stopping _at_ the goal — it never drives toward it.
    Reproduce: <code>uv run python scripts/run_deepc.py --seed 4104626047
    --episodes 1 --record docs/journey/videos</code>.
  </figcaption>
</figure>

<figure markdown>
  <video controls loop muted playsinline width="480">
    <source src="../videos/far-field-v-collapse-worst.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <figcaption>
    Far-field `v`-collapse — worst case (`libraries_v0.npz`, seed 4104626064).
    The robot drifts with only token forward speed (`v̄ = 0.71`) and never converges,
    ending the farthest of the whole sweep at `final_dist 15.77`. Reproduce:
    <code>uv run python scripts/run_deepc.py --seed 4104626064 --episodes 1
    --record docs/journey/videos</code>.
  </figcaption>
</figure>

## Root cause of the `v`-collapse — a hallucinated prediction from a degenerate past, not missing data (and not a bug)

A per-step trace settles the attribution — and it is **not** what "library quality"
first suggests. `scripts/run_deepc.py --seed 4104626047 --episodes 1 --headless
--trace` logs, each step, the QP's own predicted distance-reduction and a
**forced-forward counterfactual** (`DeePC.diagnose_forward`: re-solve with the first
step's `v` pinned `≥ 12`). On the pure-stall seed every one of the 200 steps reads
identically:

```text
step  0  d=14.61  lib=1  v=0.00  w=-1.57  σy=0.0 | plan Δd=+13.91 | forced(v≥) v=12.0 Δd=+13.92 σy=0.0 [optimal]
…
median applied v = 0.00 | median plan Δd = +13.87 | median forced Δd = +13.88 (200/200 forced probes optimal)
```

Read it carefully — it rules out the data hypothesis:

1. **The data _can_ drive in.** The forced-`v≥12` probe solves cleanly every step
   (`optimal`, `σy = 0`) and predicts approaching the goal (`Δd ≈ +13.9`). The
   selected library is _not_ missing the trajectories — pre-collection coverage is
   **not** the cause.
2. **The predictor hallucinates.** The QP's _own chosen_ plan — the one that applies
   `v = 0` this step — **also** predicts reaching the goal (`plan Δd ≈ +13.9`). It
   believes it can get ~14 units closer while doing nothing now.
3. **So the QP takes the lazy plan.** With `v = 0` and `v = 12` predicted to be
   _equally good_, the effort term `R·v²` breaks the tie toward `v = 0`. Receding
   horizon applies `v = 0`, the real robot doesn't move, the next solve faces the
   identical state and re-hallucinates the identical "I'll reach it" plan → the robot
   sits frozen at `d = 14.61` for all 200 steps.

**Why the prediction lies — and why it is _not_ a bug.** For a nonlinear system,
`Yf·g` and `Uf·g` from the same `g` need not be a _real_ trajectory: the `g` that
matches "reach the goal" in the predicted **output** can be paired with an **input**
whose first action is `v = 0`. But nonlinearity alone is not the whole story — an
in-distribution test pins it down. Feeding each library a **real** recent past plus
the **real** future inputs from held-out data, the predictor reproduces the true
future positions to **~0.01–0.05 units one step ahead** (3 of 4 libraries). So the
Hankel construction and `u`/`y` alignment are **correct**, and the predictor is
_accurate when the past is informative_ — despite the same nonlinearity. The
closed-loop hallucination (~8.6 units off at step 1) is ~200× larger than any error
the predictor makes on real data, so it is **not an implementation error**.

The hallucination is triggered by a **degenerate past**, not nonlinearity per se:

- At `reset()` the past buffer is primed with `T_ini` copies of one frozen pose — not
  a real trajectory.
- Once the robot stalls, its _real_ recent history is also near-constant (fixed
  position, spinning heading).

A (near-)constant past is matched by **many** `g`'s — the constraints `Up·g = u_ini`,
`Yp·g = y_ini` no longer pin down the future — so the QP is free to pick a blend whose
predicted future lands on the goal while its first control is `v = 0`. It is
**self-trapping**: frozen init → stall → constant history → sustains the stall. The
precise cause is the _conjunction_, the bilinear unfaithfulness of
[05](05-library-switching.md) **gated on past-conditioning**:

> **nonlinear dynamics** (matching the past no longer guarantees a real future)
> **+ an uninformative/constant past** (so many blends satisfy the match)
> → the QP picks a blend that predicts reaching the goal but whose first control is
> `v = 0`.

Nonlinearity alone would not do it — on a real, varied past the predictor is accurate.

**Consequences for the fixes:**

- **Reweighting `Q`/`R`/`λ` will not reliably fix it.** Lowering `R` only removes the
  tie-breaker; the QP is then _indifferent_ between `v = 0` and `v = 12` (both falsely
  predicted to reach) and can still stall. The rot is in the prediction, which no cost
  reweighting repairs. (Consistent with [05](05-library-switching.md)'s λ sweep.)
- **`--Q_heading 0` / `--no_bearing_ref`** still targets only the _near-goal
  pirouette_ population (~15 episodes), not this dominant collapse.
- **Attack the trigger (cheap — now the leading lever).** Because the predictor is
  accurate on an _informative_ past, keeping the past informative should help: anchor
  the prediction to the current measurement (the buffer currently lags one step), seed
  the past with a short real motion instead of a frozen pose, or force a few forward
  steps at episode start before handing to DeePC. Untested.
- **Attack the root (model class).** A _faithful_ predictor — a Koopman lift putting
  `sin δ`, `cos δ` into `y` — makes the linear span hold only physically realizable
  trajectories, so it cannot pair "reach goal" with "do nothing now." A terminal
  constraint / move-blocking is a cheaper partial guard.
- **RL on the true env reward is immune** — it learns from _real_ transitions and is
  never fooled by the hallucinated predictor (next section).

**Aside — library 1 is poorly conditioned.** The failing seed selected `lib = 1`
(heading band ≈ `3π/4`), and that library predicts notably worse even in-distribution
(horizon position RMSE ≈ 3.5 vs ≈ 0.1–0.5 for the others) — a separate data-quality
issue worth its own look.

## Direction — combine RL with DeePC

DeePC is **model-free**: its prediction _is_ the data representation
`[Up;Yp;Uf;Yf]·g = [uini;yini;u;y]` (DeeP-LCC, [arXiv:2203.10639](https://arxiv.org/abs/2203.10639)
Eq. 21 / Prop. 2), with **no physics prior to flag a wrong one**. Willems' fundamental
lemma makes that representation exact for _linear_ systems with informative data — but
this robot is nonlinear, so on a degenerate past the representation fails silently and
flows straight to the actuator. That is the limit of "truly based on data": **garbage
representation in → garbage control out, unchecked.**

RL optimizes the **true** env reward from **real transitions**, so it cannot be fooled
by a hallucinated representation — it is grounded exactly where DeePC's data-blend is
not. That makes the hybrid a _backstop_, not polish: DeePC supplies the prior on the
~39 % of seeds where its representation is valid; RL carries the ~42 % where it
collapses.

Followed up in [07](07-imitation-learning.md) — clone DeePC into a fast MLP, which also
takes the ~0.5 s/step QP out of the training loop — and [08](08-residual-rl.md), a TD3
residual on top: **39 % → 87–95 %** reach rate.

## Open question — metrics beyond success rate

Success rate alone hides _how_ a run fails (overshoot vs stall vs diverge) and how
efficient the successes are. Candidates:

| group       | metric                                                                      | catches                               |
| ----------- | --------------------------------------------------------------------------- | ------------------------------------- |
| success     | success rate; success rate 95% CI                                           | headline + variance across seeds      |
| proximity   | **min distance** ever reached; final dist                                   | near-miss (0.68/0.88) vs stall (2.75) |
| efficiency  | steps-to-reach; **path-length ratio** (actual ÷ straight-line, optimal = 1) | looping / pirouetting                 |
| effort      | Σ‖u‖² (energy); Σ‖Δu‖ (smoothness/jerk)                                     | chattering, wasted control            |
| diagnostic  | `w`-saturation fraction; mean `v` in final approach                         | spin-stall vs overshoot               |
| settling    | dwell time inside tolerance                                                 | can-it-stay (≈0 if `v_min>0`)         |
| failure tax | overshoot / stall / diverge / wall-pin %                                    | _why_ it failed, not just that        |
| compute     | QP failure rate; solve time; warm-start hit rate                            | controller health                     |

`path-length ratio` and `min distance` are the highest-value additions: together
they distinguish all three observed failure modes that success rate collapses. Still
open — the benchmarks in [07](07-imitation-learning.md) / [08](08-residual-rl.md)
report reach rate only.

# 13. Residual RL on Reacher — DAgger fixes the clone, and vanilla still ties

## Decision

Port the unicycle's `DeePC → clone → residual` pipeline
([07](07-imitation-learning.md), [08](08-residual-rl.md), [09](09-vanilla-rl.md))
to `Reacher-v5`. Adds **`ReacherGoal-v0`** (`reacher/env.py`), a shared **`rl/`**
package for the system-agnostic RL machinery, and the clone/residual stack in
`reacher/`.

The pipeline works, after one substantive fix: **plain behavioral cloning cannot
reproduce a DeePC-family controller here, and on-policy imitation (DAgger) can.**
Seven other diagnoses were proposed and retracted before that one; all are recorded
below rather than dropped.

And the control still ties it. **Vanilla RL, with no DeePC anywhere in it, matches
the full pipeline at 113/120 and is more precise on every distance metric.**
Journey 09 found the same on the unicycle. That is now two systems.

## The result

120 held-out scenarios (`--seed 90000`, disjoint from every training stream), full
horizon, early stopping **off**, both RL rows at **200k steps**
(`scripts/eval_reacher_residual.py`):

| controller | reach rate (95% CI) | best | final | best→final | path/net |
| --- | --- | --- | --- | --- | --- |
| Select-DPC (expert) | 89/120 [66–81%] | 2.9 mm | 6.4 mm | 2.2x | 1.6 |
| DAgger clone | 82/120 [60–76%] | 5.1 mm | 12.3 mm | 2.4x | 1.7 |
| clone + warm start | 83/120 [60–77%] | 6.1 mm | 11.8 mm | 1.9x | 1.8 |
| **clone + residual** | **113/120 [88–97%]** | 3.1 mm | 5.4 mm | 1.8x | 1.6 |
| **vanilla RL** | **113/120 [88–97%]** | **1.9 mm** | **3.2 mm** | **1.7x** | **1.4** |
| random torque | 3/120 [1–7%] | 56.5 mm | 181.7 mm | 3.2x | 7.6 |

| paired vs the clone | rescues | regressions | closer on | McNemar |
| --- | --- | --- | --- | --- |
| clone + residual | 36 | 5 | 79/120 | `p < 0.001` |
| vanilla RL | 38 | 7 | 87/120 | `p < 0.001` |

Three things follow.

1. **The pipeline works.** The residual lifts its baseline 82 → 113/120 and
   **beats the expert it was cloned from** (89/120). Journey 08's arc reproduces
   on a structurally different system.
2. **The drift goal is met** — the one thing [journey 12](12-select-dpc.md) said
   would lift every row rather than trade one against another. `best→final` is
   1.8x for the residual and 1.7x for vanilla, against the expert's 2.2x and
   journey 12's 2.1–2.3x band. A policy paid on every step of a full horizon holds
   position better than a receding-horizon QP with no terminal term.
3. **And vanilla ties it exactly while being more precise** — 1.9 mm against
   3.1 mm `best`, 3.2 against 5.4 mm `final`, `path/net` 1.4 against 1.6.

The honest reading: on this task the pipeline is **sound and unnecessary**. Its one
surviving advantage is the zero-init floor — a residual starts at its baseline's
competence where vanilla starts at random — which matters when a bad early policy
is expensive, and not at all when 200k steps costs nine minutes.

## How DAgger works in this pipeline

This is the fix that made the clone usable, so it is worth being precise about.

### The problem it solves

Behavioral cloning collects its dataset by running the **expert** and recording
`(state, expert action)` pairs. The network therefore becomes accurate on *states
the expert visits*. But at deployment the **clone** drives: its small errors carry
it to slightly different states, where there was no training data, so it is worse
there, which carries it further off. Error compounds — `T² · ε` in the horizon,
against DAgger's `T · ε` (Ross, Gordon & Bagnell 2011).

Measured directly here, rolling the same start states two ways:

| clone-vs-expert disagreement, measured at | median |
| --- | --- |
| states the **expert** drives to | 0.1025 |
| states the **clone** drives to | **0.2815** |

**2.75x.** That is the whole diagnosis — and every dataset collected before DAgger
ran the expert and labelled with the expert, so the clone's own states appeared in
training data exactly never.

### The loop

Each round separates *who drives* from *who labels*:

| round | drives the episode | supplies the labels |
| --- | --- | --- |
| 0 | expert | expert — this is plain BC |
| 1 | the clone | **expert, queried at the clone's states** |
| 2 | the clone (retrained) | expert again |
| 3 | … | … |

`reacher/clone_data.py::dagger_rollout` is the whole idea in four lines:

```python
u_label = expert.act(y_pre, y_ref)   # expert answers "what should you do HERE?"
acts.append(clip(u_label))           # -> the training target
u_apply = policy(env, info)          # the CLONE decides where we actually go
env.step(u_apply)
```

The dataset only grows — `scripts/run_dagger.py` aggregates rather than replaces,
hence "Dataset Aggregation" — so later rounds keep the expert's distribution while
adding the student's.

### The DeePC-specific trap

`DeePC.act` slides its past buffer with **the action it computed**
(`core/deepc.py:86-89`). Under DAgger that action is *not* what the plant
received, so from step 2 onward the expert would be answering questions about a
trajectory that never happened — and nothing would raise. `dagger_rollout`
therefore overwrites `expert._u_buf[-1]` with the **applied** action after every
step. Without it the labels are quietly wrong.

### What it bought

3 rounds × 100 episodes, expert = memoryless Select-DPC, gate on 40 scenarios:

| stage | rows | reached | reach drop | best | gate |
| --- | --- | --- | --- | --- | --- |
| expert | — | 30/40 | — | 4.2 mm | — |
| behavioral cloning | 10,000 | 22/40 | +20.0 pts | 9.5 mm | **FAIL** |
| DAgger round 1 | 15,000 | 19/40 | +27.5 pts | 10.8 mm | FAIL |
| DAgger round 2 | 20,000 | 26/40 | +10.0 pts | 7.0 mm | **PASS** |
| DAgger round 3 | 25,000 | **27/40** | **+7.5 pts** | **5.2 mm** | **PASS** |

The clone's error on the states it chose for itself fell **0.2815 → 0.2158 →
0.1454**. Round 1 got *worse* first: 5,000 rows from a poor policy dilute 10,000 BC
rows before they correct them.

A 3-episode smoke test — **150 rows on top of 10,000** — already moved the gate
from FAIL to PASS. 1.5% more data, collected in the right place, did what tripling
the expert-driven data could not.

### The metric that misled, and the one that didn't

Across those rounds, open-loop error **on the expert's distribution** went
0.1215 → 0.1281 — slightly *worse* — while closed-loop performance improved
sharply. That is the mirror image of the one-hot experiment below, where
representation improved 2x and the closed loop degraded.

Both say the same thing: **fit measured on the expert's states does not predict
behaviour on the student's.** Every earlier intervention — more data, more
capacity, richer features, a library one-hot — optimised the first while the 2.75x
was pointing at the second.

## Context

Journey 12 left Select-DPC working on Reacher and named the drift as the one
limitation whose fix would lift every row. Select-DPC costs **76.7 ms/step**
against a 20 ms budget at 50 Hz, so it is 4x too slow to deploy *and* cannot sit
inside an RL loop; the clone exists to amortize it into a forward pass.

## Considered

### `ReacherGoal-v0` departs from stock `Reacher-v5` in three ways

- **Goals are rejection-sampled against the reachable annulus.** At
  `SAFE_MARGIN = 0.02` the fingertip attains `[0.0291, 0.21]` m against a 0.20 m
  disc, so **2.1%** of stock draws are impossible and score as controller failure.
- **The episode never terminates on reach**, so "arrive and hold" and "arrive and
  leave" are distinguishable — the whole point of measuring drift.
- **Dense per-step distance, not squared.** At a 10 mm tolerance on a 210 mm
  workspace the squared term is ~1e-4 and its gradient vanishes as the tip
  approaches.

`reach_bonus` is **1.0**, not the unicycle's 100: without termination the bonus is
paid on every held step, so it *is* the station-keeping reward, and at 100 it would
swamp the distance term ~1000x.

## Outcome — what was tried before DAgger

All of the following failed, and each excluded something.

### Select-DPC is recurrent; making it memoryless does not help

`SelectDPC.act` selects against `self._tau_prev`, the previous step's prediction,
so its action depends on the whole episode history. From identical
`(u_ini, y_ini, y_current, goal)`, clearing that state moves the action by a median
**0.2442 — 0.81x the action's own magnitude**, up to the full torque box.

`core/selectdpc.py` gained `carry_prediction=False` to remove it. As a *controller*
that costs almost nothing — **30/40 against 31/40**, paired coin-flip — a design
trade the Select-DPC paper does not discuss and worth knowing independently. As a
*fix for cloning* it did nothing: the gate still failed at 20.0 points.

### The library one-hot: better representation, worse closed loop

The unicycle clone that worked was handed a 4-way library one-hot (`data/clone.pt`
ships `n_lib = 4`, those columns protected from standardization). `ReacherDeePC`
picks by nearest anchor, so a 30-way one-hot is the faithful port — and it had
never been run.

| clone | val MSE | val RMSE / std | open-loop vs a constant predictor |
| --- | --- | --- | --- |
| Select-DPC (carried) | 0.394 | 0.631 | 23% |
| Select-DPC (memoryless) | 0.343 | 0.586 | 29% |
| **fixed + one-hot** | **0.228** | **0.478** | **45%** |

Best representation by a wide margin — and the closed loop got *worse*, 25/40 →
15/40. Not mode chattering either: anchor switches per episode were 92 for the
clone against 84 for the base, a coin flip.

### Neither encoding horn works

| encoding | dim | NN distance | NN label diff | val error |
| --- | --- | --- | --- | --- |
| DeePC window | 43 | 1.78 | 0.1466 | 0.0962 |
| state `(cos/sin q0, q1, q̇, tip−goal)` | 7 | **0.43** | **0.1492** | 0.0919 |

Compressing to 7-D made the data **4x denser** and moved the neighbour label
difference not at all. And the labels are not noise: re-solving is deterministic
(median 0.0000), tightening SCS to `eps 1e-9` moves the action 0.0041, a 1e-6 state
perturbation moves it 0.0001.

### The exclusion table

| candidate | test | result |
| --- | --- | --- |
| overfitting | train vs val | 0.0875 vs 0.0918 — no gap |
| capacity | 1024x3 vs 256x256 | **worse** |
| recurrent base | `carry_prediction=False` | no change |
| cold start | warm-start hybrid | +2 reaches |
| sample count | 200 → 600 episodes | step-0 error moved 0.0002 |
| missing velocity | explicit `q̇` | 4% |
| library one-hot | 30-way, faithful port | representation 2x better, gate worse |
| **on-policy data** | **DAgger** | **fixes it** |

Every row but the last is an attempt to make the clone better **at the expert's
states**. That was never the problem.

## Retractions

### The evaluation-set leak

The first version of this entry reported the residual at 27/40 with 13 rescues.
**All of it was measured on the training set.** `make_reacher_scenarios.py` drew
goal-then-config from `default_rng(0)`; `ReacherGoalEnv.reset` draws the same order
from `self.np_random` seeded 0; SB3 seeds once and auto-resets **unseeded**.
Verified: evaluation scenario *i* was training episode *i*, identity mapping,
across all 40.

Nothing in the suite could see it — shapes, reachability and `need` were all
correct. Only provenance was wrong. Fixed by moving the generator to a disjoint
seed namespace, recording the seed in the npz, and adding
`test_scenarios_are_held_out_from_the_training_stream`, which walks the seed-0
stream exactly as SB3 does and asserts no frozen scenario appears in it.

### Eight diagnoses of the clone failure

1. **Selection discontinuity.** Real (error 3.8x higher on churning steps,
   `corr(error, action jump) = +0.73`) but not binding — removing the churn made
   the gate *worse*.
2. **Compounding error, first attempt.** The measurement was invalid: it
   referenced training rows against their own temporal neighbours, which measures
   adjacency within an episode, not distributional support.
3. **Under-sampled cold start.** Refuted by a pre-registered prediction: 3x the
   starts moved step-0 error by 0.0002.
4. **Noise floor at step 0.** Withdrawn — a deterministic target has none, and the
   comparison was a pairwise difference against a single-label deviation.
5. **Carried recurrent state.** The 0.2442 contrast is real but overstates what
   the data contains; the memoryless clone fit no better.
6. **Feature-space sparsity.** Refuted — 4x denser encoding, identical label
   spread.
7. **Mode chattering under the one-hot.** Refuted — 1.10x switch ratio.
8. **A reward misaligned with the reach criterion.** Proposed when the residual
   *harmed* the clone at 50k (82 → 65, `p = 0.037`). Refuted by the 200k control:
   it was **under-training**, and at 200k the residual improves the clone
   (82 → 113, `p < 0.001`).

Retraction 2 deserves a note. The *correct* version of that measurement — the
2.75x on-policy/off-policy gap — was in hand early. It was dismissed because the
feature-space *distance* ratio was only 1.13x. Distance was the wrong test; error
on the policy's own distribution was the right one, and it had already been
measured.

### Other withdrawn claims

- **Two claims of "reproducing" journey 12.** 159/200 is 79.5%, not 80.0%, and
  `scripts/eval_reacher_scenarios.py:131` applies no reachability filter, so ~2.1%
  of journey 12's goals were impossible. Consistent with, not identical to.
- **"The drift is not the QP's fault."** At 50k, vanilla drifted 1.9x with no QP
  anywhere, which looked like it falsified journey 12's mechanism. At 200k both RL
  rows beat the expert's drift (1.7–1.8x against 2.2x), which is what journey 12
  predicted a fix would look like. The 50k reading is withdrawn.
- **A test that could never fail.** `test_never_terminates_on_reach` set the goal
  60 mm from the held tip against a 10 mm tolerance, so `reached` never fired.
  Fixed to 5 mm with an `ever_reached` assertion, verified by injecting the
  regression it was blind to.

## Caveats

- **One seed per policy**, one `residual_frac`, one `reach_bonus`. None swept.
- **The DAgger clone imitates the *memoryless* expert**, so the `Select-DPC` row is
  built with `carry_prediction=False` to match. Comparing it against the carried
  variant would score the clone against a controller it never imitated.
- Wilson intervals on the two 113/120 rows overlap almost entirely; the paired
  McNemar tests carry the claims, not the point estimates.
- `path/net` for the `random` row is a median over 118, not 120: two scenarios made
  no progress, so `eff` is NaN there by design.
- **The invariant tests skip on a clean checkout.** `data/` is gitignored, so the
  bit-for-bit zero-residual test and both zero-init tests `pytest.skip` elsewhere
  and the suite still reports green.
- Observation normalization in `ResidualSelectEnv` is min-max against declared
  bounds and the live channels are compressed ~18x relative to the widest; untested
  as a factor now that the residual works.

## Next

1. **Sweep DAgger rounds and episodes-per-round.** Three rounds of 100 was the
   first thing tried and it worked; the knee is unmeasured, and a 3-episode smoke
   test already moved the gate.
2. **Re-run the residual on the one-hot fixed-anchor clone.** DAgger and the
   one-hot were never combined — the one-hot gave the best representation, DAgger
   fixes the distribution, and they address different halves of the problem.
3. **Decide whether the pipeline earns its complexity.** Two systems now show
   vanilla RL matching it. The zero-init floor is the remaining argument, and it is
   worth measuring rather than asserting: compare early-training reach rates, where
   a residual should be far ahead.
4. **The Panda still needs on-policy collection** ([12](12-select-dpc.md) item 4).
   `rl/` is now the shared package that would serve it, and DAgger is exactly the
   on-policy mechanism that entry asked for.

# 13. Residual RL on Reacher — DAgger fixes the clone, 400k settles the verdict

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

And the control still wins. The first evaluation stopped both RL arms at 200k
steps, where their training curves had visibly not plateaued, so both were
retrained to **400k** (~20 minutes each) and re-evaluated. At the plateau:
**vanilla RL, with no DeePC anywhere in it, reaches 120/120 against the full
pipeline's 119/120 and is the more precise arm (1.7 vs 2.4 mm final).**
Journey 09 found the same on the unicycle. That is now two systems, measured
at convergence — the pipeline is **sound and unnecessary** on this task. Its
surviving advantage is the head start, now measured on deployed checkpoints:
the residual dominates vanilla at every budget through ~100k steps and the
5-seed mean crossing sits between 125k and 150k — but the "zero-init floor"
is **not a floor** (see the crossover section).

All residual numbers in this entry are the **`--residual-frac 2.0`** retrain
(2026-08-24). The original frac-1.0 runs were re-collected after a 5-seed
control showed the narrower authority cost the residual reach and precision
on every seed (400k pooled: 578/600 at 3.13 mm against frac 2.0's 589/600 at
2.51 mm) — the unicycle's dead-zone lesson, measured again here. The frac-1.0
numbers survive only in git history; the conclusions below did not flip with
the fix, they narrowed.

## The result

120 held-out scenarios (`--seed 90000`, disjoint from every training stream), full
horizon, early stopping **off** (`scripts/eval_reacher_residual.py`). Every row
below was evaluated under the same tree — see "The eval-drift catch". Figure:
`docs/reference/reacher_residual_rl.png` (`scripts/plot_reacher_residual_rl.py`);
the per-scenario CSV records are gitignored like all CSVs — regenerate them
with `scripts/eval_reacher_residual.py` (~13 min).

| controller | reach @200k | reach @400k | best | final | best→final | path/net |
| --- | --- | --- | --- | --- | --- | --- |
| Select-DPC (expert) | 96/120 [72–86%] | — | 2.8 mm | 6.5 mm | 2.3x | 1.5 |
| DAgger clone | 82/120 [60–76%] | — | 5.1 mm | 12.3 mm | 2.4x | 1.7 |
| clone + warm start | 83/120 [60–77%] | — | 5.3 mm | 12.9 mm | 2.4x | 1.8 |
| **clone + residual** | 117/120 | **119/120 [95–100%]** | 1.7 mm | 2.4 mm | **1.4x** | **1.4** |
| **vanilla RL** | 113/120 | **120/120 [97–100%]** | **1.1 mm** | **1.7 mm** | 1.6x | **1.4** |
| random torque | 3/120 [1–7%] | — | 56.5 mm | 181.7 mm | 3.2x | 7.6 |

(`best`/`final`/drift columns are the 400k values for the RL rows; at 200k they
were residual 2.8 → 4.7 mm at 1.7x, vanilla 1.9 → 3.2 mm at 1.7x.)

Paired against the clone (the residual's own baseline), at 400k:

| paired vs the clone | rescues | regressions | closer on | McNemar |
| --- | --- | --- | --- | --- |
| clone + residual | 37 | 0 | 95/120 | `p < 0.001` |
| vanilla RL | 38 | 0 | 105/120 | `p < 0.001` |

Three things follow.

1. **The pipeline works.** The residual lifts its baseline 82 → 119/120 and
   **beats the expert it was cloned from** (96/120). Journey 08's arc reproduces
   on a structurally different system.
2. **The drift goal is met** — the one thing [journey 12](12-select-dpc.md) said
   would lift every row rather than trade one against another. `best→final` is
   1.4x for the residual against the expert's 2.3x and journey 12's 2.1–2.3x
   band. A policy paid on every step of a full horizon holds position better
   than a receding-horizon QP with no terminal term.
3. **And vanilla still wins, now narrowly** — perfect reach, 1.1 against
   1.7 mm `best`, 1.7 against 2.4 mm `final`, `path/net` tied at 1.4.

Doubling training bought both arms real precision (the residual's drift fell
1.7x → 1.4x; vanilla halved both distance medians), and the training curves say
why 200k was short: vanilla's return plateaus only around ~280k, and the
residual is still creeping at 400k.

### The four arms, on video

Claims 1 and 2 above are the two that a table states and a clip shows. Both
scenarios below are from the frozen 120, picked by outcome rather than by index
(`scripts/record_reacher_residual.py`), and both run the **full 50-step horizon
with early stopping off** — so a controller that arrives and then leaves shows up
as the readout's `now` and `best` numbers separating. RL arms are at 200k.

**Claim 1, the residual beating the expert it was cloned from.** Scenario #4, a
341 mm reach — the longest kind in the set. The QP expert stops 45 mm out, the
clone of that expert flies off to 164 mm, and the residual over that same clone
lands and holds at 6 mm:

<div style="overflow-x:auto">
<table>
<tr><th>Select-DPC (expert)</th><th>DAgger clone</th><th>clone + residual</th><th>vanilla RL</th></tr>
<tr>
<td><video controls loop muted playsinline width="200"><source src="../videos/reacher-rescue-ep4-expert.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="200"><source src="../videos/reacher-rescue-ep4-clone.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="200"><source src="../videos/reacher-rescue-ep4-residual.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="200"><source src="../videos/reacher-rescue-ep4-vanilla.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><small>24.7 → 45.0 mm</small></td>
<td><small>36.1 → 164.5 mm</small></td>
<td><small><b>5.4 → 6.0 mm</b></small></td>
<td><small>3.3 → 3.6 mm</small></td>
</tr>
</table>
</div>

**Claim 2, the drift fix, on a scenario both arms reach.** Scenario #1: the clone
touches 1.8 mm and then leaves, ending 11.1 mm out — *outside* the 10 mm ring it
had already been inside. The residual over it touches 1.1 mm and stays at 2.2 mm.
This is the `best→final` column as a picture:

<div style="overflow-x:auto">
<table>
<tr><th>DAgger clone — arrives, then leaves</th><th>clone + residual — arrives, holds</th></tr>
<tr>
<td><video controls loop muted playsinline width="330"><source src="../videos/reacher-hold-ep1-clone.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="330"><source src="../videos/reacher-hold-ep1-residual.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><small>1.8 → 11.1 mm (drifts back outside tolerance)</small></td>
<td><small><b>1.1 → 2.2 mm</b></small></td>
</tr>
</table>
</div>

### The eval-drift catch

The first 200k evaluation was recorded before later working-tree changes, and
its CSV is **not comparable** to any fresh run: under the current tree the
deterministic Select-DPC row scores **96/120 where that CSV recorded 89/120** —
same controller, same frozen scenarios. Two consecutive reruns agree with each
other (96/120 twice), and the 200k RL rows of the frac-1.0 era reproduced
exactly (113/120 both arms at the time), so the table above is internally
consistent.

The rule this buys: **never compare eval CSVs written under different code
states.** Re-running the older policies under current code costs ~13 minutes
(the QP rows dominate; the RL rows are milliseconds per episode) and turns an
apples-to-oranges table into a real one.

### Why vanilla's training return is higher — the reward decomposed

Vanilla's training return sits above the residual's from ~200k on, which reads
as a contradiction next to the residual's best-in-table drift ratio.
`scripts/decompose_reacher_returns.py` rolls both 400k policies over the 120
scenarios and splits
`return = −Σ dist + 1.0·(steps in tolerance) − 1e-3·Σ|u|²`:

| | return | Σ dist | in-tol steps (of 50) | first reach | ctrl cost |
| --- | --- | --- | --- | --- | --- |
| residual 400k | 37.3 | **1.35** | 38.6 | step 12 | 0.017 |
| vanilla 400k | **38.1** | 1.37 | **39.5** | step 12 | 0.010 |

**The gap is 0.8 points and it is the station-keeping bonus** (0.9 more
in-tolerance steps; the distance integral now favors the *residual* by 0.02,
and the residual pays ~0.007 more control cost). Paired per scenario, holding
is a tie: the residual holds longer on 47/120 against vanilla's 43/120, with
30 tied — the frac-1.0 run's 75-vs-37 deficit was the parameterization, not
the pipeline.

That still corrects a natural misreading of the drift column. The residual's
1.4x against vanilla's 1.6x is a **ratio to its own `best`**, and its `best`
is worse (1.7 vs 1.1 mm) — the denominator flatters it. In absolute terms
vanilla is closer at best and closer at the end. What the residual's 1.4x
honestly claims is narrower: it fixed the drift pathology it inherited
(clone 2.4x → 1.4x), which was journey 12's target.

What remains of the gap is small and structural: the residual is anchored to
the clone's steering and parks ~0.7 mm further from center, which on
knife-edge scenarios costs the occasional bonus step. The frac-1.0 version of
this section read the same mechanism at 3x the size; widening the residual's
authority shrank it without changing its sign.

### The three it misses

Which scenarios the residual fails is a scan, not a sample: the recorder scores
all 120 with the deployed policy and picks by outcome
(`scripts/record_reacher_residual.py --scan 120 --n-miss 3`). Both RL arms below
are their **200k** checkpoints — the `reach @200k` column above (117/120 and
113/120), not the 400k plateau where vanilla is perfect. The misses are **#0,
#52 and #93**, and the widest best→final gap the residual still owns is **#111**,
which the reach rate scores as a success.

| scenario | start | Select-DPC | DAgger clone | clone + residual | vanilla RL |
| --- | --- | --- | --- | --- | --- |
| #0 | 58 mm | 3.0 → 3.1 | 5.4 → 14.1 | **16.7 → 16.9** | 4.2 → 5.5 |
| #52 | 57 mm | 0.8 → 22.0 | 4.6 → 13.7 | **10.7 → 24.0** | 5.6 → 5.7 |
| #93 | 182 mm | 2.8 → 3.3 | 10.9 → 27.9 | **22.8 → 24.4** | 8.8 → 8.8 |
| #111 | 198 mm | 0.7 → 1.5 | 1.4 → 1.9 | **5.7 → 34.7** | 18.1 → 18.1 |

(`best → final` in mm, 10 mm tolerance, seed 0. Bold is the arm under test.)

Two things here that the aggregate rows cannot show. **The base is already
outside or on the edge in all three misses** — the clone's best is 5.4 / 4.6 /
10.9 mm and its final is outside tolerance in every one, so the residual is
correcting a trajectory that arrived badly; on #0 it ends *further* out than the
clone it wraps. And **#52 is a 0.7 mm loss** scored as a flat zero.

#111 is the residual's own worst case rather than an inherited one: expert and
clone both park (1.5 and 1.9 mm final) while the residual arrives at 5.7 mm and
leaves, ending 34.7 mm out — 6.1x on this episode, where the table's median
distances give 1.4x. It costs no reach because vanilla misses #111 outright
(18.1 mm, never inside), which is what a 113/120 arm looks like at 200k.

<div style="overflow-x:auto">
<table>
<tr><th></th><th>DAgger clone</th><th>clone + residual</th><th>vanilla RL</th></tr>
<tr>
<td><b>#0</b><br><small>the residual ends further out than the clone it wraps</small></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-miss-ep0-clone.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-miss-ep0-residual.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-miss-ep0-vanilla.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><b>#52</b><br><small>a 0.7 mm loss — best 10.7 mm against a 10 mm ring</small></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-miss-ep52-clone.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-miss-ep52-residual.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-miss-ep52-vanilla.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><b>#93</b><br><small>the longest reach in the set; the clone stalls, the residual stalls further out</small></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-miss-ep93-clone.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-miss-ep93-residual.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-miss-ep93-vanilla.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><b>#111</b><br><small>scored as a reach — arrives at 5.7 mm, ends 34.7 mm out</small></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-drift-ep111-clone.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-drift-ep111-residual.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-drift-ep111-vanilla.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
</tr>
</table>
</div>

Clips run the full 50-step horizon with early stopping **off**, so "arrive and
leave" reads as the readout's `now` and `best` separating. The Select-DPC column
is omitted above for width; all four arms are in `videos/reacher_residual_frac2/`:

```bash
uv run python scripts/record_reacher_residual.py --scan 120 \
  --residual data/reacher_ckpt_seeds/resf2_s0/ckpt_200000_steps.zip \
  --residual-frac 2.0 --n-miss 3 --only residual_miss,residual_widest_drift \
  --out-dir videos/reacher_residual_frac2
```

`videos/reacher_residual/` is the same recorder run against the **superseded
frac-1.0** policy, which misses 7 of 120 (#29, #38, #44, #68, #80, #107, #115) —
the reach cost of the narrower authority, visible episode by episode.

### The crossover, measured

Both 400k runs saved a checkpoint every 25k steps, so the sample-efficiency
claim can be read off deployed policies instead of training curves:
`scripts/sweep_reacher_checkpoints.py` evaluates every checkpoint greedily on
the 120 frozen scenarios (figure: `docs/reference/reacher_crossover.png`; the
sweep CSV is gitignored and regenerates in ~10 min).

| steps | residual | vanilla | | steps | residual | vanilla |
| --- | --- | --- | --- | --- | --- | --- |
| 25k | **44**/120 | 5/120 | | 150k | **113**/120 | 112/120 |
| 50k | **80**/120 | 29/120 | | 200k | **117**/120 | 114/120 |
| 75k | **102**/120 | 36/120 | | 275k | 113/120 | **120**/120 |
| 100k | **95**/120 | 87/120 | | 400k | 119/120 | **120**/120 |

The figure's panel C plots the per-checkpoint difference directly: **+55 pp at
its widest (75k)**, a dead tie at 125k, the first vanilla lead at 175k, and
never outside ±6 pp after that — this seed ends at −0.8 pp (119 vs 120).

Rerun across **five training seeds** per arm (seeds 1–4 added 2026-08-24;
checkpoints and per-seed sweep CSVs in `data/reacher_ckpt_seeds/`, figure
`docs/reference/reacher_crossover_seeds.png` via
`scripts/plot_reacher_crossover_seeds.py`; CSVs are gitignored repo-wide), the
crossover survives but its location does not. The residual leads on 5/5 seeds
at 25k and 50k (pooled +25 and +43 pp over 600 episodes) and vanilla leads on
5/5 seeds only from 375k on (pooled 600 vs 583, then 599 vs 589 at 400k) —
but the first checkpoint where vanilla catches the residual is
**75k / 100k / 125k / 125k / 375k** depending on seed. The pooled mean
difference crosses zero between 125k and 150k, so "~150k" survives only as
the average; the claims below are stated against the seed range.

Three measured claims replace the asserted one:

1. **The head start is worth ~125–150k steps on average — 75k–375k by seed.**
   On this seed the residual dominates at every checkpoint through 100k
   (102 vs 36 at 75k is the widest gap); the two are inside each other's
   intervals from 125k on.
   Vanilla needs ~100k steps to catch the frozen clone (82/120) and ~125–150k
   to reach expert level. Precision separates them where reach cannot: the
   residual's median final distance is under the 10 mm tolerance by 50k,
   vanilla's by ~110k, and vanilla is the more precise arm from ~150k on
   (3.3 vs 5.2 mm there, 2.0 vs 2.5 mm at 400k, medians over 5 seeds).
2. **The zero-init "floor" is not a floor.** At 25k the residual scores
   44/120 — far *below* the clone it wraps — and only recovers the clone's
   level around 75k (the frac-1.0 run read 51/120 at 25k; wider authority
   digs the early hole slightly deeper). Early SAC exploration degrades the
   base before improving it. A deployment that counts on "never worse than
   the baseline" during training does not get it from this setup; what it
   gets is "far better than learning from scratch."
3. **The zero-training point is the clone itself** — the pipeline's real
   pitch is that it starts at 82/120 having spent zero environment steps,
   where vanilla starts at random. The honest statement of the advantage is
   therefore budget-shaped, with the five-seed bands: below ~50k environment
   steps the pipeline's best policy wins on every seed (4/5 at 75k); above
   ~375k vanilla does on every seed, by 1–4 scenarios; in between the winner
   is seed-dependent (mean crossing ~125–150k).

### The head start, on one episode

The same scenario #4 as above, held fixed while only the training budget changes
(`--episode 4`, seed 0, `best → final` under each clip). This is the sweep table
as a picture: the residual is useful at 75k where vanilla is still 21 mm short,
and by 400k the ordering has reversed.

<div style="overflow-x:auto">
<table>
<tr><th>budget</th><th>clone + residual</th><th>vanilla RL</th></tr>
<tr>
<td><b>25k</b><br><small>residual below the clone it wraps; vanilla barely leaves its start</small></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-ladder-ep4-025k-residual.mp4" type="video/mp4">Your browser does not support the video tag.</video><br><small>30.8 → 43.1 mm</small></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-ladder-ep4-025k-vanilla.mp4" type="video/mp4">Your browser does not support the video tag.</video><br><small>87.1 → 88.5 mm</small></td>
</tr>
<tr>
<td><b>75k</b><br><small>the widest gap on the sweep — 102/120 against 36/120</small></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-ladder-ep4-075k-residual.mp4" type="video/mp4">Your browser does not support the video tag.</video><br><small><b>2.8 → 3.6 mm</b></small></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-ladder-ep4-075k-vanilla.mp4" type="video/mp4">Your browser does not support the video tag.</video><br><small>21.7 → 21.8 mm</small></td>
</tr>
<tr>
<td><b>200k</b><br><small>both reach; vanilla is already the tighter of the two</small></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-ladder-ep4-200k-residual.mp4" type="video/mp4">Your browser does not support the video tag.</video><br><small><b>5.4 → 6.0 mm</b></small></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-ladder-ep4-200k-vanilla.mp4" type="video/mp4">Your browser does not support the video tag.</video><br><small><b>3.3 → 3.6 mm</b></small></td>
</tr>
<tr>
<td><b>400k</b><br><small>plateau — vanilla is the more precise arm</small></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-ladder-ep4-400k-residual.mp4" type="video/mp4">Your browser does not support the video tag.</video><br><small><b>2.0 → 2.2 mm</b></small></td>
<td><video controls loop muted playsinline width="240"><source src="../videos/reacher-ladder-ep4-400k-vanilla.mp4" type="video/mp4">Your browser does not support the video tag.</video><br><small><b>0.3 → 1.2 mm</b></small></td>
</tr>
</table>
</div>

One episode is an illustration, not evidence — the 5-seed sweep above is the
evidence, and it is what the seed-dependent crossover (75k–375k) is measured
from. Reproduce any row with:

```bash
uv run python scripts/record_reacher_residual.py --episode 4 \
  --residual data/reacher_ckpt_seeds/resf2_s0/ckpt_75000_steps.zip \
  --residual-frac 2.0 --vanilla data/reacher_van_ckpt_400k/ckpt_75000_steps.zip \
  --out-dir videos/reacher_ladder/75000
```

## How DAgger solves the imitation-learning problem

This is the fix that made the clone usable, so it is worth being precise about.

The problem in one sentence: the clone was trained to answer *"what does the
expert do at the states the expert visits?"*, but at deployment it has to
answer *"what would the expert do at the states **I** visit?"* — and those are
different states, because the clone's own small errors steer it somewhere the
expert never went. The fix in one sentence: let the **student drive**, query
the expert for labels **at the student's states**, and retrain on the union —
so the training distribution becomes the deployment distribution.

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
training data exactly never. No amount of extra expert-driven data can fix that,
because more of it lands on the same wrong distribution — which is why every
off-policy intervention in the exclusion table below failed.

([Journey 14](14-clone-coverage.md) later reproduced this signature on the
*unicycle* by ablation: trained on expert-rollout data alone, its clone's
ratio is 2.68x and the closed loop halves — the failure follows the data,
not the system. The unicycle's original clone escaped only because journey
07's dataset mixed in broad synthetic coverage.)

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

### What it looks like

Both arms below are the *same* network trained the same way; the only difference
is where the training states came from — round 0 rolls the **expert** and labels
with it (the collection every dataset in this repo used before this entry), round
3 adds 15,000 rows rolled by the **clone** and labelled by the expert. On the
frozen 120 the two arms split **66/120 → 82/120**, with 30 scenarios DAgger
rescues and 14 it loses (`scripts/record_reacher_dagger.py`, early stopping off,
full 50-step horizon).

<div style="overflow-x:auto">
<table>
<tr><th></th><th>BC clone — expert-driven data only</th><th>DAgger clone — round 3</th></tr>
<tr>
<td><b>#30</b><br><small>rescue: a 338 mm reach the BC clone never starts</small></td>
<td><video controls loop muted playsinline width="300"><source src="../videos/reacher-dagger-rescue-ep30-bc.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="300"><source src="../videos/reacher-dagger-rescue-ep30-dagger.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td></td>
<td><small>157.9 → 194.7 mm</small></td>
<td><small><b>6.4 → 22.1 mm</b></small></td>
</tr>
<tr>
<td><b>#16</b><br><small>both reach; the BC clone arrives and then leaves</small></td>
<td><video controls loop muted playsinline width="300"><source src="../videos/reacher-dagger-hold-ep16-bc.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="300"><source src="../videos/reacher-dagger-hold-ep16-dagger.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td></td>
<td><small>3.6 → 66.1 mm (18x drift)</small></td>
<td><small><b>0.5 → 2.4 mm</b></small></td>
</tr>
<tr>
<td><b>#4</b><br><small>regression: BC reaches, DAgger does not — 14 of 120 go this way</small></td>
<td><video controls loop muted playsinline width="300"><source src="../videos/reacher-dagger-regress-ep4-bc.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="300"><source src="../videos/reacher-dagger-regress-ep4-dagger.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td></td>
<td><small><b>5.3 → 33.6 mm</b></small></td>
<td><small>36.1 → 164.5 mm</small></td>
</tr>
<tr>
<td><b>#8</b><br><small>neither reaches, and DAgger is the worse of the two</small></td>
<td><video controls loop muted playsinline width="300"><source src="../videos/reacher-dagger-neither-ep8-bc.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="300"><source src="../videos/reacher-dagger-neither-ep8-dagger.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td></td>
<td><small>12.1 → 14.2 mm</small></td>
<td><small>147.7 → 329.9 mm</small></td>
</tr>
</table>
</div>

The last two rows are the honest half of the +16: DAgger moves the *distribution*
of failures, it does not monotonically dominate. #4 is the same scenario the
residual section opens with — the clone flying off to 164.5 mm there is the
**DAgger** clone, and the pre-DAgger one happens to handle it. The gate is a
paired count over the whole set, and 30 rescues against 14 losses is what it is
measuring.

```bash
uv run python scripts/record_reacher_dagger.py --scan 120
```

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

### On the 400k extension

- **Continuing the 200k checkpoints instead of retraining.** The train scripts
  build a fresh model per run; adding resume for a 20-minute saving wasn't
  worth the code. The 400k runs are fresh same-seed runs, not continuations.
- **Comparing against the original 200k CSV as-is.** Rejected once the
  Select-DPC row moved 89 → 96; that difference is code drift, not learning.

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
  rows beat the expert's drift, and at 400k the residual holds 1.4x against the
  expert's 2.3x — which is what journey 12 predicted a fix would look like. The
  50k reading is withdrawn.
- **A test that could never fail.** `test_never_terminates_on_reach` set the goal
  60 mm from the held tip against a 10 mm tolerance, so `reached` never fired.
  Fixed to 5 mm with an `ever_reached` assertion, verified by injecting the
  regression it was blind to.

## Caveats

- **The headline table is one seed per policy** (seed 0), one `reach_bonus`.
  `residual_frac` IS swept now — 1.0 vs 2.0, 5 seeds each, and 2.0 wins on
  every seed, which is why this entry reports it — but the 200k → 400k deltas
  still ride on single runs. The paired McNemar tests and the 5-seed pooled
  counts carry the claims, not the point estimates.
- **The DAgger clone imitates the *memoryless* expert**, so the `Select-DPC` row is
  built with `carry_prediction=False` to match. Comparing it against the carried
  variant would score the clone against a controller it never imitated.
- The Wilson intervals of 119/120 and 120/120 overlap; "vanilla wins reach" is
  a 1-scenario difference on this seed. The load-bearing comparisons are the
  distance medians (1.7 vs 2.4 mm final) and the 5-seed pooled reach
  (599 vs 589/600), where vanilla leads on every seed at 375k–400k.
- The crossover sweep is one training run per arm sliced at 16 checkpoints, not
  16 independent runs — adjacent points share history, so the curve's noise
  (residual oscillating 106–119 after 175k, vanilla's 119 blip at 300k) is
  autocorrelated. The single-seed uncertainty on the crossing was measured
  with four more seeds per arm: first catch-up ranges 75k–375k
  (`docs/reference/reacher_crossover_seeds.png`).
- `path/net` for the `random` row is a median over 118, not 120: two scenarios made
  no progress, so `eff` is NaN there by design.
- **The invariant tests skip on a clean checkout.** `data/` is gitignored, so the
  bit-for-bit zero-residual test and both zero-init tests `pytest.skip` elsewhere
  and the suite still reports green. The training monitors behind the return
  figure live there too; every 400k run saved 25k-step checkpoints — the
  frac-2.0 residual seeds in `data/reacher_ckpt_seeds/resf2_s{0..4}/`, the
  vanilla seeds in `data/reacher_van_ckpt_400k/` (seed 0) and
  `data/reacher_ckpt_seeds/van_s{1..4}/`, and the superseded frac-1.0
  residual runs in `data/reacher_res_ckpt_400k/` and
  `data/reacher_ckpt_seeds/res_s{1..4}/`.
- Observation normalization in `ResidualSelectEnv` is min-max against declared
  bounds and the live channels are compressed ~18x relative to the widest; untested
  as a factor now that the residual works.

## Next

1. **Close the verdict with a written decision.** The crossover is measured:
   the pipeline wins below ~50k environment steps on every seed and loses
   above ~375k on every seed, with seed luck in between. Either
   retire it to "use when interaction is expensive or unsafe" — with the
   caveat that early residual training dips below the frozen clone, so the
   safe deployment during that window is the clone itself — or drop it from
   the Panda plan entirely.
2. **Sweep DAgger rounds and episodes-per-round.** Three rounds of 100 was the
   first thing tried and it worked; the knee is unmeasured, and a 3-episode smoke
   test already moved the gate.
3. **Re-run the residual on the one-hot fixed-anchor clone.** DAgger and the
   one-hot were never combined — the one-hot gave the best representation, DAgger
   fixes the distribution, and they address different halves of the problem.
4. **The Panda still needs on-policy collection** ([12](12-select-dpc.md) item 4).
   `rl/` is now the shared package that would serve it, and DAgger is exactly the
   on-policy mechanism that entry asked for.

# Metrics

Every number in the [Journey](../journey/index.md) comes from one of the measures
below. This page says what each one *is*, what it *represents*, and — the part
that matters — **which conclusion it carries**, because several of them look
interchangeable and are not. Where a metric has a trap that has already cost this
project a wrong reading, the trap is recorded with it.

Two families, answering two different questions:

| family | question | used to decide |
| --- | --- | --- |
| **open-loop** (`skill`, `cos`, `span`, tip RMSE) | *can the data predict?* | whether a library/bank is worth anything, before spending a QP on it |
| **closed-loop** (`best`, `final`, `reach rate`, `path/net`) | *can the controller do the task?* | whether the whole loop works |

The open-loop family is the cheap gate. It runs without a solver, so it is what
every new library set is screened with; a controller cannot outperform the data it
predicts from, so a failed gate ends the investigation before the expensive part.

---

## Open-loop: is the data informative?

All three come from `scripts/verify_libraries.py::probe` (Panda) and its Reacher
and Select-DPC twins, scored on a held-out trajectory the library never saw.

### `skill` — the one that carries the coverage argument

$$\text{skill} = 1 - \frac{\operatorname{MSE}(\hat{y}_{\text{tip}},\, y_{\text{tip}})}{\operatorname{MSE}(y_{\text{tip},0},\, y_{\text{tip}})}$$

Mean squared tip-prediction error over the `N`-step horizon, divided by the error
of a **"the tip does not move" predictor** that just holds the last measurement,
subtracted from 1.

**What it represents.** Skill against a null model, in the meteorological sense —
not accuracy. The scale is what makes it readable:

| value | meaning |
| --- | --- |
| `1.0` | perfect prediction |
| `0.93` | what a library achieves **at its own anchor** |
| `0.0` | **no better than assuming the arm is frozen** |
| `−9.93` | error is ~10× the frozen predictor's — actively misleading |

**How it counts.** Normalising against a null is what lets a single threshold —
`skill < 0` — mean the same thing on a 7-DoF position-servo arm and a 2-DoF
torque arm, with tip motions differing by an order of magnitude. That is what
makes the [~0.5 rad validity radius](../journey/12-select-dpc.md) a *shared*
result across the two systems rather than two unrelated numbers, and it is why a
negative skill is fatal rather than merely poor: a controller following a
below-zero predictor does worse than one that believes nothing will happen.

!!! warning "Skill is unbounded below, so medians — never means"
    A single pathological configuration can score −200 and swamp an average. Every
    table in this repo reports the **median** over held-out configurations, plus
    the *share* that are negative (`26/40` on the Panda), which is the more robust
    statement.

### `cos` — direction, separated from magnitude

Cosine between predicted and true **net tip displacement** over the horizon,
$\cos(\widehat{\Delta p},\, \Delta p)$, measured end-to-start rather than
step-by-step.

**What it represents.** Whether the prediction points the right way, independent
of whether it gets the distance right. A predictor can be badly scaled and still
steer correctly; the QP mostly needs direction.

**How it counts.** It separates two failure modes that `skill` alone conflates.
At 2 rad the Panda scores `cos = −0.03` with **50% of predictions negative** —
the library points the *wrong way* half the time. That is a qualitatively
different failure from "predicts the right direction, wrong magnitude", and it is
the specific finding that made *more anchors* the wrong fix. It is also the metric
Select-DPC improves most (0.23 → 0.52), which is how we know selection genuinely
works and is still not enough.

### `span` — can the library represent the trajectory at all?

Relative least-squares residual of $[U_p; Y_p; U_f; Y_f]\,g = \tau_{\text{true}}$,
with **no** regularisation.

**What it represents.** Willems' fundamental lemma says any trajectory of an LTI
system lies in the column span of a sufficiently exciting Hankel matrix. `span`
tests that premise directly.

!!! warning "A span of 0.000 can mean the test is vacuous"
    On the Panda it reads `0.000` at *every* radius including 4 rad — not because
    the library is good, but because 1484 columns against 289 Hankel rows span the
    entire space. Any trajectory is representable, so consistency constrains
    nothing and `λ_g` alone picks among infinitely many `g`. **A span residual is
    only informative when columns do not vastly outnumber rows.** Read it beside
    the column/row ratio or not at all.

### `tip RMSE`

Root-mean-square tip-prediction error in millimetres — `skill`'s numerator,
unnormalised.

**How it counts.** The physically interpretable companion to `skill`: 151 mm of
prediction error against a 50 mm `goal_tolerance` says the failure is large
compared to the task, which "skill = −1.83" alone does not convey. Use it for
scale, use `skill` for comparison.

---

## Distance: the axis every open-loop metric is plotted against

### `radius` / `distance from the nearest data` (rad)

Euclidean distance in **joint space** between the configuration being predicted
and the nearest configuration the library was built from. Wrapped on Reacher's
periodic `q0` (`reacher/model.py::config_distance`) — unwrapped, `−3.1` and `+3.1`
rad look 6.2 apart instead of 0.08.

**What it represents.** How far the controller is asking a *local* linear model to
extrapolate. Local libraries are local; this is the axis along which "local" is
measured.

**How it counts.** This is the independent variable of the entire Panda argument.
`skill` and `cos` are plotted against it, and the resulting curve crosses zero at
~1 rad on both arms — so the whole question becomes "where does each system
*operate* on this axis?" Reacher's anchor spacing is 0.52 rad; the Panda's nearest
data is 1.98 rad away. Same curve, different operating point, opposite outcomes.

Joint distance is a sound proxy for model difference, and that was checked rather
than assumed: `corr(log distance, model difference) = 0.749`, 56% of variance
(journey 11).

### `d_sample` / `d_bank` / `d_sel` — what a *selection* rule can reach

From `scripts/measure_selection_distance.py`, in increasing order of how much
convention they carry:

| | definition | what it bounds |
| --- | --- | --- |
| `d_sample` | distance to the nearest configuration **anywhere in the collection** | the absolute floor — no rule can select data that was never collected |
| `d_bank` | distance to the nearest Hankel **column** (located at its window start) | the floor for a selector that must take whole length-`L` windows |
| `d_sel` | median distance to the columns Algorithm 2 **actually picks** | what this particular rule achieves |

**How it counts.** `d_sample` converts an empirical result into a structural one.
"Select-DPC scored −1.83" is a statement about one algorithm; "the nearest of
97,500 collected samples is 1.48 rad away against a 0.5 rad validity radius" is a
statement about *every possible* selection rule, Isomap and oracles included. That
is the strongest form of the coverage argument, and it is why the remaining route
is on-policy collection rather than a better selector.

`d_bank` and `d_sample` agreeing to 0.03 rad is what rules out the windowing
convention as an artefact — a column is a 17-sample window that itself traverses
0.58 rad, so the check was necessary.

### `r_K` — what coverage costs

Worst-case distance from a task configuration to its nearest anchor, given `K`
anchors (`scripts/anchor_coverage.py`, farthest-point).

**How it counts.** `r_K` decays as a clean power law with **no elbow**, so the
anchor count needed to reach a target radius is an extrapolation rather than a
guess: ~570 anchors at `d = 3.1`, ~165,000 at `d = 5.7`. Without the power-law
fit, "we need more anchors" has no price attached and the argument stalls.

### `effective dimension` and `silhouette`

Effective dimension is the participation ratio of the configuration covariance
spectrum — how many directions the sampled set actually occupies. Silhouette is
the standard clustering separation score.

**How they count.** Effective dimension is the exponent in the `8^d` coverage
arithmetic, so it is the single number that moves the anchor budget from
~126,000 (uniform, `d = 5.65`) to 512 (IK-seeded, `d = 3.00`) — two orders of
magnitude, from impossible to ~8.5 hours of collection. Silhouette stuck at
0.13–0.26 is the negative result that says there is **no cluster structure to
discover**: this is a manifold, so `K` is a resolution knob you choose, not
something the data hands you.

---

## Closed-loop: does the controller do the task?

From `scripts/eval_reacher_scenarios.py::episode` and its Panda twin. `need` is
the initial tip-to-goal distance; `path` is the arc length the tip travels.

### `best` vs `final` — the distinction that was missing for months

- **`best`** — closest approach at any point in the episode. This is the reach
  criterion: `reached = best < tol`.
- **`final`** — distance at the **last** step of the full horizon.

**What they represent.** `best` asks *did it get there*; `final` asks *did it stay*.

**How they count.** They diverge by a constant factor for every controller —
2.1×, 2.2×, 2.3× for fixed anchors, Select `n_max=1`, Select `n_max=3`, and 3.8×
for random torque. **That constancy is the finding.** A data or selection problem
would hit the controllers differently; a uniform factor points at the
receding-horizon cost itself, which carries no terminal term and nothing rewarding
station-keeping once tracking is nearly satisfied over the horizon.

!!! danger "Two traps here, both already paid for"
    **Early stopping censors `final`.** Every run before journey 12 stopped at
    first contact, so a reached episode's "final" was wherever the tip happened to
    be when it crossed the threshold, not where it settled. The drift above is
    invisible under early stopping and was found only by removing it
    (`--no-early-stop`).

    **`scripts/run_select_dpc.py` prints `best` under the header `final`.** Its
    `episode()` returns `{"final": best}`. The two scripts' `final` columns are not
    the same quantity — read `eval_reacher_scenarios.py`'s as convergence and
    `run_select_dpc.py`'s as closest approach.

**Consequence, stated plainly in journey 12: every reach-rate number in this
project flatters its controller.** They touch the target and leave.

### `reach rate`

Share of scenarios with `best < goal_tolerance`, reported with a **Wilson 95%
interval** rather than a bare fraction.

**How it counts.** The headline number, and the weakest one — it is binary, so it
discards *how* the episode went and inherits the `best`-vs-`final` flattery above.
This is why no conclusion in the repo rests on reach rate alone: journey 12's
Select-DPC-beats-anchors claim is carried by the **paired** per-scenario
comparison (78/120 closer), because the Wilson intervals overlap.

Reach rate is also why the **random-action baseline is mandatory**: the env is
built so that 0/20 random-action reaches make any nonzero rate signal rather than
noise. Journey 11 records the cost of skipping it — a random walk of equal command
magnitude beat the unconstrained controller 38.5% to 6.1%, and every
interpretation made before that point was against the wrong reference.

### `path/net` — the metric that separates steering from drifting

$$\text{path/net} = \frac{\text{arc length the tip travels}}{\text{need} - \text{best}}$$

Distance travelled divided by progress made toward the goal.

**What it represents.** 1.0 means the tip travelled exactly the distance it needed
to — a straight line to the target. 50 means it moved fifty times that far to
achieve the same closure: thrashing.

**How it counts.** This is the most diagnostic closed-loop number in the repo,
because it separates *the controller is steering but ran out of budget* from *the
controller has no idea where it is going* — which reach rate alone cannot do. The
gating experiment is entirely carried by it:

| start distance from an anchor | reached | path/net |
| --- | --- | --- |
| 0.00 rad | 3/3 | **1.0** |
| 0.50 rad | 2/3 | 1.3 |
| 2.00 rad | 1/3 | **53.8** |

Same controller, same QP, same libraries — only the distance to data changes.
`path/net = 1.0` at the anchor is what establishes that **nothing is wrong with
the controller**; 53.8 at 2 rad is what establishes that the data is the problem.
Without this metric the row reads "1/3 vs 3/3", which is equally consistent with a
tuning bug.

It is also the tell in the closed-loop comparison table: three Panda controllers
all score 0/10, but random walk's 7.5 against fixed anchors' 13.9 says the
"controller" is doing *worse than nothing*, not merely failing.

### `closed %`

`100 · (need − best) / need` — the fraction of the initial gap that was closed.

**How it counts.** The graded version of reach rate, for regimes where nothing
reaches. 87.9% closed at an anchor vs 3.7% at 2 rad carries the same conclusion as
`path/net` there, and is readable when every episode scores 0/10.

### `steps`, `iters`, `ms/step`

Control steps to first reach; mean Select-DPC inner iterations; wall-clock per
control step.

**How they count.** `iters` is what explains the `n_max` sweep: the convergence
tolerance only starts firing at `n_max = 5`, so the loop mostly runs to its cap
and the prediction drifts from the measured state — which is why `n_max = 1` wins
and beyond 3 is strictly dominated. `ms/step` is what makes "better **and cheaper**"
a claim rather than a hope (5.5 m vs 22.0 m of compute).

---

## Statistics

| | what it is | when this repo uses it |
| --- | --- | --- |
| **Wilson 95% interval** | binomial CI that stays inside [0,1] at small `n` and near the boundaries | every reach rate; a normal-approximation CI is wrong at 0/10 and 96/120 alike |
| **Paired per-scenario test** | run both controllers on *identical* scenarios, count wins | when Wilson intervals overlap — it removes scenario difficulty as a variance source and is what carries the Select-DPC claim |
| **McNemar** | paired test for binary outcomes: rescues vs regressions | the residual-RL result — 38 rescues, **0** regressions, `p < 10⁻⁴`; the zero-regression half is the load-bearing part |

---

## The short version

If you read only one row per family:

- **`skill < 0`** means the data is worse than useless, and no controller
  recovers from it.
- **`path/net ≈ 1`** means the controller is fine and something else is wrong.

Every Panda conclusion in this repo is those two facts in different clothes: the
controller steers perfectly where data exists (`path/net = 1.0`), and there is no
data where it operates (`skill = −9.93`, `d_sample = 1.48 rad`).

# 11. Panda anchors — the controller works, the libraries don't reach

## Decision

Build the anchor-selection pipeline for `PandaReach`'s DeePC stage on a new
interface — **`u = q_des`** (absolute joint target) and **`y = [q; p_ee]`** (10-D)
— with anchors from **k-medoids over task-relevant configurations** and library
selection by **nearest anchor in joint space**. Two changes to the plan as written
were forced by measurement: the output is the tip *vector*, not the plan's scalar
goal-distance, and the QP carries a **hard input-rate constraint**
`|u_j − u_{j−1}| ≤ du_max`.

The result: **inside ~0.5 rad of an anchor the controller is excellent** — 3/3
reaches with a path-to-progress ratio of **1.0**, i.e. it travels almost exactly
the distance required. Beyond ~2 rad it collapses completely. Measured cluster
radius is **~4 rad**.

So the open problem is **coverage, not control**. Nothing in the formulation,
collection, or QP is broken. Raising `K` cannot close the gap at any affordable
number, which is why the next build is per-timestep data selection
([Select-DPC](https://arxiv.org/abs/2503.18845)) rather than more anchors.

## Context — a 7-DoF arm is not a unicycle, in a specific way

Entry [05](05-library-switching.md) established the unicycle needs four local
libraries because its dynamics are bilinear in heading. The Panda needs local
libraries too, but for a different reason, and the difference decides everything
downstream.

**Joint 1 is an exact symmetry, not a nonlinearity.** Its axis is vertical and
gravity is along the same axis, so the joint-space dynamics cannot depend on `q₁`.
Driving `q₁ = −1.8` and `q₁ = 0.6` with identical commands and de-rotating by
`Rz(Δq₁)` agrees to **6.661e-16 m**. The pre-existing `azimuth_key` in
`panda/deepc_setup.py` therefore keys on a quantity that carries no model
information — the four azimuth libraries differ only by a known rotation. That
result is what motivated keying on the full configuration instead.

**The state is 14-D and the tip does not observe it.** `mjd_transitionFD` at the
50 Hz control rate gives `A ∈ ℝ¹⁴ˣ¹⁴`, `B ∈ ℝ¹⁴ˣ⁷` for `x = (q, q̇)`. Building the
observability matrix against a tip-only output leaves a **4-dimensional blind
subspace** — exactly `7 − 3`, the self-motion manifold, confirming from local
linear algebra what `PandaReachEnv.y_ext`'s docstring measured globally from 132
configuration pairs. Putting `q` in the output fixes it (`cond ≈ 98` against
`1e13`–`1e20` for tip-only), which is why the plan's `y = [q; …]` is right.

Reproduce: `scripts/plot_panda_analysis.py --figure statespace`.

## Considered

### The output: the plan's `d_g` vs the tip vector

The plan specifies `y = [q; ‖p_ee − p_g‖]`. Splitting the §8 prediction error by
channel showed the scalar distance channel failing 4–6× worse than the joint
channels could account for:

| region | q RMSE | tip error q alone explains | actual `d_g` RMSE |
| ------ | ------ | -------------------------- | ----------------- |
| 0      | 0.024 rad | 12.2 mm                 | 57.4 mm           |
| 1      | 0.020 rad | 10.1 mm                 | 76.2 mm           |
| 2      | 0.021 rad | 10.4 mm                 | 40.2 mm           |
| 3      | 0.021 rad | 10.5 mm                 | 51.0 mm           |

Switched to `y = [q; p_ee]`. **Honest caveat: this did not deliver what was
predicted** — the change bought ~10%, not the 5× expected. The extra error was FK
curvature, which the vector form inherits too. The change was kept on two smaller
merits that do hold: the 2-norm has a kink exactly at the target where the
controller finishes, and — the real win — **`p_ee` does not depend on the goal, so
the libraries are goal-free**. One Hankel build serves every goal, and the per-goal
retargeting machinery `panda/qdes.py` used to carry was deleted.

`Q = diag(0₇, I₃)` keeps the tracking cost identical to tip-only: the joint block
informs *prediction* through the `Yp`/`Yf` constraints without entering the
objective — the same trick `deepc_setup(output="ext")` uses.

### Rate limiting: post-hoc clip vs a constraint in the QP

`u_bounds` limits *where* the target may sit, not how far it may jump. Under an
absolute joint target that gap is fatal — the QP commanded a median
**1.9–3.1 rad per 20 ms step**, walking straight out of the neighbourhood its
library was collected in. (The delta interface never had this problem:
`DELTA_MAX = 0.2` *was* a rate limit by construction.)

Clipping the returned action was tried first and is wrong in principle: the QP
plans a trajectory premised on a move the plant never makes, so the plan and the
action disagree from step one. The constraint now lives inside the optimization
(`core/deepc.py`, `du_max`, default `None` so every unicycle result is unchanged).
Verified to bind exactly, and the horizon prediction barely degrades — the QP
re-plans something feasible rather than being throttled.

A **hard constraint** over [DeePC-GS](https://arxiv.org/abs/2509.26334)'s soft
`S‖Δu‖²` cost, because `test_cluster_lti.py` measures the validity envelope as a
threshold, not a penalty — and a cost cannot guarantee the plan stays inside it.

`du_max = 0.02` also turns out to be the only setting a real Panda could execute:
at 50 Hz it is **1.0 rad/s**, under the rated 2.175, while 0.05 → 2.5 rad/s and
0.10 → 5.0 rad/s are above it.

### More anchors vs dropping anchors

Deferred to the Outcome — the arithmetic below decides it.

## Outcome

### The clusters are not LTI at the radius they cover

`scripts/test_cluster_lti.py` splits "LTI" into three separate questions.
Time-invariance is free (MuJoCo carries no explicit time dependence; measured
`0.00e+00`). The other two fail:

**Linearity at a point** — superposition error of the tip response, median:

| command amplitude | anchor 0 | 1 | 2 | 3 |
| ----------------- | -------- | - | - | - |
| 0.05 | 24.0% | 30.6% | 14.0% | 33.6% |
| **0.25** (collection σ) | **52.1%** | **61.3%** | **53.8%** | **58.5%** |

**Spatial invariance** — does the anchor's own linear map hold `r` rad away:

| radius | anchor 0 | 1 | 2 | 3 |
| ------ | -------- | - | - | - |
| 0.25 | 28% | 41% | 30% | 31% |
| 1.00 | 66% | 71% | 64% | 82% |
| **4.00** | **175%** | **135%** | **127%** | **157%** |

Measured cluster radii are **4.34, 3.84, 3.73, 4.25 rad** — the bottom row. Above
100% means the local model is worse than predicting zero.

### The libraries themselves are excellent — inside 0.5 rad

`scripts/verify_libraries.py` is the gate any future library set should pass.
`skill` compares against a "the tip does not move" predictor; `cos` is the
direction of predicted vs true tip displacement. Full definitions, and what each
metric is load-bearing for, in [Reference › Metrics](../reference/metrics.md).

| radius | span | skill | cos | cos > 0.5 | cos < 0 |
| ------ | ---- | ----- | --- | --------- | ------- |
| 0.00 | 0.000 | 0.93 | 0.98 | 100% | 0% |
| 0.25 | 0.000 | 0.88 | 0.96 | 97% | 0% |
| 0.50 | 0.000 | 0.72 | 0.90 | 84% | 3% |
| 1.00 | 0.000 | 0.14 | 0.85 | 66% | 9% |
| 2.00 | 0.000 | **−9.93** | **−0.03** | 31% | **50%** |
| 4.00 | 0.000 | −23.64 | 0.07 | 28% | 44% |

Nothing is wrong with the data. At 2 rad the predictor points the **wrong way half
the time** — which is why a controller following it does worse than noise.

!!! warning "The span test is vacuous here, and that is itself a finding"
    Span residual is `0.000` at every radius, including 4 rad — not because the
    library is good, but because `n_cols = 1484` against `289` Hankel rows spans
    the entire space. Any trajectory is representable, so Willems' consistency
    constrains nothing and `λ_g` alone selects among infinitely many `g`. DeePC-GS
    deliberately runs 750–1000 columns and reports that slicing improves
    conditioning. **Shorter `T` is an untested knob** that may help independently
    of anchor radius.

### The gating experiment

Every closed-loop episode before this one started 2–4 rad from the nearest anchor
— the regime above. `scripts/test_valid_region_control.py` starts at a controlled
distance instead, with `du_max = 0.02` and a goal 0.4 rad away:

| start dist | | reached | median closed | median path/net |
| ---------- | - | ------- | ------------- | --------------- |
| 0.00 | DeePC | **3/3** | **87.9%** | **1.0** |
| 0.00 | random | 0/3 | 22.6% | 5.5 |
| 0.50 | DeePC | 2/3 | 84.6% | 1.3 |
| 0.50 | random | 0/3 | 13.0% | 7.6 |
| 2.00 | DeePC | 1/3 | 3.7% | 53.8 |
| 2.00 | random | 0/3 | 0.4% | 136.7 |

**Path/net = 1.0 is the number that matters** — the tip travels almost exactly the
distance it needs to. That is steering, not drifting. The same controller started
2–4 rad from an anchor ran 5–15×.

The usable radius is bracketed between 0.5 and 2.0 rad by closed-loop outcome,
agreeing with the open-loop table above. Two independent measurements, one
boundary.

### The measurement discipline that was missing

A random-walk control of equal command magnitude was not run until late, and when
it was, it **beat the unconstrained controller 38.5% to 6.1%**. Every
interpretation made before that point was against the wrong reference. This repo
already had the discipline — the env's random-action baseline exists so that "any
nonzero reach rate is signal, not noise" — and it was not applied to the new
interface. **Run the random control first on any new interface.**

### Anchor selection — the IK stage is the load-bearing step

`panda/anchors.py` samples task-relevant configurations, clusters with k-medoids
(medoids over means so each anchor is a configuration the robot can actually
adopt and start collection from), and reports a §5 diversity check.

Plan §3 samples Cartesian goals then solves IK back to `q`. The first attempt
here skipped that, using `sample_config` — which rejection-samples *uniformly*
from the safe box and takes the FK tip as the goal — and argued it was "more
faithful." **That was wrong, and the direction is the entire point.** IK is
one-to-many, so a *deterministic* IK policy collapses a 3-D goal space onto a ~3-D
manifold. FK from uniform configurations stays uniform.

`sample_task_configs(ik=...)` now implements the choice. Goals are drawn
identically in every mode, so `ik` is the only variable:

| `--ik` | silhouette | effective dimension |
| ------ | ---------- | ------------------- |
| `none` (FK-sampled) | 0.133 | 5.65 / 7 |
| **`home`** (plan §3) | **0.256** | **3.00 / 7** |
| `previous` | 0.195 | 3.62 / 7 |
| `random` (null control) | 0.123 | 5.73 / 7 |

`home` gives **exactly 3.00** — what a deterministic map from a 3-D goal space
into ℝ⁷ must produce. And the **null control passes**: seeding IK from a random
valid configuration reproduces the `none` result, so the concentration comes from
the policy rather than from the solver quietly biasing toward some pose. That
check is what makes the 3.00 trustworthy.

Under `ik=none`, `Q_task` matched a bare uniform box draw to within 2%
(`0.230 0.209 0.191 0.166 0.086 0.069 0.050` against `0.198 ×4, 0.084, 0.073,
0.053`), so k-medoids was clustering a box and every conclusion about cluster
structure was circular.

The arithmetic this moves — anchors to cover at the 0.5 rad usable radius scale
as `8^d`:

```
d = 5.65   (uniform)     ->  ~126,000 anchors
d = 3.00   (--ik home)   ->       512
```

Two orders of magnitude: from impossible to ~8.5 hours of collection.

!!! note "The seed policy is a design decision, not a solver detail"
    It determines which 3-D slice of the configuration space the robot is asked to
    live in. `home` gives a star around the keyframe; `previous` gives a
    continuous sheet that wanders off the manifold (3.62); `random` gives nothing.
    Choose it deliberately.

Joint-space distance *is* a sound metric once measured over a range that isn't
saturated — `corr(log distance, model difference) = 0.749`, 56% of variance. The
plan's §7 choice holds up; only the cell size is wrong. Silhouette at 0.256 is
still low, so even under IK this is a **manifold, not separated clusters**: `K`
remains a resolution choice rather than something the data hands you.

## Caveats

- **n = 3 per cell** in the gating experiment, n = 8 in the rate sweep, one goal
  scale, one seed. Effects are large and were predicted rather than discovered,
  but none of this is a confirmation run.
- Reaching real task goals needs 1–4 rad of travel. These episodes used 0.4 rad,
  so the arm never had to leave the valid bubble. **Whole-task reach rate under
  this controller is unmeasured**, and the coverage gap says it will be poor.
- `panda/qdes.py`'s `predict()` uses an ℓ2 solve while `core/deepc.py` uses ℓ1, so
  §8 prediction numbers are an upper bound on what the controller experiences.
- Clustering uses goal configurations only; configurations the arm *passes
  through* mid-episode get no anchor.

## Next

1. ~~**The IK stage**~~ — done; `--ik home` gives `d = 3.00`. The pipeline has
   been re-run on IK anchors (`data/panda_anchors_k4_ik.npz`).
2. **Select-DPC** — per-timestep column selection, no cells, effective radius
   always ~0. Reuses collection, output and the QP; replaces `build_libraries` +
   `assign`. Worth prototyping the *selection rule* against
   `verify_libraries.py` before building it: if pooled-column selection gives
   skill ~0.9 at arbitrary configurations the approach works, and if it degrades
   like the fixed libraries the real gap is collection coverage, which is a
   different project.
3. **Shorter `T`** — cheap, untested, and the span result says the Hankel is 5.1×
   over-parameterized.
4. **Whole-task reach rate** under `du_max = 0.02` on the 78-scenario set. Never
   measured; every full evaluation predates the rate limit. Expected poor for
   coverage reasons, but it is the reference number future changes compare
   against, and plan §9 asks for it.

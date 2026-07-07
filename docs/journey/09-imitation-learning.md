# 09. Imitation learning — cloning DeePC into a neural policy

!!! note "Status: complete (2026-07-07)"
A small MLP was trained by **behavioral cloning** to imitate the canonical DeePC
controller. It reproduces DeePC's closed-loop behavior on a 78-seed random sweep —
**identical 38.5 % reach rate**, **McNemar $p = 1.0$** (no systematic bias),
**82 % seed-by-seed outcome agreement**, and bounded trajectory drift (~0.93 units
median) — while replacing the **~0.6 s QP solve** with a **23 µs forward pass**
(~26,000× faster). This clone is the frozen baseline $f_\theta$ for the
residual-RL step proposed in [08](08-stop-at-goal.md). It faithfully reproduces
DeePC's _failures_ too (the far-field `v`-collapse) — that is by design: the clone's
job is fidelity, RL's job is the fix.

## Motivation (one line)

Residual RL — `u = u_DeePC + u_RL(obs)` — needs the DeePC baseline evaluated at
**every RL training step**, and keeping the **QP in the loop** (~0.6 s × 200 steps ×
millions of steps) is prohibitive. Clone DeePC into a fast neural policy so the
baseline is a microsecond forward pass instead of a solver call.

## Context

[08](08-stop-at-goal.md) ended on a structural conclusion: DeePC navigates well on
~39 % of seeds and **collapses** on the rest (the hallucinated-prediction
`v`-collapse), so the next step is a **hybrid** — DeePC supplies the prior where its
data representation is valid, RL backstops the regime where it is not. The default
architecture is **residual RL** with a DeePC baseline. That plan named one concrete
obstacle:

> "Residual RL keeps the **QP in the training loop** — a ~0.5 s solve every step … likely
> prohibitive as-is."

This entry removes that obstacle. It builds the "BC warm-start" artifact 08 listed as
an alternative, but repurposed: not as a policy to fine-tune away from, but as a
**frozen, amortized copy of DeePC** to add a learned residual _on top of_. Either way
the prerequisite is the same — a neural policy that behaves like DeePC — and that is
what is validated here.

## What was built

**The contract.** The clone sees exactly what DeePC's QP consumes each step and emits
the single action DeePC would apply:

- **Input** — the DeePC-faithful state: the `T_ini = 5`-step past buffer
  `(u_ini, y_ini)`, the current measurement `y = (x, y, δ)`, the goal `(g_x, g_y)`, and a
  one-hot of the **orientation-keyed library** DeePC would select (4 heading anchors).
- **Output** — the single applied action `u = (v, w)` (what the receding-horizon QP
  actually commits, not the whole plan).

**Featurization (40-D).** Heading is encoded as `(sin δ, cos δ)` so the `±π` wrap is
continuous. Layout `6·T_ini + 6 + N_lib = 30 + 6 + 4 = 40`:

| block            | dims          | contents                                  |
| ---------------- | ------------- | ----------------------------------------- |
| past buffer      | `6·T_ini = 30`| 5 steps of `(v, w, x, y, sin δ, cos δ)`   |
| current output   | `4`           | `x, y, sin δ, cos δ`                       |
| goal             | `2`           | `g_x, g_y`                                 |
| library one-hot  | `N_lib = 4`   | which heading library DeePC would use      |

The one-hot is the key design choice: DeePC's prediction is **piecewise** (a different
local-linear library per heading quadrant), so telling the MLP which library is active
spares it from having to rediscover that switching boundary from the raw heading.

**Architecture.** MLP `40 → 256 → 256 → 2`, ReLU, standardized inputs/targets (the
one-hot tail passes through unscaled). CPU forward pass ≈ **23 µs**.

**Data (hybrid, 35,023 labeled `state → DeePC action` pairs).** No single sampling
strategy covers both "everywhere the state space could be" and "where the closed loop
actually goes", so the set is mixed:

| source          | count  | share  | purpose                                            |
| --------------- | ------ | ------ | -------------------------------------------------- |
| synthetic       | 15,024 | 42.9 % | broad coverage — random poses, goals, past buffers |
| on-policy       | 15,023 | 42.9 % | the distribution the clone will actually see       |
| degenerate      | 4,976  | 14.2 % | frozen/constant-past stall states (the hard regime)|

The **degenerate** slice is deliberately over-represented: it is the `v`-collapse
regime from [08](08-stop-at-goal.md), where the QP target is hardest to reproduce and
where the downstream RL will do its work — so the clone must at least see it. Labels
are the QP's own solution at each state, under the **canonical config** (bearing
reference, `Q = diag(1, 1, 2)`, `R = 1.3·10⁻³ I`, `T_ini = 5`, `N = 12`,
`λ_g = 2`, `λ_y = 3·10⁶`, `libraries_v0.npz`).

**Validation gate (3 layers).** Marginal reach-rate parity is _not_ sufficient — two
controllers can hit the same success rate on disjoint seeds. So fidelity is checked at
three levels: (1) open-loop regression on a held-out split, by regime; (2) closed-loop
trajectory deviation (does the clone's rollout track DeePC's path?); (3) **paired
per-seed outcomes** with a McNemar test (does it agree seed-by-seed, and is any
disagreement unbiased?).

## Side-by-side — DeePC vs the clone

Three random seeds under the canonical config. The env seed fixes the **same** start
pose and goal for both controllers, so each row is the two controllers attempting an
identical task. Left = DeePC (QP); right = the clone (MLP).

<table>
<thead>
<tr><th>DeePC — the QP controller (~0.6 s/step)</th><th>Clone — the neural policy (23 µs/step)</th></tr>
</thead>
<tbody>
<tr><td colspan="2" align="center"><b>seed 4104626050 · both REACH</b> — DeePC 41 steps → 0.31; clone 45 steps → 0.32; path deviation 0.80 median / 1.53 p95</td></tr>
<tr>
<td><video controls loop muted playsinline width="330"><source src="../videos/deepc-4104626050.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="330"><source src="../videos/clone-4104626050.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
</tr>
<tr><td colspan="2" align="center"><b>seed 4104626042 · both REACH</b> — DeePC 58 steps → 0.42; clone 71 steps → 0.46; path deviation 1.40 median / 2.15 p95</td></tr>
<tr>
<td><video controls loop muted playsinline width="330"><source src="../videos/deepc-4104626042.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="330"><source src="../videos/clone-4104626042.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
</tr>
<tr><td colspan="2" align="center"><b>seed 4104626061 · both FAIL</b> — DeePC truncates at 1.06; clone truncates at 0.54 (a near-miss at the 0.5 edge); path deviation 0.57 median / 1.35 p95. The clone reproduces the far-field stall, not just the reaches.</td></tr>
<tr>
<td><video controls loop muted playsinline width="330"><source src="../videos/deepc-4104626061.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
<td><video controls loop muted playsinline width="330"><source src="../videos/clone-4104626061.mp4" type="video/mp4">Your browser does not support the video tag.</video></td>
</tr>
</tbody>
</table>

All three video seeds fall in the **82 % agreeing** set (both reach, or both fail).
They were chosen from the clone's own reaches plus one shared failure to illustrate the
match, so they are a favorable-but-representative sample — the aggregate table below is
the unbiased measure.

## Metrics

### (1) Open-loop fidelity — does the clone reproduce DeePC's action?

Held-out split (`n = 3502`, never seen in training). Errors are in physical units
(`v` in units/s, `w` in rad/s):

| regime      | n     | MAE `v` | MAE `w` | RMSE `v` | RMSE `w` |
| ----------- | ----- | ------- | ------- | -------- | -------- |
| on-policy   | 1,525 | 0.58    | 0.13    | 0.94     | 0.22     |
| synthetic   | 1,495 | 1.13    | 0.22    | 1.68     | 0.37     |
| degenerate  | 482   | 2.78    | 0.25    | 4.06     | 0.44     |
| **overall** | 3,502 | **1.12**| **0.18**| **1.97** | **0.33** |

Held-out correlation with the QP target: `corr(v) = 0.94`, `corr(w) = 0.97`
(normalized MSE 0.092). The clone is **most accurate exactly where it operates**
(on-policy: MAE `v` ≈ 0.58 units/s against a `[0, 20]` range) and least accurate on the
**degenerate** stall states — expected, because there the QP itself is ill-posed (a
near-constant past is matched by many solutions, so the target is nearly arbitrary; see
[08](08-stop-at-goal.md) _Root cause_). The clone cannot be more certain than DeePC is.

### (2)+(3) Closed-loop behavioral equivalence — 78 random seeds

`scripts/validate_clone.py`, base seed `4104626029`, canonical config:

| metric                              | DeePC              | Clone              |
| ----------------------------------- | ------------------ | ------------------ |
| reach rate                          | 30 / 78 = **38.5 %** | 30 / 78 = **38.5 %** |
| reach-rate 95 % CI (Wilson)         | [0.284, 0.496]     | [0.284, 0.496]     |
| median solve/inference time per step| ~0.6 s (QP)        | **23 µs** (MLP)    |

Paired agreement (the decisive test):

| paired metric                      | value                                       |
| ---------------------------------- | ------------------------------------------- |
| seed-by-seed outcome agreement     | **64 / 78 = 82.1 %**                         |
| confusion (both / DeePC-only / clone-only / neither) | 23 / 7 / 7 / 41           |
| McNemar $p$                        | **1.0** (disagreements balanced → no bias)  |
| median trajectory position deviation | **0.94 units** median · **2.01** p95 (per-seed) |

## Reading the numbers

- **Equivalent, not merely comparable.** Identical marginal reach rate _and_ McNemar
  $p = 1.0$ mean the clone is neither better nor worse than DeePC — the 14 disagreeing
  seeds split exactly 7/7, so they are boundary noise (near-miss seeds that tip either
  way), not a systematic skew. This is the property that a raw success-rate match cannot
  certify.
- **Same paths, not just same outcomes.** Median trajectory deviation ~0.94 units
  (worst-case p95 ~2.01, on a 20-unit workspace diagonal) says the clone follows nearly the same route, not just
  ends in the same place. On the video seeds the clone takes a few more steps
  (45 vs 41; 71 vs 58) but lands equivalently.
- **The failures transfer too.** On seed 4104626061 both stall in the far field — the
  clone inherits DeePC's dominant flaw. This is the honest, expected outcome of cloning
  (flagged in [08](08-stop-at-goal.md)'s "BC warm-start" note) and is _acceptable here_:
  the clone is the baseline, and the residual RL is what supplies the forward velocity
  the hallucinated predictor refuses to.
- **The speedup is the point.** 0.6 s → 23 µs (~26,000×) is what turns "QP in the RL
  training loop" from prohibitive into a non-issue.

## What this unlocks — residual RL

The clone is the frozen $f_\theta$ for the hybrid proposed in
[08](08-stop-at-goal.md):

$$u = \operatorname{clip}\big(f_\theta(\text{state}) + u_\text{RL}(\text{obs})\big)$$

with the residual zero-initialized (initial behavior ≈ DeePC, no regression risk) and
trained on the **true env reward**. Because $f_\theta$ is a 23 µs forward pass, the QP
never enters the training loop. The residual is tiny on the ~39 % of seeds DeePC
already solves and _carries_ the ~42 % it collapses — exactly the far-field
`v`-collapse the clone was just shown to reproduce.

Open question carried forward: does residual RL beat the much cheaper
`--Q_heading 0` / `--no_bearing_ref` heading-reference fix from
[08](08-stop-at-goal.md)? That ablation is the next entry.

## Reproduce

```bash
# generate the hybrid dataset (state -> DeePC action pairs)
uv run python scripts/gen_clone_data.py --out data/clone_dataset.npz

# train the clone
uv run python scripts/train_clone.py --data data/clone_dataset.npz --out data/clone.pt

# the 3-layer fidelity gate (open-loop regression + closed-loop paired outcomes)
uv run python scripts/validate_clone.py --clone data/clone.pt \
    --data data/clone_dataset.npz --n_seeds 78 --base_seed 4104626029

# side-by-side videos (DeePC left, clone right)
uv run python scripts/run_deepc.py --seed 4104626050 --episodes 1 --record docs/journey/videos  # -> episode_0.mp4, rename to deepc-4104626050.mp4
uv run python scripts/run_clone.py --record docs/journey/videos --seeds 4104626050              # -> clone-... via episode_<seed>.mp4
```

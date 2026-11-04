# 21. Quadruped DeePC — a longer window lifts the Go1 from 22/78 to 60/78

## Decision

The DeePC baseline for `QuadrupedGoal-v0` is **plain DeePC with `T_ini = 20`,
`N = 25`**. Everything else is copied from the unicycle: four heading-keyed
libraries, a bearing reference, `λ_g = 2`, `λ_y = 3e6`, and SCS. It reaches
**60/78 (77%, Wilson [0.66, 0.85])** of the held-out scenarios, with no falls and
no QP failures. The interval excludes 30/78, the unicycle DeePC's 38.5%, so the
legged plant ends up with a *stronger* classical baseline than the plant it was
copied from.

The straight port of the unicycle's `T_ini = 5, N = 12` reaches **22/78 (28%,
[0.19, 0.39])**. Only one tuning setting was run, because it cleared the bar.
This baseline is "good enough to build the clone and residual on", not "the best
DeePC the Go1 can have". The price is solve time: **19 s per step** under a
10-worker load, against 3.7 s for the straight port.

## Context — the same pipeline, a legged plant

`QuadrupedGoal-v0` puts a Unitree Go1 (MuJoCo) with a hand-built open-loop trot
under a unicycle-style goal-reaching task. The agent commands a body-frame
velocity `u = (vx, vy, yaw_rate)` and observes `y = (x, y, yaw)`. The question is
the one [entry 11](11-panda-anchors.md) asked of the Panda: does the DeePC → clone
→ residual pipeline carry over? The first stage needs a DeePC that works.

The output and the cost already mirror the unicycle, so the target was "mirror
the unicycle setup, beat the 39%". Five plant facts forced changes on the way:

| Plant fact | What it forced |
|---|---|
| Backward `vx` with fast command changes makes the robot fall (per-step random commands fall 6/20) | Collection holds each command for 0.5 s plus small noise. The controller gets a rate limit `du_max = (hi − lo)/25` per step |
| The gait adds a constant ≈ −0.09 m/s to every `vx` command, so the input → velocity map is affine | Added an opt-in `1ᵀg = 1` constraint (`DeePC(affine=True)`). The baseline does not use it |
| Yaw is wrapped, and closed-loop runs cross ±π | `_YawBranch` unwraps the current yaw, the past buffer and the bearing onto the selected library's branch before the QP sees them |
| Yaw rate oscillates within each 0.4 s gait cycle | Only cycle averages are meaningful. See the horizon discussion below |
| Response lag of about 0.3 s (table below) | Sets the scale of `T_ini` and `N` |

### The gait

DeePC never sees joints: the plant it identifies is "command in, trunk pose
out", with the trot below inside it (`quadruped/gait.py::TrotGait`). Feet are
in the trunk frame, relative to each leg's hip-abduction joint. Constants:
period $T = 0.4$ s, $\Delta t = 0.02$ s, standing height $h = 0.27$ m, swing
lift $H = 0.07$ m, side $s_i = \pm 1$ (+1 = left), abduction offset
$l_\text{ab} = 0.08$ m, thigh and calf lengths $a = b = 0.213$ m.

**Phase clock.** One global phase. Each leg gets an offset
$o = (0.5, 0, 0, 0.5)$ for (FR, FL, RR, RL), so the diagonal pairs FL+RR and
FR+RL alternate:

$$
\phi_{t+1} = (\phi_t + \Delta t / T) \bmod 1, \qquad
\varphi_i = (\phi_{t+1} + o_i) \bmod 1, \qquad
\text{leg } i \text{ in stance} \iff \varphi_i < 0.5
$$

**Foot velocity.** The command $u = (v_x, v_y, \omega)$ is turned into the
planar velocity of a rigid body at each leg's nominal foot point
$r_i = (r_{i,x}, r_{i,y})$ in the trunk frame:

$$
u_i = \begin{pmatrix} v_x - \omega\, r_{i,y} \\ v_y + \omega\, r_{i,x} \end{pmatrix}
$$

**Stance** ($\varphi_i < 0.5$). The foot slides backward at $u_i$ on the
ground plane. If it does not slip, the trunk moves at $u_i$:

$$
p_i^{xy} \leftarrow p_i^{xy} - u_i \Delta t, \qquad p_i^z = -h
$$

**Swing** ($\varphi_i \ge 0.5$). Let $\sigma = 2(\varphi_i - 0.5) \in [0, 1)$
be the swing progress and $p_i^\text{lift}$ the foot position at liftoff. The
foot follows a cosine-eased path to a landing point a quarter period of travel
ahead of its nominal point $n_i = (0, s_i l_\text{ab})$, with a sine arc in
height:

$$
\ell_i = n_i + u_i \tfrac{T}{4}, \qquad
e(\sigma) = \tfrac{1}{2}\big(1 - \cos \pi\sigma\big)
$$

$$
p_i^{xy} = p_i^\text{lift} + \big(\ell_i - p_i^\text{lift}\big)\, e(\sigma), \qquad
p_i^z = -h + H \sin \pi\sigma
$$

Stance lasts $T/2$ and covers $u_i T/2$. Landing at $+u_i T/4$ therefore lifts
off at $-u_i T/4$: the stance is centred on the nominal point. The
foot-placement feedback term (`k_foot`) is off ($k_\text{foot} = 0$), so the
gait is fully open-loop.

**Leg IK.** For each foot target $p = (x, y, z)$, with $l_1 = s_i l_\text{ab}$:

$$
L = \sqrt{y^2 + z^2 - l_1^2}, \qquad
q_\text{ab} = \operatorname{atan2}(z, y) - \operatorname{atan2}(-L, l_1)
$$

$$
r = \min\!\Big(\sqrt{x^2 + L^2},\ a + b\Big), \qquad
q_\text{knee} = -\Big(\pi - \arccos \frac{a^2 + b^2 - r^2}{2ab}\Big)
$$

$$
q_\text{thigh} = \operatorname{atan2}(-x, L) - \operatorname{atan2}\big(b \sin q_\text{knee},\ a + b \cos q_\text{knee}\big)
$$

The $\min$ projects unreachable targets onto the workspace boundary. The joint
targets $q_i = \operatorname{clip}(q, q_\text{lo}, q_\text{hi})$ are
`info["ctrl"]`, and MuJoCo's PD-servo actuators track them. The −0.09 m/s
$v_x$ offset and the in-cycle yaw oscillation are what this ideal no-slip
model leaves out. Both are measured in the table above, not predicted by the
equations.

### Data

There are four libraries, keyed at headings `±π/4` and `±3π/4`. Each is one
1500-step trajectory (30 s), so the robot walks for **2 minutes** in total. No
attempt fell and no trajectory wrapped yaw. They are stored in
`data/quadruped_libraries_v0.npz`.

The random excitation does not keep a library near its anchor. The spec asked
for that check; it was only done after the baseline was chosen:

| Library | Anchor | Share of samples within π/4 of anchor | Yaw range |
|---|---|---|---|
| 0 | π/4 | 0.95 | [−0.28, 1.16] |
| 1 | 3π/4 | **0.16** | [−1.37, 2.32] |
| 2 | −3π/4 | 0.98 | [−3.03, −1.40] |
| 3 | −π/4 | 0.73 | [−0.91, 0.78] |

Library 1 drifted toward heading 0 and has no data above yaw 2.32.

### Lag

This measures the time from a command step to 63% of the measured velocity
change. "Corrected" subtracts roughly 0.2 s, the delay the 20-step trailing
average adds. That delay depends on the response shape, so the corrected column
is ±0.05 s.

| axis | command | raw 63% | corrected | gain |
|---|---|---|---|---|
| vx | 0 → 0.3 | 0.34 s | 0.14 s | 0.67 |
| vx | 0.3 → 0.6 | 0.34 s | 0.14 s | 0.85 |
| vx | 0 → −0.2 | 0.32 s | 0.12 s | 1.37 |
| vy | 0 → ±0.2 | 0.34–0.36 s | 0.14–0.16 s | 1.07–1.10 |
| yaw | 0 → ±0.8 | 0.28–0.30 s | 0.08–0.10 s | 0.91–0.92 |
| yaw | 0.4 → −0.4 | 0.30 s | 0.10 s | 0.83 |

The spread in the vx gains is the affine offset. No run fell.

## Considered — a tuning ladder, cut short

The plan was a ladder, numbered as in the spec and the data paths:

0. the straight port;
1. `T_ini × N`;
2. the regularisers;
3. the affine constraint;
4. more data;
5. the clonable no-carry Select-DPC.

It used 40 tuning scenarios plus 78 disjoint evaluation scenarios
(`data/quadruped_scenarios_v1.npz`, replayed exactly via `reset(options=…)`).
Three rules kept the evaluation set honest:

- at most 3 evaluation looks, each logged in `data/quadruped_deepc/eval_looks.csv`;
- a change counts only if it adds at least 3 tuning reaches;
- evaluation look 2 runs the first time a tuning result exceeds 15/40.

Rung 1's grid was ordered by the lag: `T_ini = 20` first, because its 0.4 s spans
one gait cycle and more than the ~0.3 s raw lag, then `N` of 25 or 50 before 12. The first cell cleared
the bar, its evaluation beat the target, and the ladder stopped there, as the
spec requires.

| Setting | Split | Reached | Wilson 95% | Timeouts < 1 m / 1–2.5 m / > 2.5 m | Steps to reach | Solve / step |
|---|---|---|---|---|---|---|
| `T_ini=5, N=12` (straight port) | tune | 14/40 (35%) | [0.22, 0.50] | 16 / 2 / 8 | 267.1 | 3.6 s |
| `T_ini=5, N=12` (straight port) | **eval, look 1** | 22/78 (28%) | [0.19, 0.39] | 36 / 4 / 16 | 311.0 | 3.7 s |
| `T_ini=20, N=25` | tune | 28/40 (70%) | [0.55, 0.82] | 4 / 1 / 7 | 240.9 | 16.9 s |
| `T_ini=20, N=25` | **eval, look 2** | **60/78 (77%)** | **[0.66, 0.85]** | 3 / 1 / 14 | 229.0 | 19.1 s |

Every row has 0 falls and 0 QP failures. Solve times are means under 10
concurrent workers on an M4 Pro. The evaluation of the winner took about 9 h of
wall-clock time.

On the 78 evaluation scenarios the longer window **gains 38 and loses none**.

**Not run:** the other seven `T_ini × N` cells, `(20, 50)`, `(10, 25)`,
`(10, 50)`, `(20, 12)`, `(10, 12)`, `(5, 25)` and `(5, 50)`; and rungs 2–5. One evaluation look is unused.

## Outcome

- **The straight port's failure was mostly stalling.** 36 of its 56 evaluation
  timeouts ended within 1 m of the goal. The longer window removes almost all of
  them (36 → 3).
- **Why it works is a hypothesis, not a measurement.** At 50 Hz, `T_ini = 5` is
  0.1 s of past. That is shorter than the raw lag (~0.3 s) and a quarter of a
  gait cycle, so the past window sits inside the in-cycle yaw oscillation and
  cannot pin down the state. `N = 12` (0.24 s) is shorter than the raw lag, so
  the cost sees little of what the first input does. `T_ini = 20` (0.4 s) covers
  one gait cycle and `N = 25` (0.5 s) about 1.5 raw lags. T_ini and N were
  changed together, so their effects are not separated.
- **What's left is walking away, and it lines up with library 1's coverage.**
  14 of the 18 remaining misses end more than 2.5 m from the goal (the straight
  port had 16). Grouping the evaluation scenarios by which library's region the
  start→goal bearing falls in:

  | Bearing region | Straight port reached | Baseline reached | Baseline far misses |
  |---|---|---|---|
  | lib 0 (π/4) | 7/16 | 15/16 | 0 |
  | lib 1 (3π/4) | 1/18 | **7/18** | 8 |
  | lib 2 (−3π/4) | 13/29 | 23/29 | 6 |
  | lib 3 (−π/4) | 1/15 | 15/15 | 0 |

  Outside library 1's region the baseline reaches 53/60. All 7 tuning-set
  walk-aways are in library 1's region too. This groups by the initial bearing,
  not by the library active at each step, so it is a correlation, not a
  confirmed mechanism. Re-collecting library 1 so it stays near 3π/4 (rung 4)
  is the obvious next lever, ahead of the residual stage.
- **Solve cost is the clone's problem now.** At about 19 s per step, the 400-step
  episodes the clone stage needs as labels are expensive to generate. In a
  one-off check during the runner's smoke test (not scripted), SCS took
  1650–11375 iterations per solve even at 5/12, with or without the rate limit.
  Loosening the SCS tolerance is the untried first lever.
- **Code:**
  - `quadruped/deepc_setup.py`: `build_canonical_quadruped_deepc` defaults to
    20/25.
  - `scripts/run_quadruped_deepc.py`: the resumable parallel runner, defaults
    also 20/25. It stores the resolved config in `<out>.cfg.json` and refuses
    to resume a file made with a different config, or to touch a file it
    cannot parse.
  - `quadruped/scenarios.py`: frozen scenarios.
  - `quadruped/data_collection.py`: the libraries.
  - `scripts/measure_quadruped_lag.py`: the lag table.
  - `scripts/quadruped_library_coverage.py`: the coverage and bearing-region
    tables.
  - `core/deepc.py`: gains the opt-in `affine` constraint. Default behaviour is
    unchanged, pinned by a golden test.

### Re-measure

Every evaluation run is logged as a look in `data/quadruped_deepc/eval_looks.csv`.

```bash
.venv/bin/python scripts/make_quadruped_scenarios.py      # frozen scenarios; checksum pinned in tests
.venv/bin/python scripts/collect_quadruped_data.py --out <npz>   # libraries (default --out overwrites the canonical file)
.venv/bin/python scripts/measure_quadruped_lag.py          # lag table
.venv/bin/python scripts/quadruped_library_coverage.py --csv data/quadruped_deepc/rung1_t20n25_eval.csv   # coverage + bearing regions
# straight port
.venv/bin/python scripts/run_quadruped_deepc.py --split tune --T-ini 5 --N 12 --out <csv>
.venv/bin/python scripts/run_quadruped_deepc.py --split eval --T-ini 5 --N 12 --out <csv>
# baseline (current defaults)
.venv/bin/python scripts/run_quadruped_deepc.py --split tune --T-ini 20 --N 25 --out <csv>
.venv/bin/python scripts/run_quadruped_deepc.py --split eval --T-ini 20 --N 25 --out <csv>
```

# DeePC

The `DeePC` controller implements the regularized hybrid form from [arXiv:2603.07395](https://arxiv.org/abs/2603.07395), built on the original DeePC framework from [arXiv:1811.05890](https://arxiv.org/abs/1811.05890).

## Signals — the control input `u` and the output `y`

`core.deepc.DeePC` is system-agnostic: it never sees a state, a goal, or an env. It sees two time series, `u` and `y`, and its entire model of the plant is the offline data compiled into the Hankel matrices. Everything below is what those two symbols mean.

### `u` — the control input, dimension `m_u`

`act()` returns `u_t` of shape `(m_u,)`. It is the **first block** of the `N`-step plan `U_f g`; the remaining `N − 1` blocks are computed and then discarded, and the QP is re-solved next step. Receding horizon — plan `N`, apply one.

`m_u` is never declared. It is derived as `Up.shape[0] // T_ini` from the first library, and every other library must agree, since they share one set of Hankel parameters. `R` must then be `(m_u, m_u)` or construction raises.

Two bounds act on `u`, and they are not the same thing:

| | what it limits | default |
|---|---|---|
| `u_bounds` | **where** `u` may sit — applied to all `N` steps of the plan, then re-clipped on return, since solver tolerance can leave sub-microscopic violations | `None` |
| `du_max` | **how far `u` may move in one step**, against the last applied input | `None` |

Omitting `du_max` is harmless when `u` is a velocity, and not harmless when `u` is an absolute position target: measured on `PandaReach-v0`, the QP asked for a median **1.9–3.1 rad of movement per 20 ms step**, far outside the neighbourhood its library was collected in, making every command an extrapolation of a locally-valid model.

!!! warning "`u` must be the plant's *true* input, not the command you issued"
    Where the plant clips or transforms a command, identifying on the pre-transform signal mislabels every affected sample. On `PandaReach-v0` the safe-box clip fires on **24–48 % of steps** under excitation, and the repo carries **two incompatible `u` conventions** because of it:

    | path | `u` is | reaches the plant via |
    |---|---|---|
    | anchor/Hankel DeePC (`panda/deepc_setup.py`) | the **delta** `Δq ∈ [−0.2, 0.2]⁷` | `env.step(u)` → `ctrl = clip(q + u, box)` |
    | Select-DPC (`panda/selectdpc.py`, `panda/qdes.py`) | the **absolute target** `q_des` | `step_qdes` → `ctrl = clip(q_des, box)`, never through `env.step()` |

    Both hardcode the key `u_i` in the collection payload with opposite units — a ~0.2 rad delta against an absolute angle spanning the whole safe box — so one flat dict cannot serve both. `panda/task_bank.py::for_select_dpc` exists solely to re-key for the second; read its module docstring before feeding either consumer a payload.

### `y` — the output, dimension `p_y`

`act(y_current, y_ref)` takes `y_current` of shape `(p_y,)`, with `p_y = Yp.shape[0] // T_ini` derived the same way as `m_u`. `Q` must be `(p_y, p_y)`.

`y` does **three separate jobs**, worth keeping apart because a component can do some and not others:

1. **Selects the library.** `_select_index_for` keys on `y[heading_index]` (default index 2), or on `key_fn(y)` when the keying quantity is a *function* of `y` rather than a component of it — under `key_fn` the key need not appear in `y` at all.
2. **Updates the past buffer.** After the solve, `y_current` is appended to `y_buf`, becoming part of the next step's `y_ini` in the soft constraint `Y_p g + \sigma_y = y_{\text{ini}}`.
3. **Is what gets tracked**, through `\lVert Y_f g - y_{\text{ref}} \rVert^2_{\bar Q}`.

**Job 3 is optional per component.** Zeroing a block of `Q` keeps that block doing jobs 1 and 2 — informing prediction through the `Y_p`/`Y_f` constraints — while removing it from the objective. Both non-unicycle systems use this deliberately: the Reacher's `Q = diag(0, 0, 1, 1)` tracks the fingertip while the joint block makes the state observable, and the Panda's `Q = diag(I₃, 0₇)` makes the 10-D extended output's tracking cost *numerically identical* to the tip-only case.

It is also why `y` carries the tip/fingertip **position** rather than a scalar goal-distance: that keeps the libraries **goal-free**, so one Hankel build serves every target and the goal enters only through `y_ref`.

`y_ref` is either `(p_y,)`, broadcast across the `N`-step horizon, or `(N, p_y)` for a per-step reference.

### Alignment

`y_t` is recorded **before** `u_t` is applied, so `y_{t+1}` is the response to `u_t`. Every collection in this repo uses that convention. A library built the other way is off by one step, and nothing downstream catches it — the QP will happily solve a wrong model.

### What each system plugs in

| | `TwoWheelGoal-v0` | `PandaReach-v0` (`output="ext"`) | Reacher-v5 |
|---|---|---|---|
| `u` | `(v, w)` velocities, `m_u = 2` | `Δq` delta, `m_u = 7` | `τ` torque, `m_u = 2` |
| `u_bounds` | `v ∈ [0, 20]`, `w ∈ [±π/2]` | `±0.2` per joint | `±1` per joint |
| `y` | `(x, y, δ)`, `p_y = 3` — **is** the full state | `(tip, q_norm)`, `p_y = 10` | `[q; fingertip]`, `p_y = 4` |
| `Q` | `diag(1, 1, 2)` | `diag(I₃, 0₇)` | `diag(0, 0, 1, 1)` |
| `R` | `1.3e-3 · I₂` | `1.0e-2 · I₇` | `1.0e-3 · I₂` |
| Library key | `y[2]`, the heading | `key_fn` = tip azimuth `atan2(y[1], y[0])` | overridden: wrapped `config_distance` on `y[:2]` |
| `du_max` | `None` — `u` is a velocity | `None` in the canonical builder | `None` — torque is natively bounded |

The unicycle is the only one of the three whose `y` observes the whole state; that asymmetry, not nonlinearity, is the main structural difference between the systems. See the [Reacher reference](../reference/reacher.md) and `panda/env.py` for the measurements behind it.

## The QP solved every step

$$
\min_{g,\ \sigma_y} \quad \lVert Y_f g - y_{\text{ref}} \rVert^2_{\bar Q} + \lVert U_f g \rVert^2_{\bar R} + \lambda_g \lVert g \rVert_1 + \lambda_y \lVert \sigma_y \rVert^2_2
$$

subject to

$$
\begin{aligned}
U_p \cdot g &= u_{\text{ini}} \\
Y_p \cdot g + \sigma_y &= y_{\text{ini}} \\
u_{\min} \le U_f \cdot g &\le u_{\max}
\end{aligned}
$$

with $\bar Q = I_N \otimes Q$, $\bar R = I_N \otimes R$.

### Breakdown of each term

| Term | What it does |
|---|---|
| $\lVert Y_f g - y_{\text{ref}} \rVert^2_{\bar Q}$ | Tracking cost: pull predicted future $y$ toward the reference. |
| $\lVert U_f g \rVert^2_{\bar R}$ | Control effort cost. |
| $\lambda_g \lVert g \rVert_1$ | **L1 sparsity on `g`** — the controller picks a small subset of past trajectories ("important" ones). |
| $\lambda_y \lVert \sigma_y \rVert^2_2$ | **L2 penalty on the past-output slack** — keeps the past prediction near the actual past, but allows mismatch under data inconsistency. |

The L1 on `g` is from the original Coulson/Lygeros/Dörfler 2019 paper; the L2 on $\sigma_y$ matches arXiv:2603.07395's specific choice. See [DeePC formulation journey entry](../journey/04-deepc-formulation.md) for the rationale.

### Defaults

| Parameter | Default | Source |
|---|---|---|
| `T_ini` | 5 | Paper |
| `N` | 12 | Paper's largest `W` |
| `lambda_g` | 2.0 | Paper |
| `lambda_y` | 3·10⁶ | Paper |
| `Q` | `diag(1, 1, 2)` (from env) | Paper |
| `R` | `1.3·10⁻³ · I₂` (from env) | Paper |

## Past-trajectory buffer

The controller maintains a sliding window of length `T_ini`:

```
u_buf : shape (T_ini, m_u)  — last T_ini applied actions
y_buf : shape (T_ini, p_y)  — last T_ini observed outputs (before each action)
```

On `reset(y_initial, u_initial)`, both are filled with `T_ini` copies of the initial values. This "synthetic past" is consistent for any zero-velocity initial condition, and the soft `λ_y ‖σ_y‖²` penalty absorbs the inconsistency for other cases.

!!! warning "Cold-start gotcha"
    If `u_initial` is `0` and the data has only non-negative `v` (e.g., paper-PE collection or broad bounds with `v ≥ 0`), the QP gets locked into outputting `u ≈ 0`. The fix is to prime `u_initial` at the **midpoint of `action_bounds`** instead. `scripts/run_deepc.py` does this automatically.

## Libraries and orientation switching

`DeePC` holds one or more orientation-keyed data libraries. The constructor
takes `libraries` (a list of `(Up, Uf, Yp, Yf)` Hankel tuples, e.g. from
`build_hankel`) and a parallel `anchor_headings` array. Each step it selects the
library whose anchor is closest to the robot's heading (read from
`y_current[heading_index]`, default index 2) and feeds that library into the QP.

A single library (`len(libraries) == 1`) is the plain single-library DeePC case:
selection is trivially index 0 and `anchor_headings` / `heading_index` are not
consulted. All libraries must share the same column count (`n_cols`), since they
feed one shared set of Hankel parameters. See the [library switching
reference](library-switching.md) for the switching details, anchor table, and
the `last_library_idx` / `last_warm_started` diagnostics.

## Caching: DPP-compliant problem

The CVXPY problem is built once at construction with `cp.Parameter`s for the
active library's Hankel matrices (`Up`, `Uf`, `Yp`, `Yf`) and for `u_ini`,
`y_ini`, and `y_ref`. The cost is written as `cp.sum_squares(Q_sqrt @ y_err)`
rather than `cp.quad_form(y_err, Q)` to keep the problem DPP-compliant — and
`sum_squares(F @ g)` with a parameter `F` is itself DPP, which is what lets the
Hankel matrices be parameters. CVXPY caches the compiled solver and subsequent
calls reuse it, so switching libraries swaps parameter values rather than
recompiling. On a library switch the warm-start `g` is cleared, since its
columns indexed the previous library. Per-step solve times are dominated by the
QP solver rather than CVXPY's canonicalization.

## Solver

At the paper's `λ_y = 3·10⁶`, the QP is ill-conditioned (the slack penalty
dominates the Hessian). cvxpy's incidental default for this problem class is
OSQP, whose iteration cap is too low — it returns `user_limit` and the solve
fails; CLARABEL's interior-point factorization can break down numerically
(`SolverError`) on some states. The controller therefore defaults to
**`solver="SCS"`** (a first-order, equilibrated solver that is robust across
operating points). Pass `solver` and `solver_opts` to override — e.g.
`solver="CLARABEL", solver_opts={"max_iter": 50000}` for higher accuracy where
the problem is well-conditioned. A hard `cvxpy.SolverError` is caught and
re-raised as the controller's `RuntimeError`, so a solver breakdown is reported
like any other QP failure rather than crashing the caller.

## Public API

::: core.deepc.DeePC
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - __init__
        - reset
        - act

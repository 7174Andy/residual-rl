# DeePC

The `DeePC` controller implements the regularized hybrid form from [arXiv:2603.07395](https://arxiv.org/abs/2603.07395), built on the original DeePC framework from [arXiv:1811.05890](https://arxiv.org/abs/1811.05890).

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
| `Q` | `diag(1, 1, 0)` (from env) | Paper |
| `R` | `1.3·10⁻³ · I₂` (from env) | Paper |

## Past-trajectory buffer

The controller maintains a sliding window of length `T_ini`:

```
u_buf : shape (T_ini, m_u)  — last T_ini applied actions
y_buf : shape (T_ini, p_y)  — last T_ini observed outputs (before each action)
```

On `reset(y_initial, u_initial)`, both are filled with `T_ini` copies of the initial values. This "synthetic past" is consistent for any zero-velocity initial condition, and the soft `λ_y ‖σ_y‖²` penalty absorbs the inconsistency for other cases.

!!! warning "Cold-start gotcha"
    If `u_initial` is `0` and the data has only non-negative `v` (e.g., paper-PE collection or broad bounds with `v ≥ 0`), the QP gets locked into outputting `u ≈ 0`. The fix is to prime `u_initial` at the **midpoint of `action_bounds`** instead. `scripts/run_deepc.py` does this automatically. See [cold-start journey entry](../journey/05-cold-start.md).

## Caching: DPP-compliant problem

The CVXPY problem is built once at construction with `cp.Parameter`s for `u_ini`, `y_ini`, and `y_ref`. The cost is written as `cp.sum_squares(Q_sqrt @ y_err)` rather than `cp.quad_form(y_err, Q)` to keep the problem DPP-compliant — CVXPY caches the compiled solver and subsequent calls reuse it. Per-step solve times are dominated by the QP solver rather than CVXPY's canonicalization.

## Public API

::: two_wheel_robot.controllers.deepc.DeePC
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - __init__
        - reset
        - act

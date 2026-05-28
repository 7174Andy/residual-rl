# 05. Cold-start bug — when the synthetic past locks the controller

## Decision

On `reset()`, prime the past-action buffer at the **midpoint of `action_bounds`**, not zeros.

## Context

`DeePC.reset(y_initial)` initializes the past-trajectory buffer with `T_ini` copies of `(u = 0, y = y_initial)`. This is a "synthetic past" — the robot hasn't actually moved, but the controller needs *some* past trajectory to satisfy the hard constraint `Up · g = u_ini`.

For most systems this works fine. For our env with broad bounds where **`v ∈ [0, 20]` is non-negative throughout the offline data**, it fails catastrophically:

1. Reset: `u_buf` is all zeros. So `u_ini = [0, 0, …, 0]` (10 zeros for `T_ini=5, m_u=2`).
2. The QP must satisfy `Up · g = 0`. The columns of `Up` come from data where every `v` is non-negative.
3. To make a non-negative-valued matrix's columns sum to zero, `g` must contain **both positive and negative entries that cancel** — non-trivially.
4. With L1 sparsity, the cheapest `g` that does this picks a small set of mutually-canceling columns. Crucially, those columns have `Uf · g ≈ 0` as well — the **predicted future `u` is near zero too**.
5. Controller outputs `u_t ≈ 0`. The buffer's newest slot gets `u = 0` appended.
6. Next step: same buffer, same prediction, same `u = 0` output. **Locked**.

This wasn't theoretical — empirically the controller output `v ≈ 0` for entire 200-step episodes regardless of where the goal was.

## Considered

| Option | Status |
|---|---|
| Prime at zeros (original) | Rejected — causes the lock-in described above. |
| Prime at `action_bounds.low` | Rejected — for paper bounds with `v_min = 10`, this primes the buffer at always-moving, but for `v_min = 0` it's identical to zero-priming. |
| **Prime at midpoint of `action_bounds`** | Chosen. Always lies inside the action range; for non-negative-`v` data, the midpoint is `v > 0` so the QP's required combinations of columns are non-trivial in a healthy way. |
| Skip the first few steps and use random actions to warm up the buffer | Considered. More complex; doesn't generalize to all caller code. |

## Outcome

`scripts/run_deepc.py` computes:

```python
u_init_midpoint = 0.5 * (base.action_bounds[:, 0] + base.action_bounds[:, 1])
controller.reset(base.y, u_initial=u_init_midpoint)
```

For broad bounds (`v ∈ [0, 20]`, `w ∈ [-π/2, π/2]`) this gives `u_init = (10, 0)` — a "moving at moderate speed, not turning" prior. The QP's prediction is then non-degenerate from the first step.

## Generalization

The bug pattern is: **synthetic-past values that lie at the boundary of the data distribution lock the controller into producing more of that same boundary value**. For non-negative-only data, zero is at the boundary. For data centered around a midpoint, the midpoint is the natural "neutral" prior.

If the offline data has a different action distribution (e.g., centered around some other value), the right initial action might differ. A future refinement would be to estimate the mean action from the data and prime there.

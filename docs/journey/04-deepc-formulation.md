# 04. DeePC formulation — L1, L2, or hybrid regularization

## Decision

Match the [arXiv:2603.07395](https://arxiv.org/abs/2603.07395) paper exactly: **L1 on `g`**, **L2 on the explicit slack `σ_y`**.

## Context

There are three commonly used regularization schemes in the DeePC literature. They look similar mathematically but behave differently:

1. **Original DeePC** ([Coulson/Lygeros/Dörfler 2019](https://arxiv.org/abs/1811.05890)): L1 on both `g` and `σ_y`. Motivated by distributional robustness and sparsity in *which past observations were inconsistent*.
2. **SUMO Deep-LCC** (`CachedDeepLCCSolver`): L2 on both. Faster solver (pure QP vs SOCP). What we initially ported when we lifted the QP from the SUMO project.
3. **arXiv:2603.07395 Reg-DDPC**: L1 on `g`, L2 on `σ_y`. The hybrid. Sparse `g` (interpretable: "controller picks important trajectories") with a Tychonov-style soft constraint on past output.

We're following [arXiv:2603.07395](https://arxiv.org/abs/2603.07395), so the hybrid is the paper-faithful choice.

## Considered

The three options were laid out as a multiple-choice question, and the hybrid was picked. The key trade-offs:

| Aspect | Original DeePC (L1+L1) | Deep-LCC (L2+L2) | **Hybrid (L1+L2)** |
|---|---|---|---|
| Sparsity in `g` | Yes | No | Yes (the property we want) |
| Slack `σ_y` interpretation | "Which past was inconsistent" — sparse | Tychonov-regularized — smooth | Tychonov — smooth |
| Solver | SOCP | QP (fastest) | SOCP / mixed |
| Paper alignment | Original DeePC paper | SUMO project | arXiv:2603.07395 (ours) |

## Outcome

The QP:

$$
\min_{g,\ \sigma_y} \quad \lVert Y_f g - y_{\text{ref}} \rVert^2_{\bar Q} + \lVert U_f g \rVert^2_{\bar R} + \lambda_g \lVert g \rVert_1 + \lambda_y \lVert \sigma_y \rVert^2_2
$$

s.t. $U_p g = u_{\text{ini}}$, $Y_p g + \sigma_y = y_{\text{ini}}$, $u_{\min} \le U_f g \le u_{\max}$.

Defaults from the paper: $\lambda_g = 2$, $\lambda_y = 3 \cdot 10^6$.

## Implementation details

- The cost `‖x‖²_Q` is written as `cp.sum_squares(Q^{1/2} @ x)` (eigendecomposition with non-negative eigenvalue clamping) rather than `cp.quad_form(x, Q)`. This keeps the parametric problem **DPP-compliant** so CVXPY caches the compiled solver and per-step solves stay fast.
- $\sigma_y$ is an explicit decision variable (`cp.Variable(T_ini * p_y)`) rather than an implicit penalty term. Mathematically equivalent for L2 but lets us inspect the slack post-solve.
- A small constant `1.3e-3` is the `R` weight, matching the paper.

## Caveat

The L1 norm on `g` is significantly more aggressive per unit than L2 (the magnitude of $\lVert g \rVert_1$ for sparse `g` is much larger than $\lVert g \rVert_2^2$). The paper's `λ_g = 2` is tuned for `n_cols ≈ 1484` with `m_u = 2`, `p_y = 3`. Tests on tiny 1-D toy systems (`n_cols` in the hundreds, `m_u = 1`, `p_y = 1`) need to use lighter weights (e.g., `λ_g = 1.0`).

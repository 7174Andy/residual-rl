# 06. Why one library isn't enough — bilinear dynamics

## Decision

A **single DeePC library cannot navigate the unicycle to a goal**. The reason is structural, not a tuning problem. Switching to four libraries (next entry) is necessary.

## Context

DeePC's behavioral predictor is a **linear span** of past trajectories. For an LTI system the lemma is exact. For a system that's "linear enough" locally, it approximates well. The unicycle is neither — its dynamics are *bilinear*:

$$
x_{t+1} = x_t + \Delta t \cdot \cos(\delta_t) \cdot v_t
$$

The product $\cos(\delta_t) \cdot v_t$ cannot be represented as a linear function of $(v_t)$ across the full range of $\delta_t \in [-\pi, \pi]$. A single linear predictor trained on all-orientations data **averages out the heading dependence** — it predicts `cos(δ) ≈ 0` and `sin(δ) ≈ 0` on average over uniform δ.

## What this looks like in practice

Trained on a broad-PE library covering all orientations:

- The controller's predicted `Δx` for a given action depends almost entirely on where the data happened to land — not on the *current* heading.
- Asked "what happens if I drive `v = 10` right now?" the predictor returns something like "could be `+0.25` or `-0.25` or `0`, depending on which columns the QP selects."
- The QP's safest answer is "do nothing" (`v ≈ 0`). It cannot reliably claim that moving will reduce the position cost, because its predictor doesn't know which direction `v` will push the robot.

Empirically:

- Broad-`w` collection + single library: `v` collapses to ~0 across all episodes regardless of hyperparameters (`Q_heading`, `λ_g`, `λ_y`, `N`).
- Narrow-`w` collection + single library: `w` saturates at the bound; the robot moves in a tight circle.
- Adding bearing-aware `y_ref` + nonzero `Q[2, 2]`: `w` varies sensibly but `v` still collapses.

Across many hyperparameter sweeps, the failure mode is the same. It's not tunable.

## Considered

1. **More regularization tuning.** Tried `λ_g ∈ {0.1, 0.5, 2.0}`, `λ_y ∈ {1e3, 3e6}`, `N ∈ {6, 12, 24}`, `T_ini ∈ {5, 10}`. None helped.
2. **Narrower PE bounds.** Tried `w_abs_max = 0.2` so the data has fewer heading wraps. Predictor was less noisy but the robot's run-time turn rate was too low to navigate (180° turn would take 600+ steps; episode is 200).
3. **Different reference / cost.** Bearing-aware `y_ref` and non-zero `Q[2,2]` fixed the `w`-saturation but not the `v`-collapse.
4. **A different controller family.** Pure pursuit, NMPC with the known analytical model — would work but bypass the DeePC story entirely.
5. **Library switching** — what the paper does (chosen, see next entry).

## Outcome

Library switching solves the problem structurally. The paper's quote — *"the data-driven constraint can be viewed as a local linear approximation, [so] our strategy is to switch the data library based on the robot's current state"* — is the explicit acknowledgement that one library isn't enough. The paper, and now this repo, addresses the bilinear nonlinearity by **piecewise linearization** over four heading quadrants.

This entry exists because the failure isn't obvious from the formulation. Spending several hours sweeping hyperparameters and reading run-output gradients eventually points to "the predictor is too uncertain to commit to motion" — but the root cause is the dynamics, not the QP. Recognizing that earlier saves time on a future similar pivot.

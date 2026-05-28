# 02. Env design — observation, action, reward

## Decision

- **State** (world frame): $(x, y, \delta)$.
- **Action**: $(v, w)$ in physical units.
- **Observation** (body frame): $(\text{distance}, \sin(\text{bearing}_{\text{rel}}), \cos(\text{bearing}_{\text{rel}}), v_{\text{prev}}, w_{\text{prev}})$.
- **Reward**: $-(y - y_{\text{ref}})^\top Q (y - y_{\text{ref}}) - u^\top R u + b \cdot [\text{reached}]$.

State and obs are different. The body-frame obs is for the RL agent; classical controllers read state directly.

## Context

Goal-reaching with an RL agent + a classical (DeePC) controller has two competing needs:

- **RL needs sample-efficient features.** Symmetries (translation, rotation) should be baked into the obs so the policy doesn't have to re-learn them at every workspace location and every orientation.
- **Classical predictive controllers need world-frame state.** They run their own optimization on absolute coordinates; body-frame obs is unnatural for them.

## Considered

### Observation

1. **Absolute coords** `(x, y, δ, g_x, g_y)`. Easy to interpret; RL has to learn translation-invariance from scratch. The heading wraps at $\pm\pi$ — discontinuity hurts NN gradients.
2. **`sin`/`cos` heading** `(x, y, sin δ, cos δ, g_x, g_y)`. Fixes the wrap but still no translation invariance.
3. **Body-frame relative** (chosen) `(distance, sin(bearing_rel), cos(bearing_rel), v_prev, w_prev)`. Translation- and rotation-invariant; continuous in heading via the sin/cos.

The chosen form is what most goal-conditioned navigation papers use.

### Action

1. **Native units, configurable bounds** (chosen). Default `v ∈ [0, 20]`, `w ∈ [-π/2, π/2]`. RL wraps with `gymnasium.wrappers.RescaleAction` (provided in `rl/wrappers.py`) to get a symmetric `[-1, 1]²` space.
2. **Normalized in the env itself** `(action ∈ [-1, 1]²)`, rescaled internally. Cleaner for RL but inconvenient for classical controllers; the rescale wrapper is a one-liner so the choice was easy.

### Reward

1. **LQR-style quadratic** $-(y - y_{\text{ref}})^\top Q (y - y_{\text{ref}}) - u^\top R u$ (chosen). Matches the paper exactly; comparable across controllers.
2. **Dense L2 distance** $-\lVert p - g \rVert$. Simpler but doesn't match the paper's `Q`/`R` framing; less comparable to DeePC's cost.
3. **Sparse +1 on reach**. Hardest for RL; not chosen.

Plus a **reach bonus** of `+100` on the step the goal is hit — makes the LQR-style return cleanly compare "reached" vs "didn't" episodes.

## Outcome

- World-frame state on `env.unwrapped.state`; body-frame obs returned by `step()`/`reset()`.
- Stored state has heading wrapped to $[-\pi, \pi]$ at every step. Caller-supplied initial headings (via `options={"state": ...}`) are wrapped on entry too — keeps the first step continuous with the rest.
- Wall clipping on `(x, y)` after the dynamics integration prevents the robot from escaping the workspace.
- `obs_space.low` for the action-component slots (`v_prev`, `w_prev`) is widened to `min(0, a_min)` so the post-`reset()` zero-init lies inside the obs space even when `action_bounds` excludes zero (e.g., paper bounds with `v_min = 10`).

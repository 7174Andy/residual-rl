# 01. Task framing — trajectory tracking vs goal-reaching

## Decision

The env is a **point-to-point goal-reaching task**, not the paper's trajectory tracking. Random start, random goal, terminate on `‖p − g‖ < ε`, truncate at `max_steps`.

## Context

Appendix D of [arXiv:2603.07395](https://arxiv.org/abs/2603.07395) defines a kinematic unicycle robot tracking a **heart-shaped reference curve** for two cycles. That's the task the paper benchmarks DDPC on.

For this repo I wanted the env to serve a different purpose: be a **benchmark for goal-conditioned controllers** (DeePC + RL baselines). Trajectory tracking and goal-reaching are different problem classes:

- **Tracking**: reference is a time-indexed curve. Robot starts on the reference. Each step's target is `r_t` for the current `t`. Controller's job is "stay close to the moving reference."
- **Goal-reaching**: target is a fixed point. Robot starts arbitrarily. Each step's target is the *same* goal. Controller's job is "plan a path to the goal."

Goal-reaching is more general — most RL navigation literature uses it. It also stresses controllers differently (planning matters; staying near a reference is less relevant).

## Considered

1. **Keep tracking as the task.** Paper-faithful. But narrows the scope to a single reference curve and doesn't naturally generalize to RL setups.
2. **Goal-reaching only** (chosen). General, reusable, supports RL out of the box.
3. **Both, behind a flag.** Doubles the surface area for not much benefit; tracking can always be re-added later if needed.

## Outcome

- Kept the paper's underlying *dynamics* exactly (`Δt = 0.025`, unicycle integration, paper's `Q`/`R` matrix shape).
- Reframed the task: `reset()` samples random `(state, goal)`; `terminated` fires on reach.
- The DeePC controller still works — the [Coulson/Lygeros/Dörfler DeePC paper](https://arxiv.org/abs/1811.05890) handles non-zero reference tracking explicitly, so goal-reaching is a valid use case.

The deviation is documented in `CLAUDE.md` and on the [environment overview](../environment/index.md). Reproducing the paper's exact heart-curve tracking would require a separate env class and is not on the current roadmap.

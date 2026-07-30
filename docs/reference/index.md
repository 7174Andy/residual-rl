# Reference

Pure reference for the env, controllers, and CLI scripts. No "why" — that lives in the [Journey](../journey/index.md). Each reference page links back to the relevant journey entry when there's design rationale worth knowing.

<div class="grid cards" markdown>

-   :material-robot:{ .lg .middle } &nbsp; __Environment__

    ---

    `TwoWheelGoal-v0` — dynamics, state/action/observation spaces, reward, DeePC interface, and API.

    [:octicons-arrow-right-24: Environment](../environment/index.md)

-   :material-cog-outline:{ .lg .middle } &nbsp; __Controllers__

    ---

    `DeePC` (with orientation-keyed library switching), plus the offline data collection and Hankel construction.

    [:octicons-arrow-right-24: Controllers](../controllers/index.md)

-   :material-console:{ .lg .middle } &nbsp; __CLI scripts__

    ---

    All 14 scripts under `scripts/` — DeePC data collection/run, clone training/eval, TD3/SAC residual training/eval, and the plotting/video tools. Flags, defaults, expected output.

    [:octicons-arrow-right-24: CLI](cli.md)

</div>

## What's where

### Environment

- [Overview](../environment/index.md) — Gym registration, configurable parameters.
- [Dynamics & spaces](../environment/dynamics.md) — equations of motion, state, action, observation.
- [Reward & episode](../environment/reward.md) — stage cost, reach bonus, termination rules.
- [DeePC interface](../environment/deepc-interface.md) — `y`, `y_ref`, `Q`, `R`.
- [API](../environment/api.md) — auto-generated from docstrings.

### Controllers

- [Overview](../controllers/index.md) — package layout, controller protocol.
- [DeePC](../controllers/deepc.md) — QP formulation, hyperparameters, caching.
- [Library switching](../controllers/library-switching.md) — orientation-keyed library selection built into DeePC.
- [Data collection](../controllers/data-collection.md) — offline PE trajectories, Hankel matrices.
- [API](../controllers/api.md) — auto-generated from docstrings.

### CLI

- [CLI scripts](cli.md) — full flag reference for all 14 runnable scripts.

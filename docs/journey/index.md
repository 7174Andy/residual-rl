# Journey

This is the **decision log** for the project. Each entry captures a real fork in the road, what was decided, what was tried, and what was ruled out. The point isn't to document the code; the code already speaks for itself. The point is to capture the _why_, including the dead ends.

Entries are roughly chronological, but each one stands alone — you can read them in any order.

| #                                | Topic                                                             |
| -------------------------------- | ----------------------------------------------------------------- |
| [01](01-task-framing.md)         | Task framing — trajectory tracking vs goal-reaching               |
| [02](02-env-design.md)           | Env design — observation, action, reward                          |
| [04](04-deepc-formulation.md)    | DeePC formulation — L1, L2, or hybrid regularization              |
| [05](05-library-switching.md)    | Library switching — bilinear dynamics force 4 local libraries      |
| [06](06-stop-at-goal.md)         | Stopping at the goal — overshoot (fixed) → over-braking (RL next) |
| [07](07-imitation-learning.md)   | Imitation learning — cloning DeePC into a fast neural policy      |
| [08](08-residual-rl.md)          | Residual RL — TD3 correction lifts reach rate 38.5%→94.9%         |
| [09](09-vanilla-rl.md)           | Vanilla RL — the control arm; what the MPC prior actually buys    |
| [10](10-sample-efficiency.md)    | Sample efficiency on the deployed policy — prior × optimizer interact |
| [11](11-panda-anchors.md)        | Panda anchors — the controller works, the libraries don't reach   |
| [12](12-select-dpc.md)           | Select-DPC — and a 2-DoF arm to test it on                        |

## Conventions

Each entry uses a small four-section template:

- **Decision** — the one-line conclusion.
- **Context** — what triggered the decision.
- **Considered** — alternatives that were on the table.
- **Outcome** — what changed in the code, and any caveats.

# Journey

This is the **decision log** for the project. Each entry captures a real fork in the road, what was decided, what was tried, and what was ruled out. The point isn't to document the code; the code already speaks for itself. The point is to capture the *why*, including the dead ends.

Entries are roughly chronological, but each one stands alone — you can read them in any order.

| # | Topic |
|---|---|
| [01](01-task-framing.md) | Task framing — trajectory tracking vs goal-reaching |
| [02](02-env-design.md) | Env design — observation, action, reward |
| [03](03-action-bounds.md) | Action bounds — paper-faithful vs broad |
| [04](04-deepc-formulation.md) | DeePC formulation — L1, L2, or hybrid regularization |
| [05](05-cold-start.md) | Cold-start bug — when the synthetic past locks the controller |
| [06](06-single-library-fails.md) | Why one library isn't enough — bilinear dynamics |
| [07](07-library-switching.md) | Library switching — local linearization via 4 quadrants |
| [08](08-bearing-reference.md) | Bearing-aware reference + nonzero `Q[2,2]` |

## Conventions

Each entry uses a small four-section template:

- **Decision** — the one-line conclusion.
- **Context** — what triggered the decision.
- **Considered** — alternatives that were on the table.
- **Outcome** — what changed in the code, and any caveats.

# Single parametric DeePC with swappable Hankel libraries

**Date:** 2026-05-28
**Status:** Approved (design); pending implementation

## Motivation

The current DeePC implementation splits two responsibilities across two classes:

- `DeePC` — holds one library's Hankel matrices `(Up, Uf, Yp, Yf)` baked into a
  CVXPY problem as **constants**, and solves a cached parametric QP each step.
- `LibrarySwitchingDeePC` — wraps a list of `DeePC` objects (one per
  orientation-keyed library), routes each `act()` to the controller whose
  anchor heading is closest, and shares a single past-`(u, y)` buffer across them.

The reference paper (arXiv:2603.07395, Appendix D) describes a **single**
controller that holds all orientation-keyed libraries and, each step, selects
the appropriate library and feeds its Hankels into one predictive QP. CVXPY's
DPP rules permit the Hankel matrices to be `cp.Parameter`s (verified:
`sum_squares(F @ g)` with `F` a parameter is DPP-compliant), so the two classes
can collapse into one parametric controller that swaps Hankel parameter values
per step instead of swapping whole compiled problems.

This is a behavior-preserving refactor: the QP math is unchanged, and outputs
must match the existing classes (single-library equals old `DeePC`,
multi-library equals old `LibrarySwitchingDeePC`).

## Goals

- One controller class implementing the paper's swappable-library strategy.
- Identical QP formulation and numerics to today.
- Single-library use supported as the degenerate `N_libraries == 1` case.
- Existing CLI (`scripts/run_deepc.py`, including `--single_library`) preserved.

## Non-goals

- No change to `hankel.build_hankel` (still the producer of Hankel tuples).
- No change to the QP cost/constraint structure or default hyperparameters.
- No Koopman-lifted single-library variant (out of scope).

## Architecture

A single class, `DeePC` (reusing the name; it is now *the* controller). It
holds all orientation-keyed libraries, builds **one** cached CVXPY QP whose
`Up`, `Uf`, `Yp`, `Yf` are `cp.Parameter` matrices, and on each `act()`:

1. Selects the library by heading (wrap-aware closest-anchor logic, carried
   over unchanged from `LibrarySwitchingDeePC._select_index`).
2. Clears the warm-start (`g.value = None`) **if the selected library changed**
   since the previous step — a warm-start `g` indexes the previous library's
   columns and is meaningless under a different library. Same-library
   consecutive steps keep the warm-start.
3. Writes the selected library's Hankels into the four matrix parameters, plus
   `u_ini`, `y_ini`, `y_ref` into their vector parameters.
4. Solves, extracts the first control move `u_future[:m_u]`, clips to
   `u_bounds`, slides the single shared past-`(u, y)` buffer.

`LibrarySwitchingDeePC` is deleted. Single-library use falls out as
`N_libraries == 1` (one anchor; selection trivially returns index 0), preserving
the old plain-`DeePC` path through the same API.

### Constructor

```python
DeePC(
    libraries: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        # one (Up, Uf, Yp, Yf) tuple per library, e.g. from hankel.build_hankel
    anchor_headings,            # parallel array, len == len(libraries)
    Q: np.ndarray,
    R: np.ndarray,
    T_ini: int = 5,
    N: int = 12,
    lambda_g: float = 2.0,
    lambda_y: float = 3e6,
    u_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    heading_index: int = 2,
    solver: str | None = None,
)
```

Validation:

- At least one library.
- All libraries share `n_cols` (required — they feed one shared parameter set).
- Each library's `(Up, Uf, Yp, Yf)` shapes match `T_ini`, `N`, and the derived
  `m_u`, `p_y` (same checks the old `DeePC.__init__` performed, applied per
  library).
- `Q` is `(p_y, p_y)`, `R` is `(m_u, m_u)`.
- `len(anchor_headings) == len(libraries)`; `0 <= heading_index < p_y`.
- For `N_libraries == 1`, `anchor_headings` may be a single-element array (and
  selection always returns 0 regardless of its value).

### QP (unchanged math)

```
min       ‖Yf·g − y_ref‖²_Q̄  +  ‖Uf·g‖²_R̄  +  λ_g·‖g‖₁  +  λ_y·‖σ_y‖²₂
g, σ_y
s.t.      Up·g       = u_ini          (hard past-input constraint)
          Yp·g + σ_y = y_ini          (soft past-output constraint, slack σ_y)
          u_min ≤ Uf·g ≤ u_max        (optional control bounds)
```

with `Q̄ = I_N ⊗ Q`, `R̄ = I_N ⊗ R`, costs written as `‖Q^{1/2}x‖²` /
`‖R^{1/2}x‖²` for DPP compliance. The only change from today is that
`Up/Uf/Yp/Yf` are `cp.Parameter`s set per step rather than constants baked in
at construction.

### API

- `reset(y_initial, u_initial=None)` — prime the single shared past buffer with
  `T_ini` copies of `(u_initial, y_initial)` (zeros for `u_initial` by default).
  Same semantics as today.
- `act(y_current, y_ref) -> u_t` — select library, set parameters, solve, slide
  buffer, return `u_t` of shape `(m_u,)`. Accepts `y_ref` of shape `(p_y,)`
  (broadcast across the horizon) or `(N, p_y)` (per-step).
- `last_library_idx: int` — diagnostic, index used on the most recent `act()`
  (−1 before the first call).

## Data flow

```
build_hankel(u_i, y_i) per library  ──►  list[(Up,Uf,Yp,Yf)]
                                            │
                                            ▼
                              DeePC(libraries, anchors, Q, R, ...)
                                            │  one cached QP, Hankels as Parameters
            reset(y0, u0) ──► shared (u,y) buffer primed
                                            │
   each step:  act(y_current, y_ref)
                 ├─ select idx by heading
                 ├─ clear warm-start if idx changed
                 ├─ set Hankel params[idx] + u_ini/y_ini/y_ref
                 ├─ solve QP
                 └─ slide shared buffer ──► u_t
```

## Error handling

- Constructor raises `ValueError` on any shape/length/range mismatch (as above).
- `act()` raises `RuntimeError` if called before `reset()`, or if the QP status
  is not `optimal`/`optimal_inaccurate` (unchanged from today).
- Returned `u_t` is clipped to `u_bounds` to absorb sub-tolerance solver
  violations (unchanged).

## Testing

Rewrite `tests/test_deepc.py`:

- **Construction/validation:** rejects empty library list, mismatched `n_cols`
  across libraries, wrong Hankel shapes, wrong `Q`/`R` shapes, bad
  `anchor_headings` length, out-of-range `heading_index`.
- **Single-library equivalence:** `N_libraries == 1` produces the same `u_t`
  sequence as the *old* `DeePC` on a fixed library and input stream (guard
  against regression; can capture old outputs as fixtures or assert against a
  re-derived expectation).
- **Multi-library equivalence:** routing + outputs match the *old*
  `LibrarySwitchingDeePC` (same anchors, same heading stream → same
  `last_library_idx` sequence and same `u_t`).
- **Warm-start reset on switch:** when the selected library changes between
  steps, `g.value` is cleared before the solve; when it stays the same, the
  warm-start is retained. (Assert on observable behavior, e.g. solve still
  yields the same optimum, plus an internal check that switching does not carry
  a stale `g`.)
- **Buffer sliding / reset semantics, QP feasibility, bounds clipping** — carry
  over existing coverage.

## Callers and docs to update

- `scripts/run_deepc.py`: build a `list[(Up,Uf,Yp,Yf)]`, pass once to `DeePC`
  with the anchor array; `--single_library i` → one-element library list with
  that library's anchor. Remove the `LibrarySwitchingDeePC` import and the
  two-branch construction.
- `two_wheel_robot/controllers/__init__.py`: update any exports.
- Docs describing the old two-class split:
  - `docs/controllers/deepc.md`
  - `docs/controllers/library-switching.md`
  - `docs/journey/07-library-switching.md` (journey doc — historical; update
    only if it documents current API rather than the development story)
  Reconcile these with the single-class design during implementation.

## Risks / open considerations

- **Performance:** parameterized Hankels prevent some constant-folding the
  constant-baked problem allowed. Expected per-step solve time is comparable
  (DPP caching still applies), but worth a sanity check during implementation;
  not a correctness concern.
- **Warm-start semantics:** clearing `g` on switch is the conservative choice.
  If profiling later shows switches are rare and warm-starting across them is
  harmless, this can be revisited — but correctness-first for now.

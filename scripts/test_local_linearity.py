"""Is the IO map `y_f = f(y_p, u_p, u_f)` locally differentiable, and at what radius?

Select-DPC's correctness rests on Appendix B of arXiv:2503.18845, which states the
equivalence outright: Select-DPC *is* SQP-MPC (its Algorithm 4) with the analytical
Jacobians

    F_p := d f / d[u_p; y_p]        F_f := d f / d u_f

of the IO map `y_f = f(u_p, u_f, y_p)` (its eq. 12) replaced by a data-driven
estimate built from the selected columns, and the appendix's closing claim is that
this estimate "approaches the analytical Jacobians assuming sufficient amounts of
data in the neighborhood around the linearization point (i.e. card(D) tending to
infinity)".

That sentence carries three separable preconditions:

1. `f` EXISTS -- `(u_p, y_p)` must pin the state, or `f` is one-to-many and there is
   no Jacobian to estimate. Settled: tip-only `y` leaves a 4-D blind subspace
   (`docs/journey/11-panda-anchors.md`), and `y = [q; p_ee]` fixes it (cond ~98).
2. `f` is DIFFERENTIABLE at the operating point, over a neighborhood wide enough to
   contain the data you have. UNTESTED -- this script.
3. There is DATA in that neighborhood. Settled, negatively, for the Panda: nearest
   collected sample 1.48 rad against a ~0.5 rad validity radius
   (`docs/journey/12-select-dpc.md`).

So this measures precondition 2, with 1 and 3 already known. It answers two
questions, and the second is the one that decides anything:

  (a) Does `f` behave like a differentiable map in the limit? Shrink the
      neighborhood radius `r` and check that the local-linear prediction error
      `E(r)` falls and the fitted Jacobian `J_r` stops moving.
  (b) Is it still differentiable AT THE RADIUS THE DATA FORCES YOU TO USE? The
      Panda's selected columns sit a median 2.01 rad away, so the eps ladder runs
      well past the textbook range into the regime the controller actually
      operates in. A map can be perfectly differentiable and still have a useless
      linearization 2 rad out.

Admissibility -- why the perturbation is applied to the state and the inputs
---------------------------------------------------------------------------
`z = [y_p; u_p; u_f]` cannot be perturbed componentwise: `y_p` is not a free
variable, it is what the plant DID given the state and `u_p`. Perturbing it
independently asks `f` about trajectories the system cannot produce, and `f` is
simply not defined there.

So the perturbation is applied to the free quantities -- the initial state
`(q, qdot)` and the input sequence `(u_p, u_f)` -- the rollout is run, and the
REALIZED `dz` is read off and regressed against. Every sample is an admissible
trajectory by construction. The consequence is that `dz` spans only a
`dim(x) + L*m`-dimensional subspace of the `T_ini*p + L*m`-dimensional `z` space
(133 of 169 for the Panda), so `J` is identifiable only on that subspace --
which is fine, and not a limitation of the test: the bank columns Select-DPC
regresses over are admissible trajectories too, so it never sees the rest either.
`lstsq` returns the minimum-norm solution and validation is on held-out samples
from the same distribution.

Reading the output
------------------
`E_self` is the held-out relative error of `J_r` at its own radius; `E_self = 1.0`
is the "predict no response" baseline, so anything at or above 1.0 means the
linearization is worse than useless. `E_cross` applies the Jacobian fitted at the
NEXT LARGER radius to this radius's held-out data -- that is the `J_eps` vs
`J_eps/2` comparison, done through prediction rather than by differencing two
minimum-norm matrices whose null spaces are not comparable. `E_cross -> E_self`
means the Jacobian has stopped moving; `E_cross >> E_self` means it is still
changing between the two radii, i.e. second-order terms dominate.

What it measured
----------------
Precondition 2 HOLDS for the Panda's smooth dynamics and FAILS for its actuators.

With the torque ceiling lifted (`--no-force-limit`), the arm is textbook: `E_self`
falls in exact proportion to `r` -- 0.138, 0.045, 0.015, 0.0054, 0.0015, 0.0005,
0.0002 down the ladder, an `O(r^2)` remainder -- and `E_cross` collapses onto
`E_self`, i.e. `J_r` converges. Reacher does the same unmodified (0.282 -> 0.0006).

As actually collected, the Panda instead bottoms out at `E_self ~ 0.24` around
`r_z = 0.47` and then gets WORSE as the radius shrinks, reaching 0.83. A
differentiable map cannot do that. The cause is the PD servo saturating: applied
torque is `clip(kp*(q_des - q) - kd*qdot, +-forcerange)` with `kp = 2000-4500`
against `forcerange = 12-87 N.m`, so the actuator is linear only for
`|q_des - q| < 0.006 rad` (wrist) to `0.025` (shoulder). `qdes.OU_SIGMA = 0.25`
is **42x** that, and 29.4% of (step, joint) pairs sit hard against the limit.
Between switching surfaces `d y_f / d u_f` is ~0; on them it is undefined. The
kinks are dense at every scale, which is why shrinking `r` does not help.

Exciting more smoothly does not escape it: `--theta 0.999` cuts saturation to 6.6%
at comparable travel and `E_self` still turns around (0.18 -> 0.51). Exciting more
weakly does, but only by not moving -- the saturation-free setting travels 4.7 mm
per window against 248 mm at the collection default.

Usage:
    python scripts/test_local_linearity.py --system panda
    python scripts/test_local_linearity.py --system panda --no-force-limit
    python scripts/test_local_linearity.py --system reacher     # working control
"""
from __future__ import annotations

import argparse

import numpy as np

# eps is a radius in SCALED source units (see `Sys.src_scale`); the ladder runs
# from the textbook limit up to the radius the Panda's selected data actually
# sits at, because only the top of it speaks to whether Select-DPC can work here.
DEFAULT_EPS = (1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0)


class Sys:
    """Plant adapter: nominal trajectory, rollout, and the scales that make the
    regression unit-free. `rollout` must return `y` of shape `(L, p)` with `y_t`
    observed BEFORE `u_t` is applied -- the alignment every collection here uses.
    """

    def __init__(self, model, data, q0, qd0, u_nom, src_scale, step, read_y, set_state,
                 y_scale, u_scale):
        self.model, self.data = model, data
        self.q0, self.qd0, self.u_nom = q0, qd0, u_nom
        self.src_scale = src_scale          # (nq, nv, m) characteristic magnitudes
        self._step, self._read_y, self._set = step, read_y, set_state
        # Per-channel scales measured from a long COLLECTION rollout at the anchor,
        # i.e. the natural spread of each channel in the data a bank is built from.
        # They must not come from the perturbation ensembles: taking them from the
        # widest one lets a channel that SATURATES there (a joint pinned against
        # the safe box) get a near-zero std, and dividing by it amplifies that
        # channel's noise at every other radius -- which is exactly what made an
        # earlier version of this script report a spurious floor below eps ~ 0.1.
        # Measured from collection, they depend on nothing the ladder varies.
        self.y_scale, self.u_scale = y_scale, u_scale
        self.nq, self.nv, self.m = len(q0), len(qd0), u_nom.shape[1]
        self.L = u_nom.shape[0]
        self.n_clipped = 0

    def zy_scales(self, T_ini: int, N: int):
        """Per-component scales for `z = [y_p; u_p; u_f]` and `y_f`, unit-free."""
        return (np.concatenate([np.tile(self.y_scale, T_ini), np.tile(self.u_scale, self.L)]),
                np.tile(self.y_scale, N))

    @property
    def src_dim(self) -> int:
        return self.nq + self.nv + self.L * self.m

    def scale_vec(self) -> np.ndarray:
        sq, sv, su = self.src_scale
        return np.concatenate([np.full(self.nq, sq), np.full(self.nv, sv),
                               np.full(self.L * self.m, su)])

    def rollout(self, dsrc: np.ndarray):
        """One admissible trajectory from a perturbed state + input sequence.

        Returns `(y, u_applied)`. Returning the APPLIED input is not a detail: the
        actuator clips `ctrl` to the safe box, so the requested `u_nom + du` is not
        always what the plant received, and regressing against the request would
        put a value in `z` that never reached the system -- the same "recorded u
        differs from applied ctrl" trap `panda/qdes.py` was written to avoid.
        """
        dq = dsrc[:self.nq]
        dqd = dsrc[self.nq:self.nq + self.nv]
        du = dsrc[self.nq + self.nv:].reshape(self.L, self.m)
        self._set(self.q0 + dq, self.qd0 + dqd)
        y = np.empty((self.L, self._read_y().shape[0]))
        u = np.empty((self.L, self.m))
        for t in range(self.L):
            y[t] = self._read_y()
            req = self.u_nom[t] + du[t]
            u[t] = self._step(req)
            if not np.allclose(u[t], req, atol=1e-12):
                self.n_clipped += 1
        return y, u


def panda_sys(seed: int, L: int, sigma: float | None = None,
               no_force_limit: bool = False, theta: float | None = None,
               servo_scale: float = 1.0) -> Sys:
    import mujoco
    from panda.model import frame_skip, load_model, safe_box, sample_config, tip_id
    from panda.qdes import OU_SIGMA, OU_THETA, collect_anchor, outputs, step_qdes

    model, data = load_model(servo_scale=servo_scale)
    sig = OU_SIGMA if sigma is None else float(sigma)
    th = OU_THETA if theta is None else float(theta)
    lo, hi = safe_box(model)
    fs, tip = frame_skip(model), tip_id(model)
    rng = np.random.default_rng(seed)

    # The PD servo saturates: applied torque is clip(kp*(q_des - q) - kd*qdot,
    # +-forcerange), and kp is 2000-4500 against a forcerange of 12-87 N.m, so the
    # command stays inside the actuator's linear range only for
    # |q_des - q| < forcerange/kp = 0.006 rad (wrist) to 0.025 rad (shoulder).
    # Above that the plant is bang-bang and `d y_f / d u_f` is ~0 between switching
    # surfaces and undefined on them. `--sigma` exists to cross that boundary.
    if no_force_limit:
        # Control: remove the torque ceiling and nothing else. If the error curve
        # becomes monotone in r only here, the saturation IS the non-smoothness
        # rather than something else that happens to correlate with it.
        model.actuator_forcelimited[:] = 0
        model.actuator_forcerange[:] = [-1e9, 1e9]
    lin = float(np.min(model.actuator_forcerange[:, 1] / model.actuator_gainprm[:, 0]))
    print(f"actuator linear range |q_des - q| < {lin:.4f} rad; excitation sigma "
          f"= {sig:.4f} rad ({sig / lin:.0f}x)")

    # An anchor with room to move. `ctrl` is clipped to the safe box, so an anchor
    # sitting near a box face makes the excitation one-sided and puts a genuine
    # kink in `f` right at the linearization point -- which would be measured as
    # non-differentiability of the ARM when it is really an artifact of where the
    # anchor landed. Insist on 3 sigma of clearance so the clip is not what this
    # script ends up measuring; `clip_frac` in the output verifies it worked.
    margin = 3.0 * max(sig, 0.05)
    for _ in range(500):
        anchor, _ = sample_config(model, data, rng, lo, hi, tip)
        if np.all(anchor - lo > margin) and np.all(hi - anchor > margin):
            break
    else:
        raise RuntimeError(f"no collision-free anchor with {margin:.2f} rad of box "
                           f"clearance in 500 draws")

    # Nominal input: the same OU excitation `panda/qdes.collect_anchor` uses, so
    # the linearization point sits in the regime the bank was collected in.
    e, u_nom = np.zeros(model.nq), np.empty((L, model.nq))
    for t in range(L):
        e = th * e + sig * np.sqrt(1 - th**2) * rng.standard_normal(model.nq)
        u_nom[t] = np.clip(anchor + e, lo, hi)

    def set_state(q, qd):
        data.qpos[:] = np.clip(q, lo, hi)
        data.qvel[:] = qd
        data.ctrl[:] = data.qpos
        mujoco.mj_forward(model, data)

    ref = collect_anchor(model, data, anchor, 400, np.random.default_rng(seed + 7),
                         theta=th, sigma=sig)
    return Sys(
        model, data, anchor, np.zeros(model.nv), u_nom,
        src_scale=(1.0, 1.0, 1.0),          # q, qdot, q_des -- all rad or rad/s
        step=lambda u: step_qdes(model, data, u, lo, hi, fs),
        read_y=lambda: np.concatenate([data.qpos.copy(), data.site_xpos[tip].copy()]),
        set_state=set_state,
        y_scale=outputs(ref["q"], ref["tip"]).std(0),
        u_scale=ref["u"].std(0),
    )


def reacher_sys(seed: int, L: int, sigma: float | None = None) -> Sys:
    import mujoco
    from reacher.deepc_setup import (K_DAMP, K_RET, OU_SIGMA, OU_THETA,
                                     collect_anchor, outputs)
    from reacher.model import (NQ_ARM, fingertip, frame_skip, load_model,
                               safe_box, sample_config, step_torque, wrap)

    model, data = load_model()
    fs = frame_skip(model)
    sig = OU_SIGMA if sigma is None else float(sigma)
    th = OU_THETA
    rng = np.random.default_rng(seed)
    anchor, _ = sample_config(model, data, rng)

    # Nominal torque: OU + the weak restoring PD `collect_anchor` uses. Without
    # the restoring term the arm random-walks off the anchor and the "local"
    # neighborhood is not local.
    e, u_nom = np.zeros(NQ_ARM), np.empty((L, NQ_ARM))
    for t in range(L):
        e = OU_THETA * e + sig * np.sqrt(1 - OU_THETA**2) * rng.standard_normal(NQ_ARM)
        q = np.asarray(data.qpos[:NQ_ARM])
        err = np.array([wrap(anchor[0] - q[0]), anchor[1] - q[1]])
        u_nom[t] = step_torque(model, data, e + K_RET * err - K_DAMP * data.qvel[:NQ_ARM], fs)

    def set_state(q, qd):
        data.qpos[:NQ_ARM] = q
        data.qvel[:NQ_ARM] = qd
        mujoco.mj_forward(model, data)

    ref = collect_anchor(model, data, anchor, 400, np.random.default_rng(seed + 7),
                         theta=th, sigma=sig)
    return Sys(
        model, data, anchor, np.zeros(NQ_ARM), u_nom,
        # u is a TORQUE here, not an angle -- scaling it like radians would make
        # the input block dominate the source ball. `OU_SIGMA` is its own unit.
        src_scale=(1.0, 1.0, sig),
        step=lambda u: step_torque(model, data, u, fs),
        read_y=lambda: np.concatenate([np.asarray(data.qpos[:NQ_ARM]).copy(),
                                       fingertip(data)]),
        set_state=set_state,
        y_scale=outputs(ref["q"], ref["tip"]).std(0),
        u_scale=ref["u"].std(0),
    )


def ensemble(sys: Sys, eps: float, n: int, rng: np.random.Generator, T_ini: int):
    """`n` admissible perturbed trajectories at source radius `eps`.

    Returns `(dz, dy, dq)` -- realized deviations in `z = [y_p; u_p; u_f]` and
    `y_f`, both RAW (scaling happens once, globally), plus the config-space
    excursion in rad so the radius can be compared against the 0.5 rad validity
    radius and the 1.48/2.01 rad selection distances.
    """
    scale = sys.scale_vec()
    y0, u0 = sys.rollout(np.zeros(sys.src_dim))
    z0, yf0 = pack(y0, u0, T_ini)
    dz, dy, dq = [], [], []
    for _ in range(n):
        v = rng.standard_normal(sys.src_dim)
        v /= np.linalg.norm(v)                    # eps IS the radius, not eps*sqrt(dim)
        y, u = sys.rollout(eps * scale * v)
        z, yf = pack(y, u, T_ini)
        dz.append(z - z0)
        dy.append(yf - yf0)
        dq.append(np.linalg.norm(y[:, :sys.nq] - y0[:, :sys.nq], axis=1).max())
    return np.array(dz), np.array(dy), np.array(dq)


def pack(y: np.ndarray, u: np.ndarray, T_ini: int):
    """`(z, y_f)` with `z = [y_p; u_p; u_f]` flattened, matching eq. (12)."""
    return (np.concatenate([y[:T_ini].ravel(), u.ravel()]), y[T_ini:].ravel())


def rel_err(dy: np.ndarray, dz: np.ndarray, J: np.ndarray) -> float:
    """Relative Frobenius error. 1.0 is the `dy ~ 0` baseline: at or above it the
    linearization carries no information about the response."""
    return float(np.linalg.norm(dy - dz @ J, "fro") / np.linalg.norm(dy, "fro"))


def fit_jacobian(dz: np.ndarray, dy: np.ndarray, rank: int):
    """Least squares truncated at a KNOWN rank, rather than `lstsq`'s default.

    `dz` is rank-deficient by construction -- it lives in the
    `dim(x) + L*m`-dimensional image of the perturbation, inside a wider `z`
    space -- so the trailing singular values are numerical noise. `lstsq(rcond=None)`
    cuts at machine precision, which keeps them, and the resulting `J` has huge
    components on directions the training set barely explored: it fits the fit set
    and explodes on held-out data. Truncating at the known source dimension is the
    fix. Returns `(J, k, tail)` where `tail` is the fraction of squared singular
    value mass discarded -- if that is not tiny, the truncation is throwing away
    signal and the rank argument is wrong.
    """
    U, s, Vt = np.linalg.svd(dz, full_matrices=False)
    k = int(min(rank, np.sum(s > s[0] * 1e-10)))
    tail = float((s[k:] ** 2).sum() / (s**2).sum()) if k < len(s) else 0.0
    return Vt[:k].T @ ((U[:, :k].T @ dy) / s[:k, None]), k, tail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=("panda", "reacher"), default="panda")
    ap.add_argument("--T-ini", type=int, default=5)
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--n-fit", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--eps", type=float, nargs="+", default=list(DEFAULT_EPS))
    ap.add_argument("--sigma", type=float, default=None,
                    help="excitation amplitude; default is the system's collection value")
    ap.add_argument("--servo-scale", type=float, default=1.0,
                    help="panda only: multiply PD servo kp and kd (0.02 = /50)")
    ap.add_argument("--theta", type=float, default=None,
                    help="panda only: OU correlation; raise it to excite more smoothly")
    ap.add_argument("--no-force-limit", action="store_true",
                    help="panda only: lift the actuator torque ceiling (control)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None, help="write the table here")
    args = ap.parse_args()

    L = args.T_ini + args.N
    sys_ = (panda_sys(args.seed, L, args.sigma, args.no_force_limit, args.theta,
                      args.servo_scale)
            if args.system == "panda" else reacher_sys(args.seed, L, args.sigma))
    rng = np.random.default_rng(args.seed + 1)
    n = args.n_fit + args.n_val
    if args.n_fit < sys_.src_dim:
        print(f"WARNING: n_fit={args.n_fit} < source dim {sys_.src_dim}; "
              f"J is underdetermined and E_self will be optimistic.")

    # Fixed, unit-free channel weighting, measured from collection -- this is what
    # keeps the Panda's rad-vs-metre mix (12-select-dpc.md caveat 3) from letting
    # the 7 joint channels swamp the 3 tip channels, without depending on the
    # ladder itself.
    sz, sy = sys_.zy_scales(args.T_ini, args.N)

    eps_sorted = sorted(args.eps, reverse=True)     # large -> small, so E_cross
    rows, J_prev = [], None                         # always uses the coarser J
    for eps in eps_sorted:
        clipped_before = sys_.n_clipped
        dz, dy, dq = ensemble(sys_, eps, n, rng, args.T_ini)
        dzn, dyn = dz / sz, dy / sy
        fit, val = slice(None, args.n_fit), slice(args.n_fit, None)
        J, k, tail = fit_jacobian(dzn[fit], dyn[fit], sys_.src_dim)
        rows.append({
            "eps": eps,
            "r_z": float(np.median(np.linalg.norm(dzn, axis=1))),
            "dq_rad": float(np.median(dq)),
            # `r_y` is the noise-floor detector. It must fall in proportion to
            # `r_z`; where it stops falling, the response has hit MuJoCo's solver
            # resolution and every error below that radius is measuring
            # arithmetic, not the plant.
            "r_y": float(np.median(np.linalg.norm(dyn, axis=1))),
            "E_self": rel_err(dyn[val], dzn[val], J),
            "E_cross": rel_err(dyn[val], dzn[val], J_prev) if J_prev is not None else np.nan,
            "rank": k,
            "svd_tail": tail,
            "clip_frac": (sys_.n_clipped - clipped_before) / (n * L),
        })
        J_prev = J
        r = rows[-1]
        print(f"eps={eps:<8g} r_z={r['r_z']:<9.3g} r_y={r['r_y']:<9.3g} "
              f"dq={r['dq_rad']:<8.3g}rad  E_self={r['E_self']:<8.4f} "
              f"E_cross={r['E_cross']:<8.4f} rank={k:<4d} tail={tail:.1e} "
              f"clip={r['clip_frac']:.1%}")

    print(f"\n{args.system}: anchor q0 = {np.round(sys_.q0, 3)}")
    print(f"source dim {sys_.src_dim}, z dim {len(sz)}, y_f dim {len(sy)}, "
          f"n_fit={args.n_fit} n_val={args.n_val}")
    print("E_self=1.0 is the 'no response' baseline; E_cross uses the Jacobian "
          "fitted one radius COARSER.")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()

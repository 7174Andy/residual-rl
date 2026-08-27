"""Per-step action decomposition for the residual and vanilla arms.

Journey 13 states the residual's remaining deficit as a mechanism -- "anchored to
the clone's steering and parks ~0.7 mm further from center" -- and that sentence
is inference, not measurement. This dumps what would measure it.

The composition is `u = clip(u_base + rho * a_res)`, and the aggregate rows can
only see `u`. Three quantities that discriminate between the candidate causes are
invisible once `u` is clipped, so all four are logged per step:

    a_res       what the policy emitted
    u_base      the frozen clone's action at the SAME state (env-cached)
    u_pre       u_base + rho * a_res, BEFORE the clip -- saturation lives here
    u_applied   what the plant integrated (post-clip; == info["action"])

Everything is split by PRE-step distance, because the one term the residual
loses to vanilla is station-keeping (journey 13: 38.6 vs 39.5 in-tolerance steps
of 50), and an episode-aggregate number cannot see a terminal-phase effect. The
phase key is the distance the policy SAW when it chose, not the distance that
resulted.

A separate script rather than a `--dump` flag on
`scripts/decompose_reacher_returns.py`: that one's numbers are quoted in
journey 13's decomposition table, and a diagnostic has no business changing the
shape of a published-number script.

    uv run python scripts/diag_reacher_residual.py
    uv run python scripts/diag_reacher_residual.py --episodes 20   # quick look
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

import reacher  # noqa: F401  registers the Gym ID
from reacher.eval import ClonePolicy
from reacher.residual_env import ResidualSelectEnv
from rl.clone import load_clone
from rl.sb3 import load_policy

COLS = ("scen", "step", "dist_pre", "dist_post", "a_res", "u_base", "u_pre",
        "u_applied", "tip", "goal", "d_tip")


def _row(scen, t, dist_pre, info, a_res, u_base, u_pre, tip_pre, base):
    """One long-form record. `u_base` is NaN for vanilla, which has no base."""
    tip = base.tip
    return dict(scen=scen, step=t, dist_pre=dist_pre, dist_post=info["dist"],
                a_res=a_res, u_base=u_base, u_pre=u_pre,
                u_applied=np.asarray(info["action"], dtype=np.float64),
                tip=tip, goal=base.goal.copy(), d_tip=tip - tip_pre)


def roll_residual(env, model, frac, qpos, goal, scen):
    obs, info = env.reset(seed=0, options={"qpos": qpos, "goal": goal})
    rows = []
    for t in range(env.base.max_steps):
        a_res = np.clip(model.predict(obs, deterministic=True)[0], -1.0, 1.0)
        u_base = np.asarray(env.u_base, dtype=np.float64)   # cached for THIS state
        u_pre = u_base + frac * env.half_range * a_res
        dist_pre, tip_pre = float(info["dist"]), env.base.tip
        obs, _r, _term, trunc, info = env.step(a_res)
        rows.append(_row(scen, t, dist_pre, info, np.asarray(a_res, dtype=np.float64),
                         u_base, u_pre, tip_pre, env.base))
        if trunc:
            break
    return rows


def roll_vanilla(env, model, qpos, goal, scen):
    base = env.unwrapped
    _obs, info = env.reset(seed=0, options={"qpos": qpos, "goal": goal})
    rows = []
    nan = np.full(2, np.nan)
    for t in range(base.max_steps):
        a = np.clip(model.predict(base.build_obs(), deterministic=True)[0], -1.0, 1.0)
        dist_pre, tip_pre = float(info["dist"]), base.tip
        _obs, _r, _term, trunc, info = env.step(a)
        rows.append(_row(scen, t, dist_pre, info, np.asarray(a, dtype=np.float64),
                         nan, np.asarray(a, dtype=np.float64), tip_pre, base))
        if trunc:
            break
    return rows


def roll_clone(env, predictor, qpos, goal, scen):
    """The base ALONE, closed loop. Needed to answer where the clone actually
    ends up: `u_base` logged along the residual's trajectory says what the clone
    would command at the RESIDUAL's states, not where the clone parks by itself.
    """
    base = env.unwrapped
    policy = ClonePolicy(predictor)
    _obs, info = env.reset(seed=0, options={"qpos": qpos, "goal": goal})
    rows = []
    nan = np.full(2, np.nan)
    for t in range(base.max_steps):
        u = np.clip(policy(env, info), -1.0, 1.0)
        dist_pre, tip_pre = float(info["dist"]), base.tip
        _obs, _r, _term, trunc, info = env.step(u)
        rows.append(_row(scen, t, dist_pre, info, nan, nan,
                         np.asarray(u, dtype=np.float64), tip_pre, base))
        if trunc:
            break
    return rows


def parking(z_scen, z_tip, z_goal):
    """Final tip-minus-goal vector per scenario -- WHERE an arm ends up, not
    just how far. Direction is the whole point: a shared direction across arms
    means one is dragging the other."""
    out = []
    for k in np.unique(z_scen):
        m = z_scen == k
        out.append(z_tip[m][-1] - z_goal[m][-1])
    return np.array(out)


def stack(rows):
    return {c: np.array([r[c] for r in rows]) for c in COLS}


def check(d, frac, arm):
    """The decomposition must reproduce what the plant actually integrated.

    This is the assert that catches the bug this script is most likely to have:
    `u_base` is cached one step ahead of `step()`, so reading it at the wrong
    moment silently shifts the whole log by one control step and every statistic
    below stays plausible. If clip(u_pre) != u_applied, the pairing is wrong.
    """
    err = np.abs(np.clip(d["u_pre"], -1.0, 1.0) - d["u_applied"]).max()
    assert err < 1e-9, f"{arm}: u_pre/u_applied misaligned by {err:.2e}"
    if arm == "residual":
        recon = np.abs(d["u_base"] + frac * d["a_res"] - d["u_pre"]).max()
        assert recon < 1e-9, f"{arm}: u_pre != u_base + frac*a_res ({recon:.2e})"


def LIMP(x):
    """Fraction of steps commanding effectively nothing."""
    return float(np.mean(np.all(np.abs(x) < 0.01, axis=1)))


def report(d, arm, tol, frac):
    near = d["dist_pre"] < tol
    far = ~near
    print(f"\n{arm}  ({near.sum()} near-phase steps of {len(near)}, "
          f"near := pre-step distance < {tol * 1e3:.0f} mm)")
    for lbl, m in (("far ", far), ("near", near)):
        if not m.any():
            print(f"  {lbl}: no steps")
            continue
        pre, app, a = d["u_pre"][m], d["u_applied"][m], d["a_res"][m]
        clipped = np.abs(pre) > 1.0 + 1e-12
        rail = np.abs(app) >= 1.0 - 1e-9
        # alignment of the achieved tip motion with the direction to the goal
        to_goal = d["goal"][m] - (d["tip"][m] - d["d_tip"][m])
        step_len = np.linalg.norm(d["d_tip"][m], axis=1)
        ok = (step_len > 1e-9) & (np.linalg.norm(to_goal, axis=1) > 1e-9)
        cos_move = np.full(len(step_len), np.nan)
        cos_move[ok] = np.einsum("ij,ij->i", d["d_tip"][m][ok], to_goal[ok]) / (
            step_len[ok] * np.linalg.norm(to_goal[ok], axis=1))
        print(f"  {lbl}: clip {clipped.any(axis=1).mean():5.1%} of steps "
              f"({clipped.mean():5.1%} of components)   "
              f"on-rail |u|=1 {rail.any(axis=1).mean():5.1%}   "
              f"mean|u|^2 {np.mean(np.sum(app ** 2, axis=1)):6.3f}   "
              f"mean|a_res| {np.abs(a).mean():5.3f}   "
              f"|a_res|>0.9 {(np.abs(a) > 0.9).any(axis=1).mean():5.1%}   "
              f"cos(move, to-goal) {np.nanmean(cos_move):+5.2f}")
        # "limp" = both components under 0.01. On a damped planar arm with no
        # gravity this is how you HOLD a position, so the fraction of terminal
        # steps spent limp is the station-keeping solution, not an absence of one.
        print(f"        limp (|u|<0.01 both components) {LIMP(app):5.1%}   "
              f"|u| median {np.median(np.abs(app)):.4f}  p90 "
              f"{np.percentile(np.abs(app), 90):.4f}")
        if arm == "residual":
            # Is the policy's authority spent ADDING to the base or CANCELLING it?
            r, b = frac * a, d["u_base"][m]
            nr, nb = np.linalg.norm(r, axis=1), np.linalg.norm(b, axis=1)
            good = (nr > 1e-9) & (nb > 1e-9)
            cos_rb = np.einsum("ij,ij->i", r[good], b[good]) / (nr[good] * nb[good])
            shrinks = np.linalg.norm(pre, axis=1) < nb
            print(f"        cancellation: cos(rho*a_res, u_base) "
                  f"{cos_rb.mean():+5.2f}   anti-parallel (cos<0) "
                  f"{(cos_rb < 0).mean():5.1%}   |u_pre|<|u_base| on "
                  f"{shrinks.mean():5.1%} of steps")
            # u = 0 is the terminal optimum, and the residual can only reach it
            # by tracking a_res = -u_base/rho -- a target that moves every step.
            # These three lines are why it misses: it OUT-commands its own base.
            print(f"        base:  limp {LIMP(b):5.1%}   |u_base| median "
                  f"{np.median(np.abs(b)):.4f}   mean|u_base|^2 "
                  f"{np.mean(np.sum(b ** 2, axis=1)):6.3f} vs applied "
                  f"{np.mean(np.sum(app ** 2, axis=1)):6.3f}")
            print(f"        |rho*a_res|/|u_base| median {np.median(nr[good] / nb[good]):5.2f}"
                  f"   out-commands base {np.mean(nr[good] > nb[good]):5.1%}"
                  f"   flips base sign {(np.sign(app) != np.sign(b)).any(axis=1).mean():5.1%}")
            print(f"        a_res that would give u=0: median |.| "
                  f"{np.median(np.abs(b / frac)):.4f}   emitted: "
                  f"{np.median(np.abs(a)):.4f}")


def report_parking(C, R, V):
    """Does the residual end up where the CLONE ends up?

    Tests the "the base parks in the wrong spot and drags the residual there"
    hypothesis. Magnitude alone cannot: the residual could be closer to the goal
    and still be pulled along the clone's error direction. So compare the final
    error VECTORS, with vanilla -- which owes the clone nothing -- as the control
    for how much alignment two independent arms show by chance.
    """
    e = {k: parking(z["scen"], z["tip"], z["goal"])
         for k, z in (("clone", C), ("residual", R), ("vanilla", V))}
    n = {k: np.linalg.norm(v, axis=1) for k, v in e.items()}
    print("\nfinal tip-minus-goal, per episode")
    for k in ("clone", "residual", "vanilla"):
        print(f"  {k:9} |error| median {np.median(n[k]) * 1e3:6.2f} mm   "
              f"p90 {np.percentile(n[k], 90) * 1e3:6.2f} mm")

    ec, nc = e["clone"], n["clone"]
    print("\nalignment with where the clone ended up (cos of the error vectors)")
    for k in ("residual", "vanilla"):
        ok = (nc > 1e-9) & (n[k] > 1e-9)
        cos = np.einsum("ij,ij->i", e[k][ok], ec[ok]) / (n[k][ok] * nc[ok])
        # component of this arm's error ALONG the clone's error direction
        along = np.einsum("ij,ij->i", e[k][ok], ec[ok]) / nc[ok]
        print(f"  {k:9} median cos {np.median(cos):+5.2f}   same side "
              f"{np.mean(cos > 0):5.1%}   median component along the clone's "
              f"error {np.median(along) * 1e3:+6.2f} mm "
              f"({np.median(along / nc[ok]):+5.1%} of it)")
    return e


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", default="data/reacher_scenarios_v1.npz")
    p.add_argument("--clone", default="data/dagger_clone_r3.pt")
    p.add_argument("--residual", default="data/reacher_ckpt_seeds/resf2_s0.zip",
                   help="default is the 400k frac-2.0 headline arm")
    p.add_argument("--residual-frac", type=float, default=2.0,
                   help="must match the frac the checkpoint was trained with")
    p.add_argument("--vanilla", default="data/reacher_vanilla_400k.zip")
    p.add_argument("--algo", default="sac")
    p.add_argument("--episodes", type=int, default=None,
                   help="scenarios to roll (default: all in the file)")
    p.add_argument("--tol", type=float, default=0.01,
                   help="phase split threshold; the env's reach tolerance")
    p.add_argument("--out", default="data/reacher_diag_steps.npz")
    args = p.parse_args()

    with np.load(args.scenarios) as z:
        n = args.episodes or len(z["qpos"])
        eps = [(z["qpos"][i], z["goal"][i]) for i in range(n)]

    res_env = ResidualSelectEnv(clone_path=args.clone,
                                residual_frac=args.residual_frac)
    res = load_policy(args.residual, algo=args.algo, device="cpu")
    R = stack([r for i, (q, g) in enumerate(eps)
               for r in roll_residual(res_env, res, args.residual_frac, q, g, i)])
    res_env.close()

    clone_env = gym.make("ReacherGoal-v0")
    predictor = load_clone(args.clone, device="cpu")
    C = stack([r for i, (q, g) in enumerate(eps)
               for r in roll_clone(clone_env, predictor, q, g, i)])
    clone_env.close()

    van_env = gym.make("ReacherGoal-v0")
    van = load_policy(args.vanilla, algo=args.algo, device="cpu")
    V = stack([r for i, (q, g) in enumerate(eps)
               for r in roll_vanilla(van_env, van, q, g, i)])
    van_env.close()

    check(R, args.residual_frac, "residual")
    check(V, args.residual_frac, "vanilla")
    check(C, args.residual_frac, "clone")

    print(f"{n} scenarios, frac {args.residual_frac}, "
          f"residual {args.residual}, vanilla {args.vanilla}")
    report(R, "residual", args.tol, args.residual_frac)
    report(V, "vanilla", args.tol, args.residual_frac)
    report(C, "clone (alone)", args.tol, args.residual_frac)
    report_parking(C, R, V)

    np.savez_compressed(args.out, **{f"res_{k}": v for k, v in R.items()},
                        **{f"van_{k}": v for k, v in V.items()},
                        **{f"clo_{k}": v for k, v in C.items()},
                        frac=args.residual_frac, tol=args.tol)
    print(f"\nwrote {args.out}  ({len(R['step'])} residual rows, "
          f"{len(V['step'])} vanilla rows)")


if __name__ == "__main__":
    main()

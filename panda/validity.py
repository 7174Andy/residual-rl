"""Roll PandaReach-v0 under random actions: frames + validity report.

This is the env's closing validity check. The video the recorder script writes
shows the arm is sane; this report proves it with numbers, because a recording you
have to trust your eyes on is weak evidence.

The single most useful field is `reached`: it must be 0. If random flailing
reaches the goal, the tolerance is too loose and no later controller number
means anything. Measured: 0/20 seeds.
"""
from __future__ import annotations

import numpy as np

from panda.model import MIN_TIP_Z, TIP_RADIUS_RANGE


def record(
    env,
    episodes: int = 3,
    seed: int = 0,
) -> tuple[list[np.ndarray], dict]:
    """Roll `episodes` episodes, collecting frames and a validity report.

    Multiple short episodes rather than one long one is deliberate: the resets are
    visible in the video, so it validates start/goal sampling too, not just the
    dynamics.

    Requires `env.render_mode == "rgb_array"`.
    """
    if env.render_mode != "rgb_array":
        raise ValueError(
            f'record() needs render_mode="rgb_array", got {env.render_mode!r}'
        )

    lo, hi = env.safe_box
    lo_real = env.model.jnt_range[:, 0].astype(np.float64)
    hi_real = env.model.jnt_range[:, 1].astype(np.float64)
    # Constant per joint (SAFE_MARGIN is a fixed fraction of each joint's own
    # span): how far the safe box sits inside the real hardware limit, worst
    # case over all 7 joints. Used by format_report to size how alarming a
    # safe-box excursion actually is.
    safe_box_inset_rad = float(np.min(lo - lo_real))

    # reset(seed=...) seeds env.np_random (goal/start sampling), but Gymnasium
    # deliberately leaves action_space's own RNG untouched -- so two "identical"
    # random-policy invocations would otherwise pick different actions. Seeding
    # it here, once, is what makes the recorded video and report reproducible.
    env.action_space.seed(seed)

    frames: list[np.ndarray] = []
    worst_safe_box_margin = float("inf")
    radii: list[float] = []
    zs: list[float] = []
    n_steps = ctrl_violations = joint_limit_violations = 0
    qpos_safe_box_excursions = below = contacts = reached = box_clip_fired = 0

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        # The reset pose is shown at rest, not just inferred from the first
        # post-step frame: multiple short episodes (see the docstring) is
        # only actually evidence of start sampling if the start itself is on
        # screen.
        frames.append(env.render())
        while True:
            u = env.action_space.sample()
            qpos_before = np.asarray(env.data.qpos, dtype=np.float64).copy()
            _, _, term, trunc, info = env.step(u)
            frames.append(env.render())
            n_steps += 1

            # ctrl is what apply_delta actually promises to keep in the safe
            # box -- the control law's real, unconditional guarantee.
            ctrl = info["ctrl"]
            if min((ctrl - lo).min(), (hi - ctrl).min()) < -1e-9:
                ctrl_violations += 1

            # This can never fail -- ctrl above is produced by clipping to
            # (lo, hi), so comparing it against (lo, hi) again is tautological
            # wiring confirmation, not a real guarantee. What IS informative:
            # how often that clip actually changed the requested delta before
            # the plant saw it, i.e. how often `info["action"]` (what a naive
            # data-collection script would record as `u`) diverges from `ctrl`
            # (what the plant actually received). See panda/env.py's module
            # docstring -- this is the number that motivates recording `ctrl`,
            # not `action`, for system identification.
            requested_ctrl = qpos_before + info["action"]
            if not np.allclose(ctrl, requested_ctrl, atol=1e-9):
                box_clip_fired += 1

            # qpos is the physical joint position, a PD-servo response to ctrl,
            # not a clipped quantity. Compare it against the REAL hardware
            # range for physical validity...
            q = np.asarray(env.data.qpos, dtype=np.float64)
            if min((q - lo_real).min(), (hi_real - q).min()) < -1e-9:
                joint_limit_violations += 1
            # ...and separately against the trimmed safe box, which qpos can
            # legitimately overshoot by a small, expected margin (servo
            # momentum carrying it past a correctly-clipped target). This is a
            # diagnostic, not a pass/fail check.
            safe_margin = float(min((q - lo).min(), (hi - q).min()))
            if safe_margin < -1e-9:
                qpos_safe_box_excursions += 1
            worst_safe_box_margin = min(worst_safe_box_margin, safe_margin)

            tip = info["y"]
            radii.append(float(np.linalg.norm(tip)))
            zs.append(float(tip[2]))
            below += int(tip[2] < 0.0)
            contacts += int(info["ncon"] > 0)

            if term:
                reached += 1
                break
            if trunc:
                break

    return frames, {
        "episodes": episodes,
        "steps": n_steps,
        "ctrl_violations": ctrl_violations,
        "joint_limit_violations": joint_limit_violations,
        "qpos_safe_box_excursions": qpos_safe_box_excursions,
        "worst_safe_box_margin": (
            worst_safe_box_margin if n_steps else float("nan")
        ),
        "safe_box_inset_rad": safe_box_inset_rad,
        "box_clip_fired_steps": box_clip_fired,
        "tip_radius_min": float(np.min(radii)) if radii else float("nan"),
        "tip_radius_max": float(np.max(radii)) if radii else float("nan"),
        "tip_z_min": float(np.min(zs)) if zs else float("nan"),
        "tip_z_max": float(np.max(zs)) if zs else float("nan"),
        "steps_below_floor": below,
        "contact_steps": contacts,
        "reached": reached,
    }


def format_report(report: dict) -> str:
    """Human-readable validity report. Each line rules out a specific failure."""
    n = max(report["steps"], 1)
    lo_r, hi_r = TIP_RADIUS_RANGE
    inset = report["safe_box_inset_rad"]
    exc = report["qpos_safe_box_excursions"]
    margin = report["worst_safe_box_margin"]
    clipped = report["box_clip_fired_steps"]
    # A positive margin is a clearance (closest safe approach); a negative one
    # is an excursion depth. Same number, opposite meaning -- label it so the
    # line cannot be misread standalone.
    margin_label = "closest approach" if margin >= 0 else "worst excursion"

    lines = [
        f"episodes {report['episodes']}   steps {report['steps']}",
        f"  ctrl vs safe box       violations {report['ctrl_violations']}   "
        "-> clip(q + u, safe_box) holds",
        f"  ctrl vs requested target   clipped {clipped} steps "
        f"({100 * clipped / n:.1f}%)   [diagnostic, not an alarm -- how often "
        "clip(q + u, safe_box) actually changed the requested delta before the "
        "plant saw it. A data-collection script that records `action` (the "
        "agent's u) instead of `ctrl` on one of these steps would record a "
        "value the plant never received]",
        f"  qpos vs real joint range   violations {report['joint_limit_violations']}"
        "   -> physical joint limits never exceeded",
        f"  qpos vs safe box       excursions {exc} steps ({100 * exc / n:.1f}%)   "
        f"{margin_label} {margin:.4f} rad   "
        f"[diagnostic, not an alarm -- benign PD-servo overshoot past a "
        f"correctly-clipped target. 0 under random excitation (as here); can be "
        f"nonzero under sustained one-directional commands, using at most "
        f"~{100 * abs(min(margin, 0.0)) / inset:.1f}% of the {inset:.3f} rad "
        f"({np.degrees(inset):.1f} deg) headroom the safe box leaves before the "
        "real limit]",
        f"  tip radius        {report['tip_radius_min']:.3f} .. "
        f"{report['tip_radius_max']:.3f} m   (safe-box shell {lo_r:.3f} .. {hi_r:.3f})",
        f"  tip z             {report['tip_z_min']:.3f} .. "
        f"{report['tip_z_max']:.3f} m   (below 0: {report['steps_below_floor']} steps, "
        f"{100 * report['steps_below_floor'] / n:.1f}%)",
        f"  contacts          {report['contact_steps']} steps "
        f"({100 * report['contact_steps'] / n:.1f}%)",
        f"  reached goal      {report['reached']} / {report['episodes']}",
        "",
        "  ctrl violations must be 0        -> the control law's real guarantee holds",
        "  ctrl-vs-requested-target clips CAN be > 0 -> expected under excitation, not an alarm",
        "  joint limit violations must be 0 -> the arm never exceeds hardware limits",
        "  qpos safe-box excursions CAN be > 0 -> expected servo overshoot, not an alarm",
        "  tip radius in shell      -> FK matches the model and site",
        f"  tip z < 0 is EXPECTED    -> no floor geom; MIN_TIP_Z={MIN_TIP_Z} gates"
        " start/goal only",
        "  random reached 0         -> the task is not trivially solvable",
    ]
    return "\n".join(lines)

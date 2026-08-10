"""panda/validity.py: frame production and the numeric validity report.

The encoder is monkeypatched where it appears, so these tests need no ffmpeg and
write no MP4 -- `encode_video` itself is already covered by
tests/test_video_encoding.py.
"""
from __future__ import annotations

import numpy as np

from panda import model as pm
from panda import validity
from panda.env import PandaReachEnv


def test_record_yields_one_frame_per_step_plus_one_reset_frame_per_episode():
    """record() now also captures the reset pose at rest (M7): one extra
    frame per episode, before any step. report["steps"] still counts only
    env.step() calls, since that's the denominator format_report uses for
    every per-step percentage -- so the relationship is steps + episodes, not
    steps alone.
    """
    env = PandaReachEnv(render_mode="rgb_array", max_steps=10)
    frames, report = validity.record(env, episodes=2, seed=0)
    env.close()
    assert report["episodes"] == 2
    assert len(frames) == report["steps"] + report["episodes"]
    assert report["steps"] == 20  # random play never reaches, so no early exit
    for f in frames[:3]:
        assert f.shape == (480, 640, 3) and f.dtype == np.uint8


def test_report_shows_no_ctrl_or_joint_limit_violations():
    """The clip(q + u, safe_box) law must hold under adversarial random input,
    and qpos must never exceed the arm's real hardware range -- these are the
    two hard guarantees. qpos vs the trimmed safe box is a separate diagnostic
    (pinned at 0 under random excitation by
    test_report_shows_no_safe_box_excursions_under_random_excitation) and is
    not asserted here.
    """
    env = PandaReachEnv(render_mode="rgb_array", max_steps=25)
    _, report = validity.record(env, episodes=2, seed=1)
    env.close()
    assert report["ctrl_violations"] == 0
    assert report["joint_limit_violations"] == 0


def test_report_measures_how_often_the_box_clip_fires():
    """ctrl_violations (above) is tautological -- apply_delta produces ctrl by
    clipping to (lo, hi), so comparing it against (lo, hi) again cannot fail.
    box_clip_fired_steps is the informative sibling: how often that clip
    actually changed the requested delta before the plant saw it. Under
    random excitation this should fire on a sizeable fraction of steps
    (measured 24-48% across excitation conditions during design), not be
    near-zero.
    """
    env = PandaReachEnv(render_mode="rgb_array", max_steps=30)
    _, report = validity.record(env, episodes=3, seed=0)
    env.close()
    assert "box_clip_fired_steps" in report
    assert 0 <= report["box_clip_fired_steps"] <= report["steps"]
    assert report["box_clip_fired_steps"] > 0
    text = validity.format_report(report)
    assert "ctrl vs requested target" in text


def test_report_tip_stays_in_measured_workspace_shell():
    env = PandaReachEnv(render_mode="rgb_array", max_steps=25)
    _, report = validity.record(env, episodes=2, seed=2)
    env.close()
    lo_r, hi_r = pm.TIP_RADIUS_RANGE
    assert report["tip_radius_min"] >= lo_r - 1e-6
    assert report["tip_radius_max"] <= hi_r + 1e-6


def test_random_policy_does_not_solve_the_task():
    """The load-bearing assertion.

    If random flailing reaches a 0.05 m ball, the task is trivial and no later
    controller number means anything. Measured: 0/20 seeds reach.
    """
    env = PandaReachEnv(render_mode="rgb_array")
    _, report = validity.record(env, episodes=3, seed=0)
    env.close()
    assert report["reached"] == 0


def test_report_shows_no_safe_box_excursions_under_random_excitation():
    """qpos_safe_box_excursions pins the benign-overshoot-absent case.

    A since-removed DLS-IK oracle used to demonstrate the *nonzero* case: chasing
    a goal near the safe-box edge for many consecutive steps let PD-servo
    momentum carry qpos a hair past the safe box (not the real hardware limit).
    With only random excitation left in this codebase, nothing drives qpos
    consistently enough in one direction to produce that, so this now only pins
    the 0 case -- it is not a claim that the count can never be nonzero.
    """
    env = PandaReachEnv(render_mode="rgb_array")
    _, report = validity.record(env, episodes=3, seed=0)
    env.close()
    assert report["qpos_safe_box_excursions"] == 0


def test_random_policy_is_reproducible_given_a_seed():
    """The video and its report ARE the environment's evidence -- "run this and
    see what I saw" has to actually hold. env.reset(seed=...) seeds env.np_random
    (goal/start sampling) but Gymnasium leaves action_space's own RNG untouched,
    so record() must seed the action space itself for the random policy's
    actions to replay identically.
    """
    env = PandaReachEnv(render_mode="rgb_array", max_steps=20)
    _, report1 = validity.record(env, episodes=2, seed=0)
    _, report2 = validity.record(env, episodes=2, seed=0)
    env.close()
    assert report1 == report2


def test_format_report_mentions_the_load_bearing_lines():
    env = PandaReachEnv(render_mode="rgb_array", max_steps=10)
    _, report = validity.record(env, episodes=1, seed=0)
    env.close()
    text = validity.format_report(report)
    for token in ("violations", "tip radius", "contacts", "reached"):
        assert token in text


def test_cli_passes_frames_to_the_encoder(monkeypatch, tmp_path):
    """Smoke-test the argparse wrapper without touching ffmpeg."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).parent.parent / "scripts" / "record_panda_video.py"
    spec = importlib.util.spec_from_file_location("record_panda_video", path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    seen = {}

    def fake_encode(frames, out, fps):
        seen.update(n=len(frames), path=out, fps=fps)
        return True

    monkeypatch.setattr(cli, "encode_video", fake_encode)
    out = tmp_path / "clip.mp4"
    monkeypatch.setattr(
        "sys.argv",
        ["record_panda_video.py", "--episodes", "1", "--max-steps", "8",
         "--out", str(out)],
    )
    cli.main()
    # 8 step frames + 1 reset frame (M7: the reset pose is now also captured).
    assert seen["n"] == 9
    assert seen["fps"] == 50
    assert seen["path"] == str(out)

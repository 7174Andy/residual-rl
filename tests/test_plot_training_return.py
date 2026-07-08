"""Smoke tests for scripts/plot_training_return.py (both input modes)."""
import csv
import subprocess
import sys

SCRIPT = "scripts/plot_training_return.py"


def _run(args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)


def test_plot_from_curve_csv(tmp_path):
    curve = tmp_path / "curve.csv"
    with open(curve, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "ep_rew_mean"])
        for e in range(1, 21):
            w.writerow([e, -1000.0 + 10 * e])  # steadily improving
    out = tmp_path / "curve.png"
    r = _run(["--input", str(curve), "--out", str(out)])
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.stat().st_size > 0


def test_plot_from_monitor_csv(tmp_path):
    # SB3 Monitor format: a leading '#{...}' metadata line, then an r,l,t table.
    mon = tmp_path / "run.monitor.csv"
    with open(mon, "w") as f:
        f.write('#{"t_start": 0}\n')
        f.write("r,l,t\n")
        for i in range(10):
            f.write(f"{-100.0 + i},200,{i * 0.1}\n")
    out = tmp_path / "run.png"
    r = _run(["--monitor", str(mon), "--out", str(out), "--window", "3"])
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.stat().st_size > 0

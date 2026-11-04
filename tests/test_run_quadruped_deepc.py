import csv
import importlib.util
import pathlib

import pytest

from rl.stats import wilson_ci

P = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_quadruped_deepc.py"
spec = importlib.util.spec_from_file_location("run_quadruped_deepc", P)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def row(i, **kw):
    d = dict(id=i, reached=True, fell=False, truncated=False, qp_fail=False,
             steps=10, final_dist=0.1, mean_solve_s=0.01)
    d.update(kw)
    return d


@pytest.fixture
def stub(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(mod, "build_canonical_quadruped_deepc", lambda **k: (None, {}))
    monkeypatch.setattr(mod, "EVAL_LOOKS_PATH", str(tmp_path / "looks.csv"))

    def fake(ctrl, info, env, scen, i):
        calls.append(i)
        return row(i, fell=(i == 3))

    monkeypatch.setattr(mod, "run_episode", fake)
    return calls


def ids(path):
    return [int(r["id"]) for r in csv.DictReader(open(path))]


def test_resume_skips_done(tmp_path, stub):
    out = tmp_path / "o.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, mod.FIELDS)
        w.writeheader()
        w.writerows([row(0), row(1)])
    mod.main(["--ids", "0,1,2", "--workers", "1", "--out", str(out)])
    assert stub == [2]
    assert ids(out) == [0, 1, 2]


def test_prune_stops(tmp_path, stub, capsys):
    out = tmp_path / "o.csv"
    mod.main(["--ids", ",".join(map(str, range(20))), "--workers", "1", "--prune", "--out", str(out)])
    assert "PRUNED" in capsys.readouterr().out
    assert ids(out) == list(range(10))


def test_eval_look_cap(tmp_path, stub):
    pathlib.Path(mod.EVAL_LOOKS_PATH).write_text("timestamp,argv,k,n\na,b,1,2\na,b,1,2\na,b,1,2\n")
    with pytest.raises(SystemExit):
        mod.main(["--split", "eval", "--workers", "1", "--out", str(tmp_path / "o.csv")])
    assert stub == []


def test_summarize():
    rows = [row(0), row(1), row(2, reached=False, fell=True), row(3, reached=False, qp_fail=True)]
    s = mod.summarize(rows)
    lo, hi = wilson_ci(2, 4)
    assert "reached 2/4" in s and f"{lo:.2f}" in s and f"{hi:.2f}" in s
    assert "falls 1" in s and "qp_fail 1" in s


def looks():
    return list(csv.DictReader(open(mod.EVAL_LOOKS_PATH)))


def test_ids_eval_is_look_and_capped(tmp_path, stub):
    mod.main(["--ids", "40,41", "--workers", "1", "--out", str(tmp_path / "a.csv")])
    r = looks()
    assert len(r) == 1 and (r[0]["k"], r[0]["n"], r[0]["pruned"]) == ("2", "2", "False")
    pathlib.Path(mod.EVAL_LOOKS_PATH).write_text("timestamp,argv,k,n,pruned\n" + "a,b,1,2,False\n" * 3)
    with pytest.raises(SystemExit):
        mod.main(["--ids", "40", "--workers", "1", "--out", str(tmp_path / "b.csv")])


def test_pruned_eval_logged(tmp_path, stub, monkeypatch):
    monkeypatch.setattr(mod, "run_episode", lambda c, i_, e, s, i: row(i, fell=(i == 43)))
    mod.main(["--ids", ",".join(map(str, range(40, 60))), "--workers", "1", "--prune",
              "--out", str(tmp_path / "a.csv")])
    r = looks()
    assert len(r) == 1 and r[0]["pruned"] == "True" and r[0]["n"] == "10"


def test_split_eval_looks_row(tmp_path, stub):
    mod.main(["--split", "eval", "--workers", "1", "--out", str(tmp_path / "a.csv")])
    r = looks()
    assert (r[0]["k"], r[0]["n"]) == ("78", "78")


def test_truncated_last_line_resumes(tmp_path, stub):
    out = tmp_path / "o.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, mod.FIELDS)
        w.writeheader()
        w.writerow(row(0))
        f.write("1,True,Fa")
    mod.main(["--ids", "0,1,2", "--workers", "1", "--out", str(out)])
    assert stub == [1, 2]
    assert ids(out) == [0, 1, 2]


def test_header_mismatch_refuses(tmp_path, stub):
    out = tmp_path / "o.csv"
    out.write_text(",".join(mod.FIELDS + ["extra"]) + "\n0,True,False,False,False,10,0.1,0.01,x\n")
    before = out.read_text()
    with pytest.raises(SystemExit):
        mod.main(["--ids", "0,1", "--workers", "1", "--out", str(out)])
    assert out.read_text() == before and stub == []


def test_config_mismatch_refuses(tmp_path, stub):
    out = tmp_path / "o.csv"
    mod.main(["--ids", "0", "--workers", "1", "--out", str(out)])
    with pytest.raises(SystemExit):
        mod.main(["--ids", "0,1", "--workers", "1", "--T-ini", "5", "--out", str(out)])
    assert stub == [0]

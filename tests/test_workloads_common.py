import json
import os
import signal

import pytest

from gso.bench.workloads import _common


def test_write_json_atomic_roundtrip(tmp_path):
    out = tmp_path / "sub" / "r.json"
    _common.write_json_atomic(out, {"a": 1})
    assert json.loads(out.read_text()) == {"a": 1}
    assert [p.name for p in out.parent.iterdir()] == ["r.json"]


def test_write_json_atomic_unserializable_leaves_nothing(tmp_path):
    out = tmp_path / "r.json"
    with pytest.raises(TypeError):
        _common.write_json_atomic(out, {"a": object()})
    assert list(tmp_path.iterdir()) == []


def test_write_json_atomic_replace_failure_removes_temp(tmp_path, monkeypatch):
    def boom(src, dst):
        raise OSError("nope")

    monkeypatch.setattr(os, "replace", boom)
    out = tmp_path / "r.json"
    with pytest.raises(OSError):
        _common.write_json_atomic(out, {"a": 1})
    assert list(tmp_path.iterdir()) == []


def test_stop_flag_handlers_set_flag_on_sigterm():
    stop = _common.StopFlag()
    old_term = signal.getsignal(signal.SIGTERM)
    old_int = signal.getsignal(signal.SIGINT)
    try:
        stop.install()
        assert not stop.is_set()
        signal.raise_signal(signal.SIGTERM)
        assert stop.is_set()
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)


def test_timed_loop_returns_none_when_stopped():
    stop = _common.StopFlag()
    calls = []

    def step():
        calls.append(1)
        if len(calls) == 15:
            stop.set()

    assert _common.timed_loop(step, 5.0, stop, warmup=3) is None


def test_timed_loop_measures():
    stop = _common.StopFlag()
    lat, elapsed = _common.timed_loop(lambda: None, 0.05, stop, warmup=2)
    assert len(lat) > 0 and elapsed >= 0.05


def test_run_cli_interrupted_writes_nothing(tmp_path, monkeypatch):
    out = tmp_path / "o.json"

    def add_args(p):
        pass

    def measure(args, stop):
        stop.set()

    monkeypatch.setattr(_common, "select_gpu", lambda n: None)
    rc = _common.run_cli("x", "d", add_args, measure,
                         ["--gpu", "0", "--duration", "1", "--out", str(out)])
    assert rc != 0
    assert list(tmp_path.iterdir()) == []


def test_run_cli_unavailable_exits_nonzero_one_line(tmp_path, monkeypatch, capsys):
    out = tmp_path / "o.json"

    def measure(args, stop):
        raise _common.WorkloadUnavailable("no thing")

    monkeypatch.setattr(_common, "select_gpu", lambda n: None)
    rc = _common.run_cli("x", "d", lambda p: None, measure,
                         ["--gpu", "0", "--duration", "1", "--out", str(out)])
    assert rc == 1
    err = capsys.readouterr().err.strip()
    assert "no thing" in err and "\n" not in err
    assert list(tmp_path.iterdir()) == []


def test_run_cli_success_writes_summary_and_extra(tmp_path, monkeypatch):
    out = tmp_path / "o.json"
    monkeypatch.setattr(_common, "select_gpu", lambda n: None)
    rc = _common.run_cli("x", "d", lambda p: None,
                         lambda a, s: ([1.0, 2.0], 2.0, {"gbps": 3.0}),
                         ["--gpu", "0", "--duration", "1", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["workload"] == "x" and data["pid"] == os.getpid()
    assert data["iterations"] == 2 and data["gbps"] == 3.0
    assert data["latency_ms"]["p50"] == 1.0

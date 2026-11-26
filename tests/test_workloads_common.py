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
    rc = _common.run_cli(
        "x",
        "d",
        add_args,
        measure,
        ["--gpu", "0", "--duration", "1", "--out", str(out)],
    )
    assert rc != 0
    assert list(tmp_path.iterdir()) == []


def test_run_cli_unavailable_exits_nonzero_one_line(tmp_path, monkeypatch, capsys):
    out = tmp_path / "o.json"

    def measure(args, stop):
        raise _common.WorkloadUnavailable("no thing")

    monkeypatch.setattr(_common, "select_gpu", lambda n: None)
    rc = _common.run_cli(
        "x",
        "d",
        lambda p: None,
        measure,
        ["--gpu", "0", "--duration", "1", "--out", str(out)],
    )
    assert rc == 1
    err = capsys.readouterr().err.strip()
    assert "no thing" in err and "\n" not in err
    assert list(tmp_path.iterdir()) == []


def test_run_cli_success_writes_summary_and_extra(tmp_path, monkeypatch):
    out = tmp_path / "o.json"
    monkeypatch.setattr(_common, "select_gpu", lambda n: None)
    rc = _common.run_cli(
        "x",
        "d",
        lambda p: None,
        lambda a, s: ([1.0, 2.0], 2.0, {"gbps": 3.0}),
        ["--gpu", "0", "--duration", "1", "--out", str(out)],
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["workload"] == "x" and data["pid"] == os.getpid()
    assert data["iterations"] == 2 and data["gbps"] == 3.0
    assert data["latency_ms"]["p50"] == 1.0


def test_run_cli_latency_workloads_declare_metric_kind(tmp_path, monkeypatch):
    out = tmp_path / "o.json"
    monkeypatch.setattr(_common, "select_gpu", lambda n: None)
    _common.run_cli(
        "x",
        "d",
        lambda p: None,
        lambda a, s: ([1.0], 1.0, {}),
        ["--gpu", "0", "--duration", "1", "--out", str(out)],
    )
    assert json.loads(out.read_text())["metric_kind"] == "latency"


def test_run_cli_throughput_only_has_null_latencies(tmp_path, monkeypatch):
    out = tmp_path / "o.json"
    monkeypatch.setattr(_common, "select_gpu", lambda n: None)
    rc = _common.run_cli(
        "x",
        "d",
        lambda p: None,
        lambda a, s: _common.ThroughputOnly(500, 2.0, {"fps": 250.0}),
        ["--gpu", "0", "--duration", "1", "--out", str(out)],
    )
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["metric_kind"] == "throughput_only"
    assert d["iterations"] == 500 and d["throughput_per_s"] == 250.0
    assert d["latency_ms"] == {"p50": None, "p95": None, "p99": None}
    assert d["fps"] == 250.0


def test_run_cli_interrupted_exception_exits_130_no_json(tmp_path, monkeypatch):
    out = tmp_path / "o.json"
    monkeypatch.setattr(_common, "select_gpu", lambda n: None)

    def measure(a, s):
        raise _common.Interrupted()

    rc = _common.run_cli(
        "x",
        "d",
        lambda p: None,
        measure,
        ["--gpu", "0", "--duration", "1", "--out", str(out)],
    )
    assert rc == _common.EXIT_INTERRUPTED
    assert list(tmp_path.iterdir()) == []


def test_file_lock_uncontended_reports_no_wait(tmp_path):
    stop = _common.StopFlag()
    with _common.file_lock(tmp_path / "k.lock", stop, lambda: None) as waited:
        assert waited == 0.0


def test_file_lock_waits_for_holder_and_notifies_once(tmp_path):
    import threading
    import time

    stop = _common.StopFlag()
    lock = tmp_path / "k.lock"
    notes = []
    got = {}

    def waiter():
        with _common.file_lock(lock, stop, lambda: notes.append(1)) as w:
            got["w"] = w

    with _common.file_lock(lock, stop, lambda: None):
        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.6)
        assert "w" not in got
    t.join(5)
    assert got["w"] >= 0.5 and notes == [1]


def test_file_lock_raises_interrupted_when_stopped_while_waiting(tmp_path):
    import threading
    import time

    stop = _common.StopFlag()
    lock = tmp_path / "k.lock"
    result = []

    def waiter():
        try:
            with _common.file_lock(lock, stop, lambda: None):
                result.append("got")
        except _common.Interrupted:
            result.append("interrupted")

    with _common.file_lock(lock, _common.StopFlag(), lambda: None):
        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.3)
        stop.set()
        t.join(5)
    assert result == ["interrupted"]


def test_stop_flag_raise_if_set():
    stop = _common.StopFlag()
    stop.raise_if_set()
    stop.set()
    with pytest.raises(_common.Interrupted):
        stop.raise_if_set()

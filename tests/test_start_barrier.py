"""--start-at 시작 장벽과 측정 창(window_start_ns/window_end_ns): 가짜 시계, GPU 없음."""

import json

import pytest

from gso.bench.workloads import _common, decode


class FakeClock:
    def __init__(self, now):
        self.now = now
        self.sleeps = []
        self.on_sleep = None

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        if self.on_sleep:
            self.on_sleep()
        self.now += s


def _run(tmp_path, monkeypatch, fake, measure, extra=()):
    monkeypatch.setattr(_common, "select_gpu", lambda n: None)
    out = tmp_path / "o.json"
    rc = _common.run_cli(
        "x", "d", lambda p: None, measure,
        ["--gpu", "0", "--duration", "0.02", "--out", str(out), *extra],
        clock=fake.clock, sleep=fake.sleep,
    )
    return rc, out


def _measure(fake):
    def measure(args, stop):
        def step():
            fake.now += 0.001

        timed = _common.timed_loop(step, args.duration, stop, warmup=3)
        return None if timed is None else (*timed, {})

    return measure


def test_waits_until_start_at_then_measures(tmp_path, monkeypatch):
    fake = FakeClock(1000.0)
    rc, out = _run(tmp_path, monkeypatch, fake, _measure(fake), ["--start-at", "1005.0"])
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["window_start_ns"] >= 1005 * 10**9
    assert d["window_start_ns"] - 1005 * 10**9 < 10**7  # no extra waiting past the barrier
    assert d["window_end_ns"] > d["window_start_ns"]
    assert sum(fake.sleeps) == pytest.approx(5.0 - 0.003, abs=0.06)  # 3 warm-up steps ran first


def test_missed_barrier_exits_1_and_writes_nothing(tmp_path, monkeypatch, capsys):
    fake = FakeClock(1000.0)
    rc, _out = _run(tmp_path, monkeypatch, fake, _measure(fake), ["--start-at", "999.0"])
    assert rc == 1
    err = capsys.readouterr().err.strip()
    assert "start barrier missed" in err and "\n" not in err
    assert list(tmp_path.iterdir()) == []


def test_stop_during_wait_writes_nothing(tmp_path, monkeypatch):
    fake = FakeClock(1000.0)

    def measure(args, stop):
        fake.on_sleep = stop.set
        return _measure(fake)(args, stop)

    rc, _out = _run(tmp_path, monkeypatch, fake, measure, ["--start-at", "1005.0"])
    assert rc == _common.EXIT_INTERRUPTED
    assert list(tmp_path.iterdir()) == []


def test_without_start_at_no_waiting_and_window_present(tmp_path, monkeypatch):
    fake = FakeClock(1000.0)
    rc, out = _run(tmp_path, monkeypatch, fake, _measure(fake))
    assert rc == 0 and fake.sleeps == []
    d = json.loads(out.read_text())
    assert isinstance(d["window_start_ns"], int) and isinstance(d["window_end_ns"], int)
    assert d["window_end_ns"] - d["window_start_ns"] >= 0.02 * 1e9 * 0.9


def test_window_present_for_throughput_only(tmp_path, monkeypatch):
    fake = FakeClock(1000.0)

    def measure(args, stop):
        _common.record_window(1000.5, 1030.5)
        return _common.ThroughputOnly(900, 30.0, {"fps": 30.0})

    rc, out = _run(tmp_path, monkeypatch, fake, measure)
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["metric_kind"] == "throughput_only"
    assert d["window_start_ns"] == int(1000.5 * 1e9)
    assert d["window_end_ns"] == int(1030.5 * 1e9)


def test_window_null_when_workload_recorded_none(tmp_path, monkeypatch):
    fake = FakeClock(1000.0)
    _rc, out = _run(tmp_path, monkeypatch, fake, lambda a, s: ([1.0], 1.0, {}))
    d = json.loads(out.read_text())
    assert d["window_start_ns"] is None and d["window_end_ns"] is None


@pytest.mark.parametrize("bad", ["nan", "inf", "-1", "0", "abc"])
def test_start_at_must_be_a_positive_finite_number(tmp_path, monkeypatch, bad):
    fake = FakeClock(1000.0)
    with pytest.raises(SystemExit) as e:
        _run(tmp_path, monkeypatch, fake, _measure(fake), ["--start-at", bad])
    assert e.value.code == 2


def test_context_is_reset_after_run_cli(tmp_path, monkeypatch):
    fake = FakeClock(1000.0)
    _run(tmp_path, monkeypatch, fake, _measure(fake), ["--start-at", "1000.5"])
    assert _common.current().start_at is None


# ---- decode: the window starts at the first progress sample at/after the barrier ----------


def test_decode_gate_without_barrier_is_warmup_index():
    assert decode.gate_start([10.0, 10.5, 11.0], 1, None) == 1


def test_decode_gate_waits_until_samples_reach_start_at():
    walls = [10.0, 10.5, 11.0, 11.5]
    assert decode.gate_start(walls, 1, 11.2) == 3
    assert decode.gate_start(walls[:3], 1, 11.2) is None  # not there yet


def test_decode_gate_not_yet_warm():
    assert decode.gate_start([10.0], None, 10.0) is None


def test_decode_gate_missed_when_warmup_ended_after_start_at():
    with pytest.raises(_common.BarrierMissed):
        decode.gate_start([10.0, 20.0], 1, 12.0)

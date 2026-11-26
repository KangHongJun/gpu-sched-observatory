import subprocess
import threading
import time
from pathlib import Path

import pytest

from gso.buffer import RingBuffer
from gso.providers.dcgm import FIELDS, DcgmProvider, parse_dmon_line
from gso.record import Quality, Scope

FIXTURE = Path(__file__).parent / "fixtures" / "dcgmi_dmon_a5000.txt"
FIXTURE_LINES = FIXTURE.read_text().splitlines(keepends=True)
DATA_ROWS = [ln for ln in FIXTURE_LINES if ln.startswith("GPU")]
IDS = [1001, 1002, 1003, 1004, 1005]
ALL = {i: True for i in IDS}
BLOCK = {"block": True}
ERR = "Error setting watches. Result: -6: Feature not supported\n"


class FakeProc:
    """Emits lines, then ends with returncode (or blocks until terminated/killed)."""

    def __init__(self, lines=(), returncode=0, block=False, ignore_term=False):
        self._lines = list(lines)
        self._final_rc = returncode
        self._block = block
        self._ignore_term = ignore_term
        self._done = threading.Event()
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.stdout = self._gen()

    def _gen(self):
        yield from self._lines
        if self._block:
            self._done.wait()
        if self.returncode is None:
            self.returncode = (
                self._final_rc if not (self.terminated or self.killed) else -15
            )

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if not self._done.wait(timeout) and self.returncode is None and self._block:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self._ignore_term:
            self._done.set()

    def kill(self):
        self.killed = True
        self._done.set()


class FakePopen:
    def __init__(self, seq=None, **proc_kwargs):
        self.kwargs = proc_kwargs
        self.seq = seq  # optional per-spawn kwargs; the last entry repeats
        self.calls = []
        self.procs = []
        self.proc = None

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        kwargs = (
            self.seq[min(len(self.calls), len(self.seq)) - 1]
            if self.seq
            else self.kwargs
        )
        self.proc = FakeProc(**kwargs)
        self.procs.append(self.proc)
        if not kwargs.get("block"):
            self.proc._done.set()
        return self.proc


def make(popen, supported=ALL, **kw):
    buf = RingBuffer(10_000)
    kw.setdefault("clock_ns", iter(range(1, 10**9)).__next__)
    p = DcgmProvider(0, "GPU-test", buf, supported, popen=popen, **kw)
    return p, buf


def wait_until(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


# ---- parse_dmon_line ----------------------------------------------------


def test_parse_fixture_lines():
    n = 0
    for ln in FIXTURE_LINES:
        r = parse_dmon_line(ln, IDS)
        if ln.startswith(("#Entity", "ID")):
            assert r is None
        else:
            n += 1
            toks = ln.split()[2:]
            assert list(r) == IDS
            assert [r[i] for i in IDS] == [float(t) for t in toks]
    assert n == len(DATA_ROWS) and n > 0
    assert sum(1 for ln in FIXTURE_LINES if ln.startswith("#Entity")) >= 2


def test_parse_na_token():
    r = parse_dmon_line("GPU 0   0.5  N/A  0.2  -  0.1\n", IDS)
    assert r == {1001: 0.5, 1002: None, 1003: 0.2, 1004: None, 1005: 0.1}


def test_parse_too_few_and_extra_columns():
    r = parse_dmon_line("GPU 0   0.5  0.4\n", IDS)
    assert r == {1001: 0.5, 1002: 0.4, 1003: None, 1004: None, 1005: None}
    r = parse_dmon_line("GPU 0   1 2 3 4 5 6 7\n", IDS)
    assert r == {1001: 1.0, 1002: 2.0, 1003: 3.0, 1004: 4.0, 1005: 5.0}


@pytest.mark.parametrize("line", ["", "   \n", "\n", ERR, "GPU", "Entity 0 1 2"])
def test_parse_non_rows(line):
    assert parse_dmon_line(line, IDS) is None


def test_parse_nonfinite():
    r = parse_dmon_line("GPU 0  nan inf -inf 0.1 0.2", IDS)
    assert r == {1001: None, 1002: None, 1003: None, 1004: 0.1, 1005: 0.2}


# ---- provider ---------------------------------------------------------------


def test_fixture_all_supported():
    fp = FakePopen(seq=[{"lines": FIXTURE_LINES}, BLOCK])
    p, buf = make(fp)
    p.start()
    assert wait_until(lambda: len(buf) == 5 * len(DATA_ROWS))
    p.stop()
    recs = buf.snapshot()
    assert len(recs) == 5 * len(DATA_ROWS)
    assert all(
        r.unit == "%" and r.source == "dcgm" and r.scope is Scope.DEVICE for r in recs
    )
    assert all(r.pid is None and r.gpu_uuid == "GPU-test" for r in recs)
    first = recs[:5]
    assert [r.metric for r in first] == [FIELDS[i] for i in IDS]
    assert len({r.ts_ns for r in first}) == 1
    assert first[0].value == pytest.approx(99.2)
    assert all(r.quality is Quality.MEASURED for r in recs)
    assert fp.calls[0] == [
        "dcgmi",
        "dmon",
        "-i",
        "0",
        "-e",
        "1001,1002,1003,1004,1005",
        "-d",
        "1000",
        "-c",
        "15",
    ]


def test_partial_support_argv_and_unsupported_records():
    sup = {1001: True, 1002: True, 1003: True, 1004: False, 1005: True}
    fp = FakePopen(seq=[{"lines": DATA_ROWS[:3]}, BLOCK])
    p, buf = make(fp, supported=sup)
    p.start()
    assert wait_until(lambda: len(buf) == 15)
    p.stop()
    argv = fp.calls[0]
    assert argv[argv.index("-e") + 1] == "1001,1002,1003,1005"
    recs = buf.snapshot()
    assert len(recs) == 15
    tensor = [r for r in recs if r.metric == "dcgm.tensor_active"]
    assert len(tensor) == 3
    assert all(r.quality is Quality.UNSUPPORTED and r.value is None for r in tensor)
    # columns map to requested ids, not positions of all five
    row = DATA_ROWS[0].split()[2:]
    five = recs[:5]
    assert five[3].metric == "dcgm.tensor_active"
    assert five[4].value == pytest.approx(float(row[3]) * 100)


def test_na_token_gives_dropped():
    fp = FakePopen(seq=[{"lines": ["GPU 0  0.5 N/A 0.2 0.0 0.1\n"]}, BLOCK])
    p, buf = make(fp)
    p.start()
    assert wait_until(lambda: len(buf) == 5)
    p.stop()
    recs = buf.snapshot()
    assert len(recs) == 5
    assert recs[1].quality is Quality.DROPPED and recs[1].value is None
    assert recs[0].quality is Quality.MEASURED


def test_no_supported_fields_no_spawn():
    fp = FakePopen()
    p, _ = make(fp, supported={i: False for i in IDS})
    p.start()
    assert fp.calls == []
    assert p.error
    assert not p.running
    p.stop()


def test_container_argv_exact():
    fp = FakePopen(block=True)
    p, _ = make(fp, interval_ms=100, dcgmi_cmd=["docker", "exec", "gso-dcgm", "dcgmi"])
    p.start()
    assert fp.calls[0] == [
        "docker",
        "exec",
        "gso-dcgm",
        "dcgmi",
        "dmon",
        "-i",
        "0",
        "-e",
        "1001,1002,1003,1004,1005",
        "-d",
        "100",
        "-c",
        "150",
    ]
    p.start()  # no-op while running
    assert len(fp.calls) == 1
    p.stop()


def test_interval_below_100_rejected():
    with pytest.raises(ValueError):
        DcgmProvider(0, "u", RingBuffer(10), ALL, interval_ms=50)


def test_child_failure_sets_error():
    fp = FakePopen(lines=["\n", ERR], returncode=1)
    p, buf = make(fp)
    p.start()
    assert wait_until(lambda: not p.running and p.error is not None)
    assert "1" in p.error and "Feature not supported" in p.error
    assert len(buf) == 0
    assert not p.running


def test_spawn_failure():
    def boom(argv, **kw):
        raise FileNotFoundError("dcgmi")

    p, buf = make(boom)
    p.start()
    assert p.error
    assert not p.running
    assert len(buf) == 0
    p.stop()


def test_stop_blocking_child():
    fp = FakePopen(lines=DATA_ROWS[:2], block=True)
    p, buf = make(fp)
    p.start()
    assert wait_until(lambda: len(buf) == 10)
    assert p.running
    t = time.monotonic()
    p.stop()
    assert time.monotonic() - t < 1.0
    assert not p.running
    assert p.error is None
    p.stop()


def test_stop_before_start():
    p, _ = make(FakePopen())
    p.stop()
    assert not p.running and p.error is None


def test_stop_child_ignoring_sigterm():
    fp = FakePopen(block=True, ignore_term=True)
    p, _ = make(fp)
    p.start()
    t = time.monotonic()
    p.stop()
    assert time.monotonic() - t < 3.0
    assert fp.proc.killed
    assert not p.running and p.error is None


def test_real_process_false():
    buf = RingBuffer(10)
    p = DcgmProvider(0, "u", buf, ALL, dcgmi_cmd=["false"], popen=subprocess.Popen)
    p.start()
    assert wait_until(lambda: p.error is not None)
    assert len(buf) == 0
    p.stop()
    assert not p.running


# ---- fix round 1 ---------------------------------------------------------------


def test_run_seconds_validation_and_count():
    with pytest.raises(ValueError):
        DcgmProvider(0, "u", RingBuffer(10), ALL, run_seconds=0)
    fp = FakePopen(block=True)
    p, _ = make(fp, interval_ms=300, run_seconds=1.0)
    p.start()
    assert fp.calls[0][-2:] == ["-c", "4"]  # ceil(1000/300)
    p.stop()


def test_respawn_after_clean_exit():
    fp = FakePopen(seq=[{"lines": DATA_ROWS[:2], "returncode": 0}, {"block": True}])
    p, buf = make(fp)
    p.start()
    assert wait_until(lambda: len(fp.calls) == 2)
    assert fp.calls[0] == fp.calls[1]
    assert wait_until(lambda: len(buf) == 10)
    assert p.error is None and p.running
    p.stop()
    assert not p.running and p.error is None
    assert len(fp.calls) == 2
    assert fp.procs[1].terminated


def test_no_respawn_after_nonzero_exit():
    fp = FakePopen(lines=DATA_ROWS[:1], returncode=3)
    p, _ = make(fp)
    p.start()
    assert wait_until(lambda: not p.running)
    assert len(fp.calls) == 1
    assert p.error and "3" in p.error


def test_exit_zero_without_data_is_error():
    fp = FakePopen(lines=["#Entity 1001\n", "ID\n", "something odd\n"], returncode=0)
    p, buf = make(fp)
    p.start()
    assert wait_until(lambda: not p.running)
    assert len(fp.calls) == 1
    assert p.error and "without data" in p.error and "something odd" in p.error
    assert len(buf) == 0


def test_stop_while_first_child_blocking_no_second_spawn():
    fp = FakePopen(block=True)
    p, _ = make(fp)
    p.start()
    p.stop()
    time.sleep(0.05)
    assert len(fp.calls) == 1
    assert fp.procs[0].terminated and not p.running


def test_reader_exception_sets_error_and_terminates_child():
    class BadBuf:
        def push(self, r):
            raise RuntimeError("boom")

    fp = FakePopen(lines=DATA_ROWS[:1], block=True)
    p = DcgmProvider(0, "u", BadBuf(), ALL, popen=fp)
    p.start()
    assert wait_until(lambda: not p.running)
    assert "RuntimeError" in p.error and "boom" in p.error
    assert fp.procs[0].terminated
    p.stop()


def test_error_uses_last_output_line():
    fp = FakePopen(
        lines=["WARNING: something\n", "Error: real problem\n"], returncode=2
    )
    p, _ = make(fp)
    p.start()
    assert wait_until(lambda: not p.running)
    assert "real problem" in p.error and "2" in p.error
    assert "WARNING" not in p.error

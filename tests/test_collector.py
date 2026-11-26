import threading
import time

from gso.buffer import RingBuffer
from gso.capability import Capabilities, DcgmStatus, GpuInfo
from gso.collector import Collector
from gso.providers.mps import MpsStatus
from gso.record import Quality, Record, Scope

DCGM_METRICS = {
    "dcgm.gr_engine_active",
    "dcgm.sm_active",
    "dcgm.sm_occupancy",
    "dcgm.tensor_active",
    "dcgm.dram_active",
}
UUID = "GPU-test-0001"


def make_caps(dcgm_ok=False, reason="host engine is not running"):
    return Capabilities(
        gpus=[GpuInfo(1, UUID, "Test GPU", "535", False)],
        dcgm=DcgmStatus(dcgm_ok, "" if dcgm_ok else reason, {1001: dcgm_ok}),
        nsys_path=None,
        is_root=False,
        mps_active=False,
    )


class FakeNvmlProvider:
    def __init__(self, fail_first=0, fail_always=False, pids=(), open_error=None):
        self.opened = 0
        self.closed = 0
        self.calls = 0
        self.fail_first = fail_first
        self.fail_always = fail_always
        self.pids = pids
        self.open_error = open_error
        self.ticks = threading.Semaphore(0)

    def open(self):
        if self.open_error:
            raise self.open_error
        self.opened += 1

    def close(self):
        self.closed += 1

    def sample(self, now_ns):
        self.calls += 1
        self.ticks.release()
        if self.fail_always or self.calls <= self.fail_first:
            raise RuntimeError("nvml boom")
        recs = [
            Record(
                now_ns,
                UUID,
                "gpu.util",
                50.0,
                "%",
                "nvml",
                Scope.DEVICE,
                Quality.MEASURED,
            )
        ]
        for pid in self.pids:
            recs.append(
                Record(
                    now_ns,
                    UUID,
                    "proc.mem_used",
                    10.0,
                    "MiB",
                    "nvml",
                    Scope.PROCESS,
                    Quality.MEASURED,
                    pid=pid,
                )
            )
        return recs


class FakeDcgm:
    def __init__(self, error=None, running=True):
        self.error = error
        self.running = running
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def build(caps=None, nvml=None, dcgm=None, mps=None, **kw):
    buf = RingBuffer(10_000)
    c = Collector(
        buf,
        caps or make_caps(),
        1,
        interval_s=0.01,
        nvml_provider=nvml or FakeNvmlProvider(),
        dcgm_provider=dcgm,
        mps_query=mps or (lambda: MpsStatus(False, [], [], None)),
        **kw,
    )
    return buf, c


def ticks(nvml, n):
    for _ in range(n):
        assert nvml.ticks.acquire(timeout=3)


def test_records_reach_buffer_and_status_filled():
    nvml = FakeNvmlProvider()
    buf, c = build(nvml=nvml)
    c.start()
    try:
        ticks(nvml, 3)
        assert wait_for(lambda: len(buf.snapshot("gpu.util")) >= 2)
        st = c.status
        assert st.gpu_uuid == UUID and st.gpu_name == "Test GPU"
        assert st.nvml_error is None
        assert st.last_sample_ns is not None
    finally:
        c.stop()


def test_sample_that_raises_does_not_stop_loop():
    nvml = FakeNvmlProvider(fail_first=1)
    buf, c = build(nvml=nvml)
    c.start()
    try:
        assert wait_for(lambda: len(buf.snapshot("gpu.util")) >= 2)
        assert nvml.calls >= 3
        assert (
            c.status.nvml_error is None
        )  # recovered: error describes the current state
    finally:
        c.stop()


def test_persistent_failure_sets_nvml_error_and_keeps_ticking():
    nvml = FakeNvmlProvider(fail_always=True)
    buf, c = build(nvml=nvml)
    c.start()
    try:
        ticks(nvml, 3)
        assert "nvml boom" in c.status.nvml_error
        assert c.status.last_sample_ns is None
        assert buf.snapshot("gpu.util") == []
    finally:
        c.stop()


def test_open_failure_sets_nvml_error():
    nvml = FakeNvmlProvider(open_error=RuntimeError("no driver"))
    _, c = build(nvml=nvml)
    c.start()
    try:
        assert wait_for(
            lambda: c.status.nvml_error and "no driver" in c.status.nvml_error
        )
    finally:
        c.stop()


def dcgm_recs(buf):
    return [r for r in buf.snapshot() if r.metric.startswith("dcgm.")]


def test_dcgm_unavailable_pushes_unsupported_each_tick():
    nvml = FakeNvmlProvider()
    buf, c = build(caps=make_caps(False, "no host engine"), nvml=nvml)
    c.start()
    try:
        ticks(nvml, 3)
        assert wait_for(lambda: len(dcgm_recs(buf)) >= 10)
    finally:
        c.stop()
    recs = dcgm_recs(buf)
    assert {r.metric for r in recs} == DCGM_METRICS
    assert all(r.quality is Quality.UNSUPPORTED and r.value is None for r in recs)
    assert len(recs) % 5 == 0
    assert c.status.dcgm_error == "no host engine"


def test_dcgm_healthy_collector_pushes_no_dcgm_records():
    nvml = FakeNvmlProvider()
    dcgm = FakeDcgm()
    buf, c = build(caps=make_caps(True), nvml=nvml, dcgm=dcgm)
    c.start()
    try:
        ticks(nvml, 3)
        assert dcgm.started == 1
    finally:
        c.stop()
    assert dcgm_recs(buf) == []
    assert c.status.dcgm_error is None
    assert dcgm.stopped >= 1


def test_dcgm_provider_error_gives_unsupported_and_error():
    nvml = FakeNvmlProvider()
    dcgm = FakeDcgm(error="dcgmi exited 1")
    buf, c = build(caps=make_caps(True), nvml=nvml, dcgm=dcgm)
    c.start()
    try:
        ticks(nvml, 2)
        assert wait_for(lambda: len(dcgm_recs(buf)) >= 5)
    finally:
        c.stop()
    assert {r.quality for r in dcgm_recs(buf)} == {Quality.UNSUPPORTED}
    assert c.status.dcgm_error == "dcgmi exited 1"


def test_gpu_index_not_in_caps_no_loop_no_exception():
    nvml = FakeNvmlProvider()
    buf = RingBuffer(100)
    c = Collector(
        buf,
        make_caps(),
        7,
        interval_s=0.01,
        nvml_provider=nvml,
        mps_query=lambda: MpsStatus(False, [], [], None),
    )
    c.start()
    time.sleep(0.05)
    assert nvml.calls == 0 and len(buf) == 0
    assert "7" in c.status.nvml_error
    c.stop()
    c.stop()


def test_stop_is_prompt_idempotent_and_cleans_up():
    nvml = FakeNvmlProvider()
    dcgm = FakeDcgm()
    buf = RingBuffer(1000)
    c = Collector(
        buf,
        make_caps(True),
        1,
        interval_s=30.0,
        nvml_provider=nvml,
        dcgm_provider=dcgm,
        mps_query=lambda: MpsStatus(False, [], [], None),
    )
    c.start()
    ticks(nvml, 1)  # loop is now waiting 30 s
    t0 = time.monotonic()
    c.stop()
    c.stop()
    assert time.monotonic() - t0 < 1.0
    assert nvml.closed == 1
    assert dcgm.stopped >= 1
    assert not c._thread.is_alive()


def test_stop_without_start_does_not_raise():
    _, c = build()
    c.stop()


def test_mps_queried_at_most_once_per_window():
    nvml = FakeNvmlProvider()
    calls = []
    now = [1_000_000_000]

    def clock():
        now[0] += 1_000_000  # 1 ms per call
        return now[0]

    def mps():
        calls.append(1)
        return MpsStatus(True, [5], [6], 100.0)

    _, c = build(nvml=nvml, mps=mps, clock_ns=clock)
    c.start()
    try:
        ticks(nvml, 5)
        assert wait_for(lambda: c.status.mps is not None)
        assert len(calls) == 1 and c.status.mps.active
        now[0] += 6_000_000_000  # past the 5 s window
        assert wait_for(lambda: len(calls) == 2)
    finally:
        c.stop()


def test_mps_query_exception_leaves_none():
    nvml = FakeNvmlProvider()

    def mps():
        raise RuntimeError("x")

    _, c = build(nvml=nvml, mps=mps)
    c.start()
    try:
        ticks(nvml, 3)
        assert c.status.mps is None
    finally:
        c.stop()


def test_process_names_from_proc_root(tmp_path):
    (tmp_path / "100").mkdir()
    (tmp_path / "100" / "comm").write_text("python3\n")
    nvml = FakeNvmlProvider(pids=(100, 200))  # 200 has no comm file
    _, c = build(nvml=nvml, proc_root=str(tmp_path))
    c.start()
    try:
        assert wait_for(lambda: c.status.process_names)
    finally:
        c.stop()
    assert c.status.process_names == {100: "python3", 200: "?"}


def test_status_snapshot_is_a_copy():
    _, c = build()
    s = c.status
    s.process_names["x"] = "y"
    assert "x" not in c.status.process_names

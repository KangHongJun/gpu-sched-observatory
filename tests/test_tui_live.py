from textual.widgets import Static

from gso.buffer import RingBuffer
from gso.capability import Capabilities, DcgmStatus, GpuInfo
from gso.collector import CollectorStatus
from gso.providers.mps import MpsStatus
from gso.record import Quality, Record, Scope, unsupported
from gso.tui.app import GsoApp

NOW = 1_000_000_000_000
UUID = "GPU-test-0001"
BAR_IDS = ["gr-engine", "sm-active", "sm-occupancy", "tensor-active", "dram-active", "nvdec"]


def caps(dcgm_ok=False, reason="host engine not running", mps=False):
    return Capabilities(
        gpus=[GpuInfo(1, UUID, "Test GPU 9000", "535", False)],
        dcgm=DcgmStatus(dcgm_ok, "" if dcgm_ok else reason, {}),
        nsys_path=None,
        is_root=False,
        mps_active=mps,
    )


def put(buf, metric, value, ts=NOW, pid=None, scope=None, quality=Quality.MEASURED):
    if value is None and quality is Quality.MEASURED:
        quality = Quality.DROPPED
    scope = scope or (Scope.PROCESS if pid is not None else Scope.DEVICE)
    buf.push(Record(ts, UUID, metric, value, "%", "t", scope, quality, pid=pid))


def full_buffer(ts=NOW):
    buf = RingBuffer(10_000)
    for m, v in [
        ("gpu.util", 37.0), ("gpu.temp", 64.0), ("gpu.power", 178.0), ("gpu.dec_util", 84.0),
        ("dcgm.gr_engine_active", 90.0), ("dcgm.sm_active", 78.0),
        ("dcgm.sm_occupancy", 42.0), ("dcgm.tensor_active", 32.0), ("dcgm.dram_active", 67.0),
    ]:
        put(buf, m, v, ts)
    return buf


def add_procs(buf, ts=NOW):
    for pid, sm, mem in [(1420, 8.0, 2457.6), (1421, 62.0, 512.0)]:
        put(buf, "proc.sm_util", sm, ts, pid=pid)
        put(buf, "proc.mem_used", mem, ts, pid=pid)
        put(buf, "proc.is_mps", 0.0, ts, pid=pid)


def make_app(buf, c=None, collector=None, now=NOW):
    return GsoApp(buf, c or caps(), 1, collector=collector, clock_ns=lambda: now)


def text(app, wid):
    return str(app.query_one(f"#{wid}", Static).content)


def screen_text(app):
    return "\n".join(str(w.content) for w in app.query(Static))


async def settle(pilot):
    await pilot.pause()
    await pilot.pause()


def unsupported_buffer(metric):
    buf = RingBuffer(100)
    buf.push(unsupported(NOW, UUID, metric, "%", "dcgm", Scope.DEVICE))
    return buf


async def test_unsupported_dcgm_shows_na():
    app = make_app(unsupported_buffer("dcgm.tensor_active"))
    async with app.run_test() as pilot:
        await settle(pilot)
        assert "N/A" in text(app, "tensor-active")


async def test_narrow_terminal_does_not_crash():
    app = make_app(RingBuffer(10))
    async with app.run_test(size=(60, 20)) as pilot:
        await settle(pilot)
        add_procs_buf = full_buffer()
        add_procs(add_procs_buf)
    app2 = make_app(add_procs_buf)
    async with app2.run_test(size=(60, 20)) as pilot:
        await settle(pilot)
        assert "1421" in text(app2, "proc-table")


async def test_measured_values_and_header():
    app = make_app(full_buffer())
    async with app.run_test() as pilot:
        await settle(pilot)
        assert "78%" in text(app, "sm-active")
        assert "42%" in text(app, "sm-occupancy")
        assert "32%" in text(app, "tensor-active")
        assert "67%" in text(app, "dram-active")
        assert "90%" in text(app, "gr-engine")
        assert "84%" in text(app, "nvdec")
        h = text(app, "header")
        assert h.startswith("GPU 1 | Test GPU 9000 |")
        assert "64 C" in h and "178 W" in h and "MPS OFF" in h
        assert "F2 Metrics | F3 Timeline | F4 Capture | F5 Compare" in text(app, "footer")


async def test_empty_buffer_is_all_na_and_no_zero_percent():
    app = make_app(RingBuffer(10))
    async with app.run_test() as pilot:
        await settle(pilot)
        for wid in BAR_IDS:
            assert "N/A" in text(app, wid)
        h = text(app, "header")
        assert h.count("N/A") == 2  # temp, power
        assert "0%" not in screen_text(app)


async def test_stale_values_are_na_not_old_numbers():
    buf = full_buffer(ts=NOW - 10_000_000_000)  # 10 s old, window is 3 s
    add_procs(buf, ts=NOW - 10_000_000_000)
    app = make_app(buf)
    async with app.run_test() as pilot:
        await settle(pilot)
        for wid in BAR_IDS:
            assert "N/A" in text(app, wid)
        assert "78" not in text(app, "sm-active")
        assert "64 C" not in text(app, "header")
        assert "1420" not in text(app, "proc-table")
        assert "0%" not in screen_text(app)


async def test_process_table_rows():
    buf = full_buffer()
    add_procs(buf)

    class C:
        status = CollectorStatus(gpu_name="Test GPU 9000", process_names={1420: "decoder"})
        started = stopped = 0

        def start(self):
            pass

        def stop(self):
            pass

    app = make_app(buf, collector=C())
    async with app.run_test() as pilot:
        await settle(pilot)
        t = text(app, "proc-table")
        for col in ("PID", "PROCESS", "SM UTIL", "VRAM", "CTX SW/s"):
            assert col in t
        rows = t.splitlines()[1:]
        assert rows[0].split()[:4] == ["1421", "?", "62%", "512M"]  # sorted by SM util desc
        assert rows[1].split()[:4] == ["1420", "decoder", "8%", "2.4G"]
        assert all(r.rstrip().endswith("N/A") and "N/A (MPS)" not in r for r in rows)


async def test_process_table_mps_active_from_caps():
    buf = full_buffer()
    add_procs(buf)
    app = make_app(buf, caps(mps=True))
    async with app.run_test() as pilot:
        await settle(pilot)
        rows = text(app, "proc-table").splitlines()[1:]
        assert len(rows) == 2 and all(r.rstrip().endswith("N/A (MPS)") for r in rows)
        assert "MPS ON" in text(app, "header")


async def test_process_table_mps_active_from_collector_status():
    buf = full_buffer()
    add_procs(buf)

    class C:
        status = CollectorStatus(mps=MpsStatus(True, [1], [2], None))

        def start(self):
            pass

        def stop(self):
            pass

    app = make_app(buf, collector=C())
    async with app.run_test() as pilot:
        await settle(pilot)
        assert "N/A (MPS)" in text(app, "proc-table")
        assert "MPS ON" in text(app, "header")


async def test_dropped_process_value_shows_na_in_cell():
    buf = full_buffer()
    put(buf, "proc.sm_util", None, pid=1500)
    put(buf, "proc.mem_used", 100.0, pid=1500)
    app = make_app(buf)
    async with app.run_test() as pilot:
        await settle(pilot)
        row = text(app, "proc-table").splitlines()[1]
        assert row.split()[:4] == ["1500", "?", "N/A", "100M"]


async def test_dcgm_unavailable_reason_on_status_line():
    app = make_app(RingBuffer(10), caps(reason="host engine is not running"))
    async with app.run_test(size=(100, 30)) as pilot:
        await settle(pilot)
        assert "DCGM: host engine is not running" in text(app, "status")


async def test_nvml_error_and_dropped_on_status_line():
    buf = RingBuffer(2)
    for i in range(5):
        put(buf, "gpu.util", 1.0, NOW + i)

    class C:
        status = CollectorStatus(nvml_error="boom")

        def start(self):
            pass

        def stop(self):
            pass

    app = make_app(buf, caps(dcgm_ok=True), collector=C())
    async with app.run_test(size=(100, 30)) as pilot:
        await settle(pilot)
        s = text(app, "status")
        assert "boom" in s and "dropped: 3" in s


async def test_function_keys_show_notice_and_q_quits():
    app = make_app(full_buffer(), caps(dcgm_ok=True))
    async with app.run_test(size=(100, 30)) as pilot:
        for key, name in (("f5", "F5"),):
            await pilot.press(key)
            await settle(pilot)
            assert f"{name}" in text(app, "status") and "not available yet" in text(app, "status")
        await pilot.press("f2")
        await settle(pilot)
        assert "not available yet" not in text(app, "status")
        await pilot.press("q")
        await pilot.pause()
    assert not app.is_running


async def test_ctrl_c_quits():
    app = make_app(RingBuffer(10))
    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause()
    assert not app.is_running


async def test_collector_started_on_mount_and_stopped_on_exit():
    class C:
        status = CollectorStatus()

        def __init__(self):
            self.log = []

        def start(self):
            self.log.append("start")

        def stop(self):
            self.log.append("stop")

    c = C()
    app = make_app(RingBuffer(10), collector=c)
    async with app.run_test() as pilot:
        await settle(pilot)
        assert c.log == ["start"]
        await pilot.press("q")
        await pilot.pause()
    assert c.log == ["start", "stop"]


async def test_history_sparklines_show_gap_when_unknown():
    buf = RingBuffer(100)
    put(buf, "gpu.util", 100.0, NOW - 1_000_000_000)
    app = make_app(buf)
    async with app.run_test(size=(100, 30)) as pilot:
        await settle(pilot)
        lines = text(app, "history").splitlines()
        assert "█" in lines[1]
        assert set(lines[2].split(None, 2)[-1]) == {"·"}

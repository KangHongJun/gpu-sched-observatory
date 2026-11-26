import asyncio
import threading
import time

from textual.widgets import Static

from gso.trace.nsys_capture import CaptureCancelled, CaptureError, CaptureUnavailable
from gso.tui.app import GsoApp
from gso.tui.capture_job import CaptureReport
from tests.test_tui_live import (
    NOW,
    add_procs,
    caps,
    full_buffer,
    screen_text,
    settle,
    text,
)

STARTED_NS = 1_700_000_000_000_000_000
TIMELINE = "GPU 1 | mode: global | window: 5.00 s | on-demand capture\n\n[hold] estimate\n    1420 |====    |\n"


def report():
    summary = {"ctx": {"per_context": [{"pid": 1420, "restores_per_s": 123.456}]}}
    return CaptureReport(summary, TIMELINE, STARTED_NS, 5)


class FakeRunner:
    def __init__(self, rep=None, raises=None, block=False, ignore_cancel=False):
        self.rep, self.raises, self.block = rep or report(), raises, block
        self.ignore_cancel = ignore_cancel
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.thread_names = []

    def __call__(self, duration_s, out_dir, caps_, gpu, cancel):
        self.calls.append((duration_s, gpu))
        self.thread_names.append(threading.current_thread().name)
        self.started.set()
        if self.block:
            while not (self.release.is_set() or (cancel.is_set() and not self.ignore_cancel)):
                time.sleep(0.01)
            if cancel.is_set() and not self.ignore_cancel:
                raise CaptureCancelled("capture cancelled")
        if self.raises:
            raise self.raises
        return self.rep


def make(runner, c=None):
    buf = full_buffer()
    add_procs(buf)
    return GsoApp(buf, c or caps(), 1, clock_ns=lambda: NOW, capture_runner=runner,
                  capture_duration_s=5)


async def until(pilot, cond, tries=300):
    for _ in range(tries):
        if cond():
            return True
        await pilot.pause(0.02)
    return False


async def start_capture(app, pilot):
    await pilot.press("f4")
    await settle(pilot)
    await pilot.click("#capture-ok")


def row(app, pid):
    return next(ln for ln in text(app, "proc-table").splitlines() if ln.startswith(str(pid)))


async def test_f4_confirm_runs_worker_and_fills_ctx_column():
    r = FakeRunner()
    app = make(r)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        assert "N/A" in row(app, 1420)
        await start_capture(app, pilot)
        assert await until(pilot, lambda: "123.5" in text(app, "proc-table"))
        assert r.calls == [(5, 1)]
        assert r.thread_names[0] != threading.main_thread().name  # worker thread, not the UI
        assert row(app, 1420).rstrip().endswith("123.5")
        assert row(app, 1421).rstrip().endswith("N/A")
        head = text(app, "proc-table").splitlines()[0]
        t = time.strftime("%H:%M:%S", time.localtime(STARTED_NS / 1e9))
        assert f"CTX SW/s (capture {t}, 5 s)" in head


async def test_confirm_text_is_honest():
    app = make(FakeRunner())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await pilot.press("f4")
        await settle(pilot)
        t = "\n".join(str(w.content) for w in app.screen.query(Static))
        assert "on-demand capture" in t and "5 s" in t and "GPU 1" in t and "GLOBAL" in t
        assert "7 %" in t
        assert "real-time" not in t.lower() and "실시간" not in t
        await pilot.click("#capture-cancel")


async def test_cancel_runs_nothing():
    r = FakeRunner()
    app = make(r)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await pilot.press("f4")
        await settle(pilot)
        await pilot.click("#capture-cancel")
        await pilot.pause(0.2)
        assert r.calls == []
        assert "capture running" not in text(app, "status")
        await pilot.press("f4")
        await settle(pilot)
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert r.calls == []


async def test_mps_overrides_ctx_column():
    app = make(FakeRunner(), caps(mps=True))
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await start_capture(app, pilot)
        await until(pilot, lambda: app._capture_report is not None)
        await settle(pilot)
        assert "N/A (MPS)" in row(app, 1420) and "123.5" not in text(app, "proc-table")


async def test_running_state_f4_ignored_and_dcgm_marked():
    r = FakeRunner(block=True)
    app = make(r)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        assert "캡처 중" not in text(app, "sm-active")
        await start_capture(app, pilot)
        assert await until(pilot, r.started.is_set)
        await settle(pilot)
        assert "capture running" in text(app, "status")
        assert "캡처 중" in text(app, "sm-active") and "[" in text(app, "sm-active")
        assert "캡처 중" not in text(app, "nvdec")  # NVML value, not DCGM
        await pilot.press("f4")
        await settle(pilot)
        assert "already running" in text(app, "status")
        assert len(r.calls) == 1
        r.release.set()
        assert await until(pilot, lambda: not app._capture_running)
        await settle(pilot)
        assert "캡처 중" not in text(app, "sm-active")
        assert "capture running" not in text(app, "status")
        assert len(r.calls) == 1


async def test_unavailable_shows_hint_and_changes_nothing_else():
    r = FakeRunner(raises=CaptureUnavailable("nsys not found: set the NSYS environment variable"))
    app = make(r)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        before = text(app, "proc-table")
        await start_capture(app, pilot)
        assert await until(pilot, lambda: "nsys not found" in text(app, "status"))
        assert text(app, "proc-table") == before
        assert app._capture_report is None


async def test_failed_shows_first_line_only():
    r = FakeRunner(raises=CaptureError("nsys profile failed (exit code 1):\nsecret second line"))
    app = make(r)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await start_capture(app, pilot)
        assert await until(pilot, lambda: "nsys profile failed" in text(app, "status"))
        assert "secret second line" not in text(app, "status")


async def test_f3_before_and_after_capture_and_f2_returns():
    app = make(FakeRunner())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await pilot.press("f3")
        await settle(pilot)
        assert "no capture yet" in text(app, "timeline-view") and "F4" in text(app, "timeline-view")
        assert app.query_one("#timeline-view").display
        assert not app.query_one("#metrics-view").display
        await pilot.press("f2")
        await settle(pilot)
        assert app.query_one("#metrics-view").display and not app.query_one("#timeline-view").display
        await start_capture(app, pilot)
        assert await until(pilot, lambda: app._capture_report is not None)
        await pilot.press("f3")
        await settle(pilot)
        tl = text(app, "timeline-view")
        assert "1420 |====    |" in tl and "[hold]" in tl and "on-demand capture" in tl
        await pilot.press("f2")
        await settle(pilot)
        assert "PID" in screen_text(app)


async def test_f5_without_bench_dir_says_na():
    app = make(FakeRunner())
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await pilot.press("f5")
        await settle(pilot)
        assert "N/A — no benchmark results loaded (--bench-dir DIR)" in text(app, "compare-text")


async def test_exit_during_running_capture_is_bounded():
    r = FakeRunner(block=True)
    app = make(r)
    t0 = time.monotonic()
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await start_capture(app, pilot)
        assert await until(pilot, r.started.is_set)
        await pilot.press("q")
        await asyncio.sleep(0.5)
    assert time.monotonic() - t0 < 10
    assert app._done.is_set()


async def test_quit_action_waits_for_the_capture_to_end():
    r = FakeRunner(block=True, ignore_cancel=True)
    app = make(r)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await start_capture(app, pilot)
        assert await until(pilot, r.started.is_set)
        t0 = time.monotonic()
        threading.Timer(0.6, r.release.set).start()
        quit_task = asyncio.create_task(app.action_quit())
        assert await until(pilot, lambda: "waiting for capture to finish…" in text(app, "status"))
        assert not quit_task.done()  # still blocked on the capture
        await quit_task
        assert time.monotonic() - t0 >= 0.5  # returned only after the release
        assert r.release.is_set() and app._done.is_set()


async def test_temp_dir_removed_on_exit():
    r = FakeRunner()
    app = make(r)
    async with app.run_test(size=(120, 35)) as pilot:
        await settle(pilot)
        await start_capture(app, pilot)
        assert await until(pilot, lambda: app._capture_report is not None)
        assert app._tmpdir is not None and app._tmpdir.is_dir()
        tmp = app._tmpdir
    assert not tmp.exists()


def test_late_cancel_removes_capture_files(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    import pytest

    from gso.trace.nsys_capture import CaptureResult
    from gso.tui import capture_job

    def fake_capture(duration_s, out_dir, caps_, run=None, name="capture", **kw):
        sq, rep = Path(out_dir) / "capture.sqlite", Path(out_dir) / "capture.nsys-rep"
        shutil.copy(Path(__file__).parent / "fixtures" / "nsys_ctxsw_min.sqlite", sq)
        rep.write_bytes(b"x")
        return CaptureResult(sq, rep, "global", True, duration_s, 0, 1.0, ())

    monkeypatch.setattr(capture_job, "capture", fake_capture)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CaptureCancelled):
        capture_job.run_global_capture(5, tmp_path, caps(), 1, cancel)
    assert list(tmp_path.iterdir()) == []
    ok = capture_job.run_global_capture(5, tmp_path, caps(), 1, threading.Event())
    assert ok.summary["gpu"] == 1 and list(tmp_path.iterdir()) == []

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import gso.collector
import gso.export
from gso import cli
from gso.buffer import RingBuffer
from gso.capability import Capabilities, DcgmStatus, GpuInfo
from gso.collector import Collector
from gso.export import read_jsonl
from gso.record import Quality, Record, Scope
from gso.tui.app import GsoApp
from tests.fakes import FakeCollector, FakeTime


def caps():
    return Capabilities(
        gpus=[GpuInfo(1, "GPU-test-0001", "Test GPU", "535", False)],
        dcgm=DcgmStatus(False, "down", {}),
        nsys_path=None,
        is_root=False,
        mps_active=False,
    )


def test_gpu_is_required():
    with pytest.raises(SystemExit) as e:
        cli.main(["live"])
    assert e.value.code == 2


@pytest.mark.parametrize(
    "bad",
    [
        ["--interval", "0"],
        ["--interval", "-1"],
        ["--dcgm-interval-ms", "50"],
        ["--interval", "nan"],
        ["--interval", "inf"],
    ],
)
def test_bad_intervals_rejected(bad):
    with pytest.raises(SystemExit) as e:
        cli.main(["live", "--gpu", "1", *bad])
    assert e.value.code == 2


def test_build_live_wires_collector_without_starting(monkeypatch):
    seen = {}

    def fake_detect(**kw):
        seen.update(kw)
        return caps()

    monkeypatch.setattr(cli, "detect", fake_detect)
    args = cli.build_parser().parse_args(
        [
            "live",
            "--gpu",
            "1",
            "--interval",
            "0.5",
            "--dcgm-interval-ms",
            "200",
            "--dcgmi-cmd",
            "docker exec gso-dcgm dcgmi",
            "--dcgm-gpu-index",
            "0",
        ]
    )
    app = cli.build_live(args)
    assert isinstance(app, GsoApp)
    assert seen == {
        "dcgmi_cmd": ["docker", "exec", "gso-dcgm", "dcgmi"],
        "dcgm_gpu_index": 0,
    }
    c = app._collector
    assert isinstance(c, Collector)
    assert c._interval_s == 0.5
    assert c._thread is None  # not started
    assert app._interval_s == 0.5
    assert isinstance(app._buffer, RingBuffer)


def test_buffer_capacity_covers_five_minutes():
    # 6 device + (3 per process, say 8 processes) + 5 dcgm per tick; DCGM at the chosen rate
    assert cli.live_buffer_capacity(1.0, 1000) >= 300 * (6 + 24) + 300 * 5
    assert cli.live_buffer_capacity(1.0, 100) >= 300 * (6 + 24) + 3000 * 5
    assert cli.live_buffer_capacity(0.1, 1000) >= 3000 * (6 + 24) + 300 * 5


# -- export / headless ---------------------------------------------------------


def mk(ts, metric="gpu.util", value=50.0):
    return Record(
        ts, "GPU-test-0001", metric, value, "%", "nvml", Scope.DEVICE, Quality.MEASURED
    )


@pytest.fixture
def fake_collector(monkeypatch):
    FakeCollector.made = []
    FakeCollector.records_on_start = []
    FakeCollector.nvml_error = None
    FakeCollector.dcgm_error = None
    monkeypatch.setattr(cli, "detect", lambda **kw: caps())
    monkeypatch.setattr(gso.collector, "Collector", FakeCollector)
    return FakeCollector


def run_main(argv, ft=None):
    ft = ft or FakeTime()
    return cli.main(argv, sleep=ft.sleep, clock=ft.clock), ft


@pytest.mark.parametrize(
    "argv",
    [
        ["--headless", "--export", "{p}/o.jsonl"],  # no duration
        ["--headless", "--duration", "5"],  # no export
        ["--duration", "5", "--export", "{p}/o.jsonl"],  # duration w/o headless
        ["--headless", "--duration", "0", "--export", "{p}/o.jsonl"],
        ["--headless", "--duration", "-3", "--export", "{p}/o.jsonl"],
        ["--headless", "--duration", "nan", "--export", "{p}/o.jsonl"],
        ["--headless", "--duration", "5", "--export", "{p}/o.txt"],  # bad extension
        [
            "--headless",
            "--duration",
            "5",
            "--export",
            "{p}/nodir/o.jsonl",
        ],  # parent missing
        ["--export", "{p}/o.parquet"],  # interactive, bad ext
        ["--interval", "0.05"],
        ["--headless", "--duration", "5", "--export", "{p}"],  # target is a directory
    ],
)
def test_live_argument_validation(argv, tmp_path, fake_collector):
    argv = [a.replace("{p}", str(tmp_path)) for a in argv]
    with pytest.raises(SystemExit) as e:
        run_main(["live", "--gpu", "1", *argv])
    assert e.value.code == 2
    assert fake_collector.made == []  # nothing was built, let alone started
    assert list(tmp_path.iterdir()) == []


def test_interval_lower_bound_accepts_point_one():
    args = cli.build_parser().parse_args(["live", "--gpu", "1", "--interval", "0.1"])
    assert args.interval == 0.1


def test_headless_writes_expected_records(tmp_path, fake_collector, capsys):
    out = tmp_path / "o.jsonl"
    fake_collector.records_on_start = [mk(1, "a"), mk(1, "b")]

    def on_sleep(n):
        for c in FakeCollector.made:
            c.buffer.push(mk(100 + n, f"s{n}"))

    code, ft = run_main(
        ["live", "--gpu", "1", "--headless", "--duration", "3", "--export", str(out)],
        FakeTime(on_sleep),
    )
    assert code == 0
    (c,) = fake_collector.made
    assert c.starts == 1 and c.stops == 1
    assert sum(ft.sleeps) == pytest.approx(3.0)
    recs = read_jsonl(out)
    assert [r.metric for r in recs] == ["a", "b"] + [
        f"s{n}" for n in range(1, len(ft.sleeps) + 1)
    ]
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1
    assert (
        f"wrote {len(recs)} records" in err[0]
        and str(out) in err[0]
        and "3.0s" in err[0]
    )
    assert "dropped" not in err[0]


def test_headless_csv(tmp_path, fake_collector):
    out = tmp_path / "o.CSV"
    fake_collector.records_on_start = [mk(1)]
    code, _ = run_main(
        ["live", "--gpu", "1", "--headless", "--duration", "1", "--export", str(out)]
    )
    assert code == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("ts_ns,gpu_uuid,pid") and len(lines) == 2


def test_headless_reports_dropped(tmp_path, fake_collector, capsys, monkeypatch):
    monkeypatch.setattr(cli, "live_buffer_capacity", lambda i, d: 3)
    fake_collector.records_on_start = []
    out = tmp_path / "o.jsonl"

    def on_sleep(n):
        if n == 1:
            for i in range(10):
                FakeCollector.made[0].buffer.push(mk(i, f"m{i}"))

    code, _ = run_main(
        ["live", "--gpu", "1", "--headless", "--duration", "2", "--export", str(out)],
        FakeTime(on_sleep),
    )
    assert code == 0
    assert len(read_jsonl(out)) == 3
    err = capsys.readouterr().err
    assert "7 records dropped" in err


def test_headless_zero_records_exits_1_with_reason(tmp_path, fake_collector, capsys):
    fake_collector.nvml_error = "NVMLError: no driver"
    fake_collector.dcgm_error = "host engine down"
    out = tmp_path / "o.jsonl"
    code, _ = run_main(
        ["live", "--gpu", "1", "--headless", "--duration", "1", "--export", str(out)]
    )
    assert code == 1
    err = capsys.readouterr().err.splitlines()
    assert any("wrote 0 records" in ln for ln in err)
    assert any("NVMLError: no driver" in ln for ln in err)
    assert any("host engine down" in ln for ln in err)
    assert not out.exists() and list(tmp_path.iterdir()) == []  # nothing to commit


@pytest.mark.parametrize("gpu", ["7", "-1"])
def test_headless_unknown_gpu_fails_fast(tmp_path, fake_collector, capsys, gpu):
    out = tmp_path / "o.jsonl"
    code, ft = run_main(
        ["live", "--gpu", gpu, "--headless", "--duration", "60", "--export", str(out)]
    )
    assert code == 1  # nothing written
    assert ft.sleeps == [] and ft.now == 0.0  # did not idle for the duration
    assert all(c.starts == 0 for c in fake_collector.made)
    err = capsys.readouterr().err
    assert f"GPU {gpu} not found" in err
    assert list(tmp_path.iterdir()) == []


def test_headless_status_errors_printed_each_line_even_with_records(
    tmp_path, fake_collector, capsys
):
    fake_collector.records_on_start = [mk(1)]
    fake_collector.dcgm_error = "host engine down"
    out = tmp_path / "o.jsonl"
    code, _ = run_main(
        ["live", "--gpu", "1", "--headless", "--duration", "1", "--export", str(out)]
    )
    assert code == 0
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 2 and "host engine down" in err[1]


def test_headless_keyboard_interrupt(tmp_path, fake_collector, capsys):
    out = tmp_path / "o.jsonl"
    fake_collector.records_on_start = [mk(1, "a")]

    def on_sleep(n):
        FakeCollector.made[0].buffer.push(mk(2, "b"))
        if n == 2:
            raise KeyboardInterrupt

    code, _ = run_main(
        ["live", "--gpu", "1", "--headless", "--duration", "60", "--export", str(out)],
        FakeTime(on_sleep),
    )
    assert code == 130
    (c,) = fake_collector.made
    assert c.stops == 1
    assert [r.metric for r in read_jsonl(out)] == ["a", "b", "b"]  # flushed and closed
    assert "wrote 3 records" in capsys.readouterr().err


def test_headless_stops_collector_when_loop_fails(tmp_path, fake_collector):
    def on_sleep(n):
        raise RuntimeError("sleep failed")

    with pytest.raises(RuntimeError):
        run_main(
            [
                "live",
                "--gpu",
                "1",
                "--headless",
                "--duration",
                "5",
                "--export",
                str(tmp_path / "o.jsonl"),
            ],
            FakeTime(on_sleep),
        )
    assert fake_collector.made[0].stops == 1


def test_headless_does_not_import_textual():
    script = textwrap.dedent(
        """
        import sys, tempfile, os
        from types import SimpleNamespace
        import gso.collector
        from gso import cli
        from tests.fakes import FakeCollector, FakeTime
        from gso.capability import Capabilities, DcgmStatus, GpuInfo
        from gso.record import Quality, Record, Scope
        caps = Capabilities(gpus=[GpuInfo(1, "GPU-test-0001", "T", "535", False)],
                            dcgm=DcgmStatus(False, "down", {}), nsys_path=None,
                            is_root=False, mps_active=False)
        cli.detect = lambda **kw: caps
        mk = lambda ts: Record(ts, "GPU-test-0001", "gpu.util", 1.0, "%", "nvml",
                               Scope.DEVICE, Quality.MEASURED)
        gso.collector.Collector = FakeCollector
        FakeCollector.records_on_start = [mk(1)]
        d = tempfile.mkdtemp()
        ft = FakeTime()
        code = cli.main(["live", "--gpu", "1", "--headless", "--duration", "1",
                         "--export", os.path.join(d, "o.jsonl")], sleep=ft.sleep, clock=ft.clock)
        assert code == 0, code
        assert "textual" not in sys.modules, "textual was imported"
        print("ok")
        """
    )
    r = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout


class FakeApp:
    def __init__(self, buffer, collector, run):
        self.buffer = buffer
        self.collector = collector
        self.run = run


def patch_app(monkeypatch, run, records=()):
    buf = RingBuffer(100)
    for r in records:
        buf.push(r)
    col = SimpleNamespace(stops=0)
    col.stop = lambda: setattr(col, "stops", col.stops + 1)
    app = FakeApp(buf, col, run)
    monkeypatch.setattr(cli, "build_live", lambda args: app)
    return app, col


def test_interactive_export_writes_buffer_after_run(tmp_path, monkeypatch, capsys):
    out = tmp_path / "o.jsonl"
    _app, col = patch_app(monkeypatch, lambda: None, [mk(1, "a"), mk(1, "b")])
    code = cli.main(["live", "--gpu", "1", "--export", str(out)])
    assert code == 0 and col.stops == 1
    assert [r.metric for r in read_jsonl(out)] == ["a", "b"]
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "wrote 2 records" in err[0] and str(out) in err[0]


def test_interactive_without_export_writes_nothing(tmp_path, monkeypatch, capsys):
    _app, col = patch_app(monkeypatch, lambda: None, [mk(1)])
    assert cli.main(["live", "--gpu", "1"]) == 0
    assert col.stops == 1 and capsys.readouterr().err == ""


def test_interactive_run_failure_still_stops_collector(monkeypatch):
    def boom():
        raise RuntimeError("app failed")

    _app, col = patch_app(monkeypatch, boom)
    with pytest.raises(RuntimeError):
        cli.main(["live", "--gpu", "1"])
    assert col.stops == 1


# -- fix round 1: .partial streaming, exactly-once drain, I/O errors -----------------

HEADLESS = ["live", "--gpu", "1", "--headless", "--duration", "3", "--export"]


def test_headless_normal_run_replaces_old_target_no_partial(tmp_path, fake_collector):
    out = tmp_path / "o.jsonl"
    out.write_text("OLD\n", encoding="utf-8")
    fake_collector.records_on_start = [mk(1, "a")]
    code, _ = run_main([*HEADLESS, str(out)])
    assert code == 0
    assert [r.metric for r in read_jsonl(out)] == ["a"]
    assert [p.name for p in tmp_path.iterdir()] == ["o.jsonl"]


def test_headless_failure_before_any_record_leaves_target_alone(
    tmp_path, fake_collector
):
    out = tmp_path / "o.jsonl"
    out.write_text("OLD\n", encoding="utf-8")

    def on_sleep(n):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_main([*HEADLESS, str(out)], FakeTime(on_sleep))
    assert out.read_text(encoding="utf-8") == "OLD\n"
    assert [p.name for p in tmp_path.iterdir()] == ["o.jsonl"]


def test_headless_interrupt_commits_collected_records(tmp_path, fake_collector):
    out = tmp_path / "o.jsonl"
    out.write_text("OLD\n", encoding="utf-8")
    fake_collector.records_on_start = [mk(1, "a"), mk(1, "b")]

    def on_sleep(n):
        raise KeyboardInterrupt

    code, _ = run_main([*HEADLESS, str(out)], FakeTime(on_sleep))
    assert code == 130
    assert [r.metric for r in read_jsonl(out)] == ["a", "b"]
    assert [p.name for p in tmp_path.iterdir()] == ["o.jsonl"]


def test_headless_zero_records_keeps_existing_target(tmp_path, fake_collector):
    out = tmp_path / "o.jsonl"
    out.write_text("OLD\n", encoding="utf-8")
    code, _ = run_main([*HEADLESS, str(out)])
    assert code == 1
    assert out.read_text(encoding="utf-8") == "OLD\n"
    assert [p.name for p in tmp_path.iterdir()] == ["o.jsonl"]


def test_headless_stream_is_opened_before_detect(tmp_path, monkeypatch, fake_collector):
    order = []
    real_open = gso.export.open_stream
    monkeypatch.setattr(
        gso.export, "open_stream", lambda p: (order.append("open"), real_open(p))[1]
    )
    monkeypatch.setattr(cli, "detect", lambda **kw: (order.append("detect"), caps())[1])
    run_main([*HEADLESS, str(tmp_path / "o.jsonl")])
    assert order == ["open", "detect"]


def test_headless_ctrl_c_during_detect_exits_130(
    tmp_path, monkeypatch, fake_collector, capsys
):
    def interrupted(**kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "detect", interrupted)
    out = tmp_path / "o.jsonl"
    out.write_text("OLD\n", encoding="utf-8")
    code, _ = run_main([*HEADLESS, str(out)])
    assert code == 130
    assert out.read_text(encoding="utf-8") == "OLD\n"
    assert [p.name for p in tmp_path.iterdir()] == ["o.jsonl"]
    assert fake_collector.made == []


@pytest.mark.skipif(
    hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() == 0,
    reason="root ignores directory permissions",
)
def test_unwritable_parent_directory_rejected(tmp_path, fake_collector):
    d = tmp_path / "ro"
    d.mkdir()
    d.chmod(0o555)
    try:
        with pytest.raises(SystemExit) as e:
            run_main([*HEADLESS, str(d / "o.jsonl")])
        assert e.value.code == 2
        assert fake_collector.made == []
    finally:
        d.chmod(0o755)


class FlakyStream(gso.export.JsonlStream):
    """Raises ``exc`` when asked to write record number ``fail_at`` (once unless ``always``)."""

    fail_at = 0
    exc: BaseException = KeyboardInterrupt()
    always = False

    def write(self, records):
        out = 0
        for r in records:
            if self.count + 1 == type(self).fail_at:
                if not type(self).always:
                    type(self).fail_at = 0
                raise type(self).exc
            out += super().write([r])
        return out


def test_headless_interrupt_inside_drain_writes_each_record_once(
    tmp_path, monkeypatch, fake_collector
):
    FlakyStream.fail_at, FlakyStream.exc, FlakyStream.always = (
        3,
        KeyboardInterrupt(),
        False,
    )
    monkeypatch.setattr(gso.export, "open_stream", FlakyStream)
    fake_collector.records_on_start = [mk(i, f"r{i}") for i in range(8)]
    out = tmp_path / "o.jsonl"
    code, _ = run_main([*HEADLESS, str(out)])
    assert code == 130
    assert [r.metric for r in read_jsonl(out)] == [f"r{i}" for i in range(8)]


def test_headless_io_error_exits_3_and_keeps_partial(
    tmp_path, monkeypatch, fake_collector, capsys
):
    FlakyStream.fail_at = 3
    FlakyStream.exc = OSError(28, "No space left on device")
    FlakyStream.always = True
    monkeypatch.setattr(gso.export, "open_stream", FlakyStream)
    fake_collector.records_on_start = [mk(i, f"r{i}") for i in range(8)]
    out = tmp_path / "o.jsonl"
    out.write_text("OLD\n", encoding="utf-8")
    code, _ = run_main([*HEADLESS, str(out)])
    assert code == 3
    assert fake_collector.made[0].stops == 1
    assert out.read_text(encoding="utf-8") == "OLD\n"
    partial = tmp_path / "o.jsonl.partial"
    assert [r.metric for r in read_jsonl(partial)] == ["r0", "r1"]
    err = capsys.readouterr().err.splitlines()
    assert "wrote 2 records" in err[0] and str(partial) in err[0]
    assert any(
        "No space left on device" in ln and "o.jsonl.partial" in ln for ln in err
    )


def test_live_help_documents_exit_codes(capsys):
    with pytest.raises(SystemExit):
        cli.main(["live", "--help"])
    out = capsys.readouterr().out
    for frag in ("0 ", "1 ", "2 ", "3 ", "130"):
        assert frag in out.split("exit codes", 1)[1]


def test_headless_duration_excludes_detect_time(tmp_path, monkeypatch, fake_collector):
    ft = FakeTime()

    def slow_detect(**kw):
        ft.now += 5.0  # detect() is slow (dcgmi probe)
        return caps()

    monkeypatch.setattr(cli, "detect", slow_detect)
    fake_collector.records_on_start = [mk(1)]
    code, _ = run_main([*HEADLESS, str(tmp_path / "o.jsonl")], ft)
    assert code == 0 and sum(ft.sleeps) == pytest.approx(3.0)


def test_staleness_window_follows_slower_source(fake_collector):
    def age(interval, dcgm_ms):
        args = cli.build_parser().parse_args(
            [
                "live",
                "--gpu",
                "1",
                "--interval",
                str(interval),
                "--dcgm-interval-ms",
                str(dcgm_ms),
            ]
        )
        return cli.build_live(args)._max_age_ns

    assert age(0.1, 1000) == 3_000_000_000  # fast NVML, default DCGM: no N/A flicker
    assert age(1.0, 5000) == 15_000_000_000  # slow DCGM interval is honoured
    assert age(2.0, 200) == 6_000_000_000

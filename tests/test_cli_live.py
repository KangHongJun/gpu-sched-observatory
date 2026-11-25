import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import gso.collector
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
    [["--interval", "0"], ["--interval", "-1"], ["--dcgm-interval-ms", "50"],
     ["--interval", "nan"], ["--interval", "inf"]],
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
        ["live", "--gpu", "1", "--interval", "0.5", "--dcgm-interval-ms", "200",
         "--dcgmi-cmd", "docker exec gso-dcgm dcgmi", "--dcgm-gpu-index", "0"]
    )
    app = cli.build_live(args)
    assert isinstance(app, GsoApp)
    assert seen == {"dcgmi_cmd": ["docker", "exec", "gso-dcgm", "dcgmi"], "dcgm_gpu_index": 0}
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
    return Record(ts, "GPU-test-0001", metric, value, "%", "nvml", Scope.DEVICE, Quality.MEASURED)


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
        ["--headless", "--export", "{p}/o.jsonl"],                      # no duration
        ["--headless", "--duration", "5"],                              # no export
        ["--duration", "5", "--export", "{p}/o.jsonl"],                 # duration w/o headless
        ["--headless", "--duration", "0", "--export", "{p}/o.jsonl"],
        ["--headless", "--duration", "-3", "--export", "{p}/o.jsonl"],
        ["--headless", "--duration", "nan", "--export", "{p}/o.jsonl"],
        ["--headless", "--duration", "5", "--export", "{p}/o.txt"],     # bad extension
        ["--headless", "--duration", "5", "--export", "{p}/nodir/o.jsonl"],  # parent missing
        ["--export", "{p}/o.parquet"],                                  # interactive, bad ext
        ["--interval", "0.05"],
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
    assert [r.metric for r in recs] == ["a", "b"] + [f"s{n}" for n in range(1, len(ft.sleeps) + 1)]
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1
    assert f"wrote {len(recs)} records" in err[0] and str(out) in err[0] and "3.0s" in err[0]
    assert "dropped" not in err[0]


def test_headless_csv(tmp_path, fake_collector):
    out = tmp_path / "o.CSV"
    fake_collector.records_on_start = [mk(1)]
    code, _ = run_main(["live", "--gpu", "1", "--headless", "--duration", "1", "--export", str(out)])
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
    code, _ = run_main(["live", "--gpu", "1", "--headless", "--duration", "1", "--export", str(out)])
    assert code == 1
    err = capsys.readouterr().err.splitlines()
    assert any("wrote 0 records" in ln for ln in err)
    assert any("NVMLError: no driver" in ln for ln in err)
    assert any("host engine down" in ln for ln in err)
    assert out.exists()  # a valid empty file


def test_headless_status_errors_printed_each_line_even_with_records(tmp_path, fake_collector, capsys):
    fake_collector.records_on_start = [mk(1)]
    fake_collector.dcgm_error = "host engine down"
    out = tmp_path / "o.jsonl"
    code, _ = run_main(["live", "--gpu", "1", "--headless", "--duration", "1", "--export", str(out)])
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
            ["live", "--gpu", "1", "--headless", "--duration", "5", "--export",
             str(tmp_path / "o.jsonl")],
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
    r = subprocess.run([sys.executable, "-c", script], check=False,
                       capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout


class FakeApp:
    def __init__(self, buffer, collector, run):
        self._buffer = buffer
        self._collector = collector
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

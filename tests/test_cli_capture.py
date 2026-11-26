import json
import shutil
from pathlib import Path

import pytest

from gso import cli, cli_capture
from gso.capability import Capabilities, DcgmStatus, GpuInfo
from gso.trace.nsys_capture import CaptureError, CaptureResult, CaptureUnavailable

FIX = Path(__file__).parent / "fixtures"


def caps(root=False):
    return Capabilities(
        gpus=[GpuInfo(1, "GPU-test-0001", "Test GPU", "535", False)],
        dcgm=DcgmStatus(False, "down", {}),
        nsys_path="/x/nsys",
        is_root=root,
        mps_active=False,
    )


class Fake:
    """Stands in for nsys_capture.capture; copies a fixture SQLite as the capture output."""

    def __init__(self, fixture="nsys_ctxsw_min.sqlite", mode="global", raises=None):
        self.fixture, self.mode, self.raises, self.calls = fixture, mode, raises, []

    def __call__(self, duration_s, out_dir, caps, **kw):
        self.calls.append((duration_s, Path(out_dir), kw))
        if self.raises:
            raise self.raises
        out = Path(out_dir)
        sq = out / f"{kw['name']}.sqlite"
        rep = out / f"{kw['name']}.nsys-rep"
        shutil.copy(FIX / self.fixture, sq)
        rep.write_bytes(b"rep")
        return CaptureResult(sq, rep, self.mode, True, duration_s, 1_700_000_000_000_000_000,
                             duration_s + 1.0, ("warn one", "warn two"))


@pytest.fixture
def fake(monkeypatch):
    f = Fake()
    monkeypatch.setattr(cli_capture, "detect", lambda **kw: caps())
    monkeypatch.setattr(cli_capture, "capture", f)
    return f


def run(argv):
    return cli.main(["capture", *argv])


@pytest.mark.parametrize(
    "argv",
    [
        ["--out", "{out}"],  # missing --gpu
        ["--gpu", "1", "--duration", "4", "--out", "{out}"],
        ["--gpu", "1", "--duration", "11", "--out", "{out}"],
        ["--gpu", "1", "--duration", "6.5", "--out", "{out}"],
        ["--gpu", "1", "--out", "{out}", "--end-target"],
        ["--gpu", "1", "--out", "{file}"],
        ["--gpu", "1"],  # missing --out
        ["--gpu", "1", "--out", "{out}", "--"],  # empty target
    ],
)
def test_validation_exits_2_without_capture(argv, tmp_path, fake):
    f = tmp_path / "afile"
    f.write_text("x")
    argv = [a.format(out=tmp_path / "o", file=f) for a in argv]
    with pytest.raises(SystemExit) as e:
        run(argv)
    assert e.value.code == 2
    assert fake.calls == []


def test_unavailable_exit_4(tmp_path, fake, capsys):
    fake.raises = CaptureUnavailable("nsys not found: set NSYS")
    assert run(["--gpu", "1", "--out", str(tmp_path)]) == 4
    assert "nsys not found" in capsys.readouterr().err


def test_failed_exit_5(tmp_path, fake, capsys):
    fake.raises = CaptureError("nsys profile failed (exit code 1)")
    assert run(["--gpu", "1", "--out", str(tmp_path)]) == 5
    assert "nsys profile failed" in capsys.readouterr().err
    assert not (tmp_path / "summary.json").exists()


def test_interrupt_exit_130(tmp_path, fake):
    fake.raises = KeyboardInterrupt()
    assert run(["--gpu", "1", "--out", str(tmp_path)]) == 130


def test_success_writes_files_and_removes_rep(tmp_path, fake, capsys):
    out = tmp_path / "new" / "dir"
    assert run(["--gpu", "1", "--duration", "5", "--out", str(out)]) == 0
    assert (out / "summary.json").is_file()
    assert (out / "timeline.txt").is_file()
    assert (out / "capture.sqlite").is_file()
    assert not (out / "capture.nsys-rep").exists()
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["gpu"] == 1 and s["mode"] == "global" and s["masked"] is True
    assert "\n  " in (out / "summary.json").read_text(encoding="utf-8")  # pretty
    dur, d, kw = fake.calls[0]
    assert dur == 5 and d == out and kw["name"] == "capture"
    assert kw.get("target") is None and kw.get("env") is None
    err = capsys.readouterr().err
    assert "warn one" in err and "summary.json" in err and "timeline.txt" in err
    assert "restore_save" in err and "hold" in err
    assert "N/A" in err  # no kernels in a global capture
    assert "real-time" not in err and "실시간" not in err


def test_keep_rep(tmp_path, fake):
    assert run(["--gpu", "1", "--out", str(tmp_path), "--keep-rep"]) == 0
    assert (tmp_path / "capture.nsys-rep").exists()


def test_default_duration_is_10(tmp_path, fake):
    run(["--gpu", "1", "--out", str(tmp_path)])
    assert fake.calls[0][0] == 10


def test_target_passthrough_and_env_defaults(tmp_path, fake, monkeypatch, capsys):
    fake.fixture, fake.mode = "nsys_launched_min.sqlite", "launched"
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    argv = ["--gpu", "1", "--out", str(tmp_path), "--",
            "uv", "run", "python", "-m", "w", "--gpu", "1", "--duration", "8"]
    assert run(argv) == 0
    kw = fake.calls[0][2]
    assert kw["target"] == ["uv", "run", "python", "-m", "w", "--gpu", "1", "--duration", "8"]
    assert kw["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "3"  # the user's value wins
    assert kw["end_target"] is False
    assert "may still be running" in capsys.readouterr().err
    assert json.loads((tmp_path / "summary.json").read_text())["kernel"]["count"] == 2000


def test_env_default_cuda_visible_devices(tmp_path, fake, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    run(["--gpu", "2", "--out", str(tmp_path), "--", "true"])
    env = fake.calls[0][2]["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "2" and env["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


def test_end_target_passed(tmp_path, fake):
    run(["--gpu", "1", "--out", str(tmp_path), "--end-target", "--", "true"])
    assert fake.calls[0][2]["end_target"] is True


def test_na_for_nulls_and_lower_bound(tmp_path, fake, capsys, monkeypatch):
    import gso.trace.report as r

    monkeypatch.setattr(r, "switch_rate_is_lower_bound_by_gpu", lambda *a, **k: {1: True})
    run(["--gpu", "1", "--out", str(tmp_path)])
    err = capsys.readouterr().err
    assert "≥" in err
    run(["--gpu", "0", "--out", str(tmp_path)])  # masked-only GPU: rate unknown
    err = capsys.readouterr().err
    assert "context switch rate: N/A" in err


def test_atomic_write_failure_leaves_no_partial(tmp_path, fake, monkeypatch, capsys):
    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(cli_capture.os, "replace", boom)
    assert run(["--gpu", "1", "--out", str(tmp_path)]) == 3
    assert not (tmp_path / "summary.json").exists()
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob(".*tmp*"))
    assert "disk full" in capsys.readouterr().err


def test_help_mentions_env_and_no_realtime(capsys):
    with pytest.raises(SystemExit):
        cli.main(["capture", "--help"])
    out = capsys.readouterr().out
    assert "CUDA_VISIBLE_DEVICES" in out and "on-demand" in out
    assert "real-time" not in out.lower() and "실시간" not in out

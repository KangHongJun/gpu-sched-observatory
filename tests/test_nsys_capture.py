"""nsys on-demand 캡처 래퍼 테스트. 실제 nsys는 쓰지 않는다."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from gso.capability import Capabilities, DcgmStatus, GpuInfo
from gso.trace import nsys_capture
from gso.trace.nsys_capture import (
    CaptureError,
    CaptureUnavailable,
    build_command,
    build_export_command,
    capture,
)
from tests.fakes import fake_run

NSYS = "/opt/fake/nsys"


def caps(nsys=NSYS, root=False):
    return Capabilities(
        gpus=[GpuInfo(0, "GPU-test-0000", "NVIDIA RTX A5000", "535.00", False)],
        dcgm=DcgmStatus(False, "off", {}),
        nsys_path=nsys,
        is_root=root,
        mps_active=False,
    )


def touch_outputs(out_dir, name="capture", rep=True, sqlite=True):
    """profile 호출이면 .nsys-rep, export 호출이면 .sqlite 를 만든다."""

    def on_call(argv, index):
        if argv[1] == "profile" and rep:
            (out_dir / f"{name}.nsys-rep").write_bytes(b"rep")
        if argv[1] == "export" and sqlite:
            (out_dir / f"{name}.sqlite").write_bytes(b"sqlite")

    return on_call


# --- build_command / build_export_command --------------------------------
def test_build_command_global_exact():
    assert build_command(NSYS, 10, Path("/o/c")) == [
        NSYS, "profile", "--gpuctxsw=true", "--sample=none", "--cpuctxsw=none",
        "--duration=10", "--force-overwrite=true", "-o", "/o/c", "sleep", "10",
    ]


def test_build_command_launched_exact():
    argv = build_command(NSYS, 5, Path("/o/c"), target=["python", "-m", "w"])
    assert argv == [
        NSYS, "profile", "--gpuctxsw=true", "--sample=none", "--cpuctxsw=none",
        "--trace=cuda", "--duration=5", "--force-overwrite=true", "-o", "/o/c",
        "python", "-m", "w",
    ]


def test_build_command_gpuctxsw_false_drops_only_that_option():
    on = build_command(NSYS, 7, Path("/o/c"))
    off = build_command(NSYS, 7, Path("/o/c"), gpuctxsw=False)
    assert off == [a for a in on if a != "--gpuctxsw=true"]


@pytest.mark.parametrize("bad", [4, 11, True, 5.0, "5", None])
def test_build_command_rejects_bad_duration(bad):
    with pytest.raises(ValueError):
        build_command(NSYS, bad, Path("/o/c"))


def test_build_command_rejects_empty_target():
    with pytest.raises(ValueError):
        build_command(NSYS, 5, Path("/o/c"), target=[])


def test_build_export_command_exact():
    assert build_export_command(NSYS, Path("/o/c.nsys-rep"), Path("/o/c.sqlite")) == [
        NSYS, "export", "--type=sqlite", "--force-overwrite=true",
        "-o", "/o/c.sqlite", "/o/c.nsys-rep",
    ]


# --- capture: preconditions ---------------------------------------------
def test_unavailable_when_no_nsys(tmp_path):
    run = fake_run()
    with pytest.raises(CaptureUnavailable, match="NSYS"):
        capture(5, tmp_path, caps(nsys=None), run=run)
    assert run.argvs == []
    assert list(tmp_path.iterdir()) == []


def test_duration_out_of_range_raises(tmp_path):
    run = fake_run()
    with pytest.raises(ValueError):
        capture(60, tmp_path, caps(), run=run)
    assert run.argvs == []


@pytest.mark.parametrize("name", ["../x", "a/b", ".hidden", "", "a b", "a\\b"])
def test_bad_name_raises_before_running(tmp_path, name):
    run = fake_run()
    with pytest.raises(ValueError):
        capture(5, tmp_path, caps(), run=run, name=name)
    assert run.argvs == []


def test_missing_out_dir_raises(tmp_path):
    run = fake_run()
    with pytest.raises(ValueError):
        capture(5, tmp_path / "nope", caps(), run=run)
    assert run.argvs == []


# --- capture: success ----------------------------------------------------
def test_profile_then_export_in_order(tmp_path):
    run = fake_run(on_call=touch_outputs(tmp_path, "g"))
    res = capture(5, tmp_path, caps(), run=run, name="g", clock_ns=lambda: 123)
    assert run.argvs == [
        build_command(NSYS, 5, tmp_path / "g"),
        build_export_command(NSYS, tmp_path / "g.nsys-rep", tmp_path / "g.sqlite"),
    ]
    assert res.sqlite_path == tmp_path / "g.sqlite"
    assert res.rep_path == tmp_path / "g.nsys-rep"
    assert res.mode == "global"
    assert res.duration_s == 5
    assert res.started_ns == 123
    assert res.wall_s >= 0


def test_non_root_is_masked_with_warning(tmp_path):
    run = fake_run(on_call=touch_outputs(tmp_path))
    res = capture(5, tmp_path, caps(root=False), run=run)
    assert res.masked is True
    assert any("마스킹" in w and "PID" in w for w in res.warnings)
    assert isinstance(res.warnings, tuple)


def test_root_is_not_masked_and_has_no_mask_warning(tmp_path):
    run = fake_run(on_call=touch_outputs(tmp_path))
    res = capture(5, tmp_path, caps(root=True), run=run)
    assert res.masked is False
    assert not any("마스킹" in w for w in res.warnings)


def test_global_mode_warns_no_kernel_intervals(tmp_path):
    run = fake_run(on_call=touch_outputs(tmp_path))
    res = capture(5, tmp_path, caps(root=True), run=run)
    assert any("커널 구간" in w for w in res.warnings)


def test_launched_mode_warns_ctxsw_only_around_target(tmp_path):
    run = fake_run(on_call=touch_outputs(tmp_path, "l"))
    res = capture(5, tmp_path, caps(root=True), run=run, name="l", target=["w", "--x"])
    assert res.mode == "launched"
    assert run.argvs[0][-2:] == ["w", "--x"]
    assert any("컨텍스트 전환" in w and "실행한 프로세스" in w for w in res.warnings)
    assert not any("커널 구간" in w for w in res.warnings)


def test_env_and_timeouts_passed(tmp_path):
    run = fake_run(on_call=touch_outputs(tmp_path))
    env = {"CUDA_VISIBLE_DEVICES": "1"}
    capture(7, tmp_path, caps(), run=run, env=env)
    assert run.kwargs[0]["env"] == env
    assert run.kwargs[0]["timeout"] == 7 + 60
    assert run.kwargs[1]["timeout"] == 120
    assert run.kwargs[0]["capture_output"] is True
    assert run.kwargs[0]["text"] is True


def test_default_env_is_inherit(tmp_path):
    run = fake_run(on_call=touch_outputs(tmp_path))
    capture(5, tmp_path, caps(), run=run)
    assert run.kwargs[0]["env"] is None


# --- capture: failures ---------------------------------------------------
def test_profile_failure_cleans_up_and_skips_export(tmp_path):
    other = tmp_path / "keep.txt"
    other.write_text("x")
    run = fake_run(
        results=[(3, "", "a\n\nb\nc\nd\ne\nf\ng\n")],
        on_call=lambda argv, i: (tmp_path / "c.nsys-rep").write_bytes(b"partial"),
    )
    with pytest.raises(CaptureError) as ei:
        capture(5, tmp_path, caps(), run=run, name="c")
    msg = str(ei.value)
    assert "3" in msg and "g" in msg and "f" in msg
    assert "\na\n" not in f"\n{msg}\n"  # only the tail
    assert len(run.argvs) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]
    assert other.read_text() == "x"


def test_profile_ok_but_no_rep_is_error(tmp_path):
    run = fake_run()
    with pytest.raises(CaptureError, match="nsys-rep"):
        capture(5, tmp_path, caps(), run=run)
    assert len(run.argvs) == 1


def test_export_failure_removes_rep_and_partial_sqlite(tmp_path):
    def on_call(argv, i):
        if argv[1] == "profile":
            (tmp_path / "capture.nsys-rep").write_bytes(b"r")
        else:
            (tmp_path / "capture.sqlite").write_bytes(b"half")

    run = fake_run(results=[(0, "", ""), (1, "boom", "")], on_call=on_call)
    with pytest.raises(CaptureError, match="boom"):
        capture(5, tmp_path, caps(), run=run)
    assert list(tmp_path.iterdir()) == []


def test_export_ok_but_no_sqlite_is_error(tmp_path):
    run = fake_run(on_call=touch_outputs(tmp_path, sqlite=False))
    with pytest.raises(CaptureError, match="sqlite"):
        capture(5, tmp_path, caps(), run=run)
    assert list(tmp_path.iterdir()) == []


def test_timeout_is_capture_error(tmp_path):
    run = fake_run(raises=subprocess.TimeoutExpired(cmd="nsys", timeout=65))
    with pytest.raises(CaptureError, match="시간 초과|timed out|timeout"):
        capture(5, tmp_path, caps(), run=run)
    assert list(tmp_path.iterdir()) == []


def test_oserror_is_capture_error(tmp_path):
    run = fake_run(raises=OSError("no exec"))
    with pytest.raises(CaptureError, match="no exec"):
        capture(5, tmp_path, caps(), run=run)


# --- wording -------------------------------------------------------------
def test_module_never_claims_real_time_or_streaming():
    src = Path(nsys_capture.__file__).read_text(encoding="utf-8").lower()
    for word in ("실시간", "스트리밍", "real-time", "realtime", "streaming"):
        assert word not in src


# --- process-group helper ------------------------------------------------
def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_group_runner_kills_only_its_own_tree_on_timeout(tmp_path):
    pidfile = tmp_path / "pids"
    sentinel = subprocess.Popen(["sleep", "30"])  # 같은 부모, 다른 그룹: 건드리면 안 된다
    try:
        script = f"echo $$ > {pidfile}; sleep 30 & echo $! >> {pidfile}; wait"
        with pytest.raises(subprocess.TimeoutExpired):
            nsys_capture._run_in_group(
                ["sh", "-c", script], capture_output=True, text=True, timeout=0.5, env=None
            )
        pids = [int(x) for x in pidfile.read_text().split()]
        assert len(pids) == 2
        deadline = time.monotonic() + 3
        while any(alive(p) for p in pids) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not any(alive(p) for p in pids)
        assert sentinel.poll() is None
    finally:
        sentinel.kill()
        sentinel.wait()


def test_group_runner_returns_completed_process():
    r = nsys_capture._run_in_group(
        ["sh", "-c", "echo out; echo err >&2; exit 4"],
        capture_output=True, text=True, timeout=5, env=None,
    )
    assert (r.returncode, r.stdout.strip(), r.stderr.strip()) == (4, "out", "err")


# --- launched target outliving the duration ------------------------------
def test_launched_target_stopped_by_nsys_is_success_with_warning(tmp_path):
    """nsys가 --duration 끝에 대상을 SIGTERM으로 끝내면 종료 코드 143이지만 보고서는 정상이다."""
    run = fake_run(
        results=[(143, "Generated: l.nsys-rep", ""), (0, "", "")],
        on_call=touch_outputs(tmp_path, "l"),
    )
    res = capture(5, tmp_path, caps(root=True), run=run, name="l", target=["w"])
    assert res.mode == "launched"
    assert len(run.argvs) == 2
    assert any("대상" in w and "종료" in w for w in res.warnings)


def test_exit_143_in_global_mode_is_still_an_error(tmp_path):
    run = fake_run(results=[(143, "", "x")], on_call=touch_outputs(tmp_path))
    with pytest.raises(CaptureError, match="143"):
        capture(5, tmp_path, caps(), run=run)
    assert list(tmp_path.iterdir()) == []


def test_launched_exit_143_without_rep_is_an_error(tmp_path):
    run = fake_run(results=[(143, "", "x")])
    with pytest.raises(CaptureError, match="143"):
        capture(5, tmp_path, caps(), run=run, target=["w"])


def test_launched_other_nonzero_exit_is_an_error(tmp_path):
    run = fake_run(results=[(1, "", "bad")], on_call=touch_outputs(tmp_path))
    with pytest.raises(CaptureError, match="bad"):
        capture(5, tmp_path, caps(), run=run, target=["w"])
    assert list(tmp_path.iterdir()) == []

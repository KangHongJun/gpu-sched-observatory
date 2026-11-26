import signal
from types import SimpleNamespace

import pytest

from gso import cli, cli_bench
from gso.bench.runner import RunOutcome


class Spy:
    def __init__(self, status="complete", message=""):
        self.calls = []
        self.status, self.message = status, message
        self.sig_during = None

    def __call__(self, *a, **kw):
        self.calls.append((a, kw))
        self.sig_during = signal.getsignal(signal.SIGTERM)
        return RunOutcome(kw["out"] if "out" in kw else a[3] / "results.jsonl", self.status, self.message)


@pytest.fixture
def spy(monkeypatch):
    s = Spy()
    caps = SimpleNamespace(
        gpus=[SimpleNamespace(index=2, name="Test GPU", driver="1.2")],
        dcgm=SimpleNamespace(available=True, fields={1002: True, 1003: False}),
        nsys_path="/opt/nsys/bin/nsys", nvml_error=None,
    )
    seen = {}
    monkeypatch.setattr(cli_bench, "run_matrix", s)
    monkeypatch.setattr(cli_bench, "detect", lambda **kw: seen.update(detect=kw) or caps)
    monkeypatch.setattr(cli_bench, "NvmlDcgmTelemetry", lambda *a, **kw: SimpleNamespace(a=a, kw=kw))
    monkeypatch.setattr(cli_bench, "nsys_version", lambda path: "nsys 9.9")
    s.caps, s.seen = caps, seen
    return s


def _bench(tmp_path, *extra, wa="small_kernels", wb="small_kernels"):
    return ["bench", "--gpu", "2", "--a", wa, "--b", wb, "--out", str(tmp_path / "o"), *extra]


def test_defaults_are_passed_through(spy, tmp_path):
    assert cli.main(_bench(tmp_path)) == 0
    (a, kw), = spy.calls
    assert a[:3] == (2, "small_kernels", "small_kernels")
    assert kw["reps"] == 3 and kw["duration_s"] == 30 and kw["warmup_s"] == 10
    assert kw["profile"] is False and kw["conditions"] is None
    assert kw["a_args"] == [] and kw["b_args"] == []
    assert kw["env_info"]["gpu_name"] == "Test GPU" and kw["env_info"]["driver_version"] == "1.2"


def test_options_are_parsed(spy, tmp_path):
    rc = cli.main(_bench(
        tmp_path, "--reps", "4", "--duration", "25", "--warmup", "5", "--profile",
        "--conditions", "mps_off,mps_50_50", "mps_70_30", "--a-args", "--size 512",
        "--b-args", "--size 64 --x 'a b'", "--dcgmi-cmd", "docker exec gso-dcgm dcgmi",
        "--dcgm-gpu-index", "0", "--setup-budget", "20", wa="membound", wb="decode"))
    assert rc == 0
    (a, kw), = spy.calls
    assert a[1:3] == ("membound", "decode")
    assert kw["reps"] == 4 and kw["duration_s"] == 25 and kw["warmup_s"] == 5
    assert kw["profile"] is True and kw["setup_budget_s"] == 20
    assert kw["conditions"] == ["mps_off", "mps_50_50", "mps_70_30"]
    assert kw["a_args"] == ["--size", "512"] and kw["b_args"] == ["--size", "64", "--x", "a b"]
    assert spy.seen["detect"]["dcgmi_cmd"] == ["docker", "exec", "gso-dcgm", "dcgmi"]
    assert spy.seen["detect"]["dcgm_gpu_index"] == 0
    assert kw["hooks"].telemetry.kw["dcgmi_cmd"] == ["docker", "exec", "gso-dcgm", "dcgmi"]
    assert kw["hooks"].telemetry.a[1] == {1002: True, 1003: False}
    assert kw["hooks"].profiler is not None
    assert kw["env_info"]["nsys_version"] == "nsys 9.9"


@pytest.mark.parametrize(
    "extra",
    [["--reps", "2"], ["--duration", "0"], ["--duration", "-3"], ["--warmup", "0"],
     ["--profile", "--duration", "19"], ["--conditions", "mps_99"],
     ["--a-args", "--gpu 1"], ["--b-args", "--out x"], ["--a-args", "--start-at 5"],
     ["--a-args", "'unterminated"], ["--setup-budget", "0"]],
)
def test_invalid_arguments_exit_2_and_launch_nothing(spy, tmp_path, extra):
    with pytest.raises(SystemExit) as e:
        cli.main(_bench(tmp_path, *extra))
    assert e.value.code == 2 and spy.calls == []


def test_invalid_workload_exits_2(spy, tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(_bench(tmp_path, wa="nope"))
    assert e.value.code == 2 and spy.calls == []


def test_gpu_is_required_without_default(spy, tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(["bench", "--a", "small_kernels", "--b", "small_kernels", "--out", str(tmp_path)])
    assert e.value.code == 2


def test_existing_results_file_exits_2(spy, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "results.jsonl").write_text("")
    with pytest.raises(SystemExit) as e:
        cli.main(_bench(tmp_path))
    assert e.value.code == 2 and spy.calls == []


def test_profile_without_nsys_exits_2_before_running(spy, tmp_path, capsys):
    spy.caps.nsys_path = None
    assert cli.main(_bench(tmp_path, "--profile")) == 2
    assert "nsys" in capsys.readouterr().err and spy.calls == []


@pytest.mark.parametrize(
    ("status", "code"),
    [("complete", 0), ("incomplete", 1), ("aborted", 3), ("interrupted", 130)],
)
def test_exit_codes(spy, tmp_path, status, code):
    spy.status = status
    assert cli.main(_bench(tmp_path)) == code


def test_value_error_from_runner_is_a_usage_error(monkeypatch, spy, tmp_path, capsys):
    def boom(*a, **kw):
        raise ValueError("bad thing")

    monkeypatch.setattr(cli_bench, "run_matrix", boom)
    assert cli.main(_bench(tmp_path)) == 2
    assert "bad thing" in capsys.readouterr().err


def test_sigterm_becomes_keyboard_interrupt_during_the_run_and_is_restored(spy, tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    cli.main(_bench(tmp_path))
    handler = spy.sig_during
    assert callable(handler) and handler is not before
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGTERM, None)
    assert signal.getsignal(signal.SIGTERM) is before


def test_summary_line_names_outcome_and_results(spy, tmp_path, capsys):
    spy.status, spy.message = "incomplete", "1 repetition(s) failed"
    cli.main(_bench(tmp_path))
    err = capsys.readouterr().err
    assert "incomplete" in err and "1 repetition(s) failed" in err and "results.jsonl" in err

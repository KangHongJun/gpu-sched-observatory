import os
import re
import subprocess
from pathlib import Path

import pytest

from gso.bench.mps import MpsSession, UnsafeGpuError, _send_control, ensure_safe
from tests.fakes import FakeNvml

SERVER_TXT = "4242\n"
_QUERY_STDIN = re.compile(
    r"(get_server_list|get_client_list [1-9][0-9]*|get_active_thread_percentage [1-9][0-9]*)\n"
)
_BINARY = "nvidia-cuda-mps-control"

_CURRENT: list[str] | None = None


@pytest.fixture(autouse=True)
def violations():
    """Fresh violation list per test; fails at teardown if a fake call had a forbidden shape,
    even when the code under test swallowed the exception."""
    global _CURRENT
    _CURRENT = []
    yield _CURRENT
    found, _CURRENT = list(_CURRENT), None
    assert not found, "forbidden calls recorded by fake run: " + "; ".join(found)


class Daemon:
    """Fake subprocess.run plus fake process table for one session."""

    def __init__(self, base: Path, start_ok=True, up_after_start=True, quit_removes=True):
        self.pipe = str(base / "pipe")
        self.log = str(base / "log")
        self.up = False
        self.start_ok = start_ok
        self.up_after_start = up_after_start
        self.quit_removes = quit_removes
        self.starts: list[dict] = []
        self.quits: list[dict] = []
        self.queries: list[str] = []
        self.killed: list[int] = []

    def own(self):
        return [(555, "control")] if self.up else []

    def run(self, argv, *args, **kwargs):
        assert _CURRENT is not None
        env = kwargs.get("env") or {}
        argv = list(argv)
        stdin = kwargs.get("input")
        if argv == [_BINARY, "-d"]:
            if env.get("CUDA_MPS_PIPE_DIRECTORY") != self.pipe or "CUDA_VISIBLE_DEVICES" not in env:
                _CURRENT.append(f"start env {env.get('CUDA_MPS_PIPE_DIRECTORY')!r}")
            self.starts.append(env)
            self.up = self.start_ok and self.up_after_start
            return subprocess.CompletedProcess(argv, 0 if self.start_ok else 1, "", "boom")
        if argv == [_BINARY] and stdin == "quit\n":
            if env.get("CUDA_MPS_PIPE_DIRECTORY") != self.pipe:
                _CURRENT.append(f"quit pipe {env.get('CUDA_MPS_PIPE_DIRECTORY')!r}")
            self.quits.append(env)
            if self.quit_removes:
                self.up = False
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv == [_BINARY] and isinstance(stdin, str) and _QUERY_STDIN.fullmatch(stdin):
            if env.get("CUDA_MPS_PIPE_DIRECTORY") != self.pipe:
                _CURRENT.append(f"query pipe {env.get('CUDA_MPS_PIPE_DIRECTORY')!r}")
            self.queries.append(stdin)
            if not self.up:
                return subprocess.CompletedProcess(argv, 1, "", "")
            out = SERVER_TXT if stdin == "get_server_list\n" else "100.0\n"
            return subprocess.CompletedProcess(argv, 0, out, "")
        _CURRENT.append(f"unexpected call {argv!r} input={stdin!r}")
        return subprocess.CompletedProcess(argv, 1, "", "")


def make(tmp_path, nvml=None, listing=None, **kw):
    base = tmp_path / "gso-mps"
    d = Daemon(base, **{k: kw.pop(k) for k in list(kw) if k in
                        ("start_ok", "up_after_start", "quit_removes")})
    s = MpsSession(
        2,
        base_dir=base,
        run=d.run,
        nvml=nvml or FakeNvml(gpus=[{}, {}, {}]),
        mps_listing=listing or (list),
        find_own=lambda: d.own(),
        kill=lambda pid, sig: d.killed.append(pid),
        **kw,
    )
    return s, d, base


# ---- ensure_safe -------------------------------------------------------------------------


def test_refuses_gpu_with_foreign_process():
    with pytest.raises(UnsafeGpuError):
        ensure_safe(0, nvml=FakeNvml(procs={0: [999]}), mps_listing=list)


def test_allows_own_processes():
    ensure_safe(1, nvml=FakeNvml(gpus=[{}, {}], procs={1: [111]}), own_pids={111},
                mps_listing=list)


def test_idle_gpu_ok_and_nvml_balanced():
    nv = FakeNvml(gpus=[{}, {}])
    ensure_safe(1, nvml=nv, mps_listing=list)
    assert nv.calls.count("nvmlInit") == nv.calls.count("nvmlShutdown") == 1


def test_foreign_process_only_in_mps_list_refused():
    with pytest.raises(UnsafeGpuError):
        ensure_safe(0, nvml=FakeNvml(mps_procs={0: [999]}), mps_listing=list)


def test_foreign_mps_daemon_refused():
    with pytest.raises(UnsafeGpuError, match="MPS"):
        ensure_safe(0, nvml=FakeNvml(), mps_listing=lambda: ["777 nvidia-cuda-mps-control -d"])


def test_own_mps_daemon_allowed_via_own_pids():
    ensure_safe(0, nvml=FakeNvml(), own_pids={777},
                mps_listing=lambda: ["777 nvidia-cuda-mps-control -d"])


@pytest.mark.parametrize("idx", [-1, 1, 5, True, "0"])
def test_invalid_index_refused(idx):
    with pytest.raises(UnsafeGpuError):
        ensure_safe(idx, nvml=FakeNvml(), mps_listing=list)


def test_nvml_init_failure_refused():
    nv = FakeNvml(errors={"nvmlInit": RuntimeError("no driver")})
    with pytest.raises(UnsafeGpuError, match="no driver"):
        ensure_safe(0, nvml=nv, mps_listing=list)


def test_nvml_query_failure_refused_and_shutdown_called():
    nv = FakeNvml(errors={"nvmlDeviceGetComputeRunningProcesses": RuntimeError("x")})
    with pytest.raises(UnsafeGpuError):
        ensure_safe(0, nvml=nv, mps_listing=list)
    assert "nvmlShutdown" in nv.calls


def test_listing_failure_refused():
    def boom():
        raise OSError("pgrep missing")

    with pytest.raises(UnsafeGpuError):
        ensure_safe(0, nvml=FakeNvml(), mps_listing=boom)


def test_message_has_count_not_pids_or_names():
    nv = FakeNvml(procs={0: [123456, 654321]}, mps_procs={0: [777777]})
    with pytest.raises(UnsafeGpuError) as e:
        ensure_safe(0, nvml=nv, mps_listing=list)
    msg = str(e.value)
    assert "GPU 0" in msg and "3" in msg
    for pid in ("123456", "654321", "777777"):
        assert pid not in msg
    with pytest.raises(UnsafeGpuError) as e:
        ensure_safe(0, nvml=FakeNvml(), mps_listing=lambda: ["888888 secretname -x"])
    assert "888888" not in str(e.value) and "secretname" not in str(e.value)


# ---- private quit sender -----------------------------------------------------------------


@pytest.mark.parametrize("cmd", ["get_server_list", "quit\n", "quit 1", "", "QUIT", "start_server", None])
def test_send_control_refuses_anything_but_quit(cmd):
    calls = []
    with pytest.raises(ValueError):
        _send_control(cmd, "/p", "/l", lambda *a, **k: calls.append(a))
    assert calls == []


def test_send_control_sends_quit_with_our_pipe():
    seen = {}

    def run(argv, **kw):
        seen.update(argv=argv, **kw)
        return subprocess.CompletedProcess(argv, 0, "", "")

    _send_control("quit", "/p", "/l", run)
    assert seen["argv"] == [_BINARY] and seen["input"] == "quit\n"
    assert seen["env"]["CUDA_MPS_PIPE_DIRECTORY"] == "/p" and seen["timeout"]


def test_provider_has_no_quit():
    src = (Path(__file__).parent.parent / "src/gso/providers/mps.py").read_text()
    assert "quit" not in src


# ---- MpsSession --------------------------------------------------------------------------


def test_client_env_sets_thread_percentage(tmp_path):
    s, _, base = make(tmp_path)
    env = s.client_env(thread_pct=70)
    assert env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "70"
    assert env["CUDA_MPS_PIPE_DIRECTORY"].startswith(str(base))
    assert env["CUDA_MPS_LOG_DIRECTORY"].startswith(str(base))
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in s.client_env(None)


@pytest.mark.parametrize("pct", [0, 101, -5])
def test_client_env_rejects_bad_percentage(tmp_path, pct):
    s, _, _ = make(tmp_path)
    with pytest.raises(ValueError):
        s.client_env(pct)


def test_client_gpu_arg_is_zero(tmp_path):
    s, _, _ = make(tmp_path)
    assert s.client_gpu_arg == 0


def test_exit_only_quits_daemon_it_started(tmp_path):
    s, d, base = make(tmp_path)
    environ_before = dict(os.environ)
    with s:
        assert d.up and s.own_daemon_pids == frozenset({555, 4242})
    assert dict(os.environ) == environ_before
    assert len(d.starts) == 1 and len(d.quits) == 1
    assert d.quits[0]["CUDA_MPS_PIPE_DIRECTORY"].startswith(str(base))
    assert d.starts[0]["CUDA_VISIBLE_DEVICES"] == "2"
    assert d.starts[0]["CUDA_MPS_LOG_DIRECTORY"] == str(base / "log")
    assert not d.up and d.killed == []


def test_base_dir_created_mode_700(tmp_path):
    s, _, base = make(tmp_path)
    with s:
        pass
    assert (base.stat().st_mode & 0o777) == 0o700
    assert (base / "pipe").is_dir() and (base / "log").is_dir()


def test_symlink_base_refused_before_anything_runs(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = tmp_path / "gso-mps"
    link.symlink_to(target)
    s, d, _ = make(tmp_path)
    with pytest.raises(UnsafeGpuError, match="symlink"), s:
        pass
    assert d.starts == [] and d.quits == [] and d.queries == []


def test_base_dir_not_owned_refused(tmp_path):
    base = tmp_path / "gso-mps"
    base.mkdir(mode=0o700)
    d = Daemon(base)
    s = MpsSession(2, base_dir=base, run=d.run, nvml=FakeNvml(gpus=[{}, {}, {}]),
                   mps_listing=list, find_own=lambda: d.own(), owner_check=lambda p: False)
    with pytest.raises(UnsafeGpuError, match="owned"), s:
        pass
    assert d.starts == [] and d.quits == []


def test_base_dir_is_file_refused(tmp_path):
    (tmp_path / "gso-mps").write_text("x")
    s, d, _ = make(tmp_path)
    with pytest.raises(UnsafeGpuError), s:
        pass
    assert d.starts == []


def test_unsafe_gpu_refuses_before_starting(tmp_path):
    s, d, base = make(tmp_path, nvml=FakeNvml(gpus=[{}, {}, {}], procs={2: [999]}))
    with pytest.raises(UnsafeGpuError), s:
        pass
    assert d.starts == [] and d.quits == [] and not base.exists()


def test_foreign_mps_daemon_refuses_session(tmp_path):
    s, d, _ = make(tmp_path, listing=lambda: ["9 nvidia-cuda-mps-control"])
    with pytest.raises(UnsafeGpuError), s:
        pass
    assert d.starts == []


def test_startup_timeout_quits_our_daemon(tmp_path):
    s, d, _ = make(tmp_path, up_after_start=False, start_timeout_s=0.05)
    # daemon process exists but never answers on the pipe
    d.own = lambda: [(555, "control")] if d.starts and not d.quits else []
    with pytest.raises(UnsafeGpuError, match="did not become active"), s:
        pass
    assert len(d.starts) == 1 and len(d.quits) == 1


def test_start_command_failure_quits_and_raises(tmp_path):
    s, d, _ = make(tmp_path, start_ok=False, start_timeout_s=0.05)
    with pytest.raises(UnsafeGpuError, match="boom"), s:
        pass
    assert len(d.quits) == 1


def test_exception_in_body_quits_and_propagates(tmp_path):
    s, d, _ = make(tmp_path)

    class Marker(Exception):
        pass

    marker = Marker("body")
    with pytest.raises(Marker) as e, s:
        raise marker
    assert e.value is marker
    assert len(d.quits) == 1 and not d.up


def test_keyboard_interrupt_in_body_quits(tmp_path):
    s, d, _ = make(tmp_path)
    with pytest.raises(KeyboardInterrupt), s:
        raise KeyboardInterrupt
    assert len(d.quits) == 1 and not d.up


def test_leftover_reported_on_clean_exit_and_not_killed_without_identity(tmp_path):
    s, d, _ = make(tmp_path, quit_removes=False, stop_timeout_s=0.05)
    d.own = lambda: [(555, "control")]  # never goes away
    s._identity = lambda pid: None
    with pytest.raises(RuntimeError, match="still running"), s:
        pass
    assert d.killed == []  # identity unreadable for the fake pid: never killed


def test_leftover_does_not_hide_body_exception(tmp_path):
    s, d, _ = make(tmp_path, quit_removes=False, stop_timeout_s=0.05)
    d.own = lambda: [(555, "control")]
    s._identity = lambda pid: None
    with pytest.warns(RuntimeWarning, match="still running"), pytest.raises(ZeroDivisionError), s:
        raise ZeroDivisionError


def test_leftover_control_pid_terminated_only_if_same_process(tmp_path):
    s, d, _ = make(tmp_path, quit_removes=False, stop_timeout_s=0.05)
    d.own = lambda: [(555, "control")]
    s._identity = lambda pid: (pid, 7)  # same start time at start and at exit
    with pytest.raises(RuntimeError, match="still running"), s:
        pass
    assert d.killed == [555]


def test_reused_pid_not_killed(tmp_path):
    s, d, _ = make(tmp_path, quit_removes=False, stop_timeout_s=0.05)
    d.own = lambda: [(555, "control")]
    import itertools

    ids = itertools.chain([(555, 7)], itertools.repeat((555, 8)))
    s._identity = lambda pid: next(ids)  # start time changed: a different process now
    with pytest.raises(RuntimeError), s:
        pass
    assert d.killed == []


def test_find_own_processes_matches_user_comm_and_pipe_dir(tmp_path):
    from gso.bench.mps import find_own_processes

    def proc(pid, cmd, env):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode() + b"\0")
        (d / "environ").write_bytes(b"\0".join(e.encode() for e in env) + b"\0")

    me = os.getuid()
    proc(10, "nvidia-cuda-mps-control -d", ["CUDA_MPS_PIPE_DIRECTORY=/ours"])
    proc(11, "nvidia-cuda-mps-server", ["A=1", "CUDA_MPS_PIPE_DIRECTORY=/ours"])
    proc(12, "nvidia-cuda-mps-control -d", ["CUDA_MPS_PIPE_DIRECTORY=/other"])
    proc(13, "python app.py", ["CUDA_MPS_PIPE_DIRECTORY=/ours"])
    proc(14, "nvidia-cuda-mps-control -d", ["CUDA_MPS_PIPE_DIRECTORY=/ours-not"])
    (tmp_path / "self").mkdir()
    assert sorted(find_own_processes("/ours", uid=me, proc_root=str(tmp_path))) == [
        (10, "control"),
        (11, "server"),
    ]
    assert find_own_processes("/ours", uid=me + 1, proc_root=str(tmp_path)) == []


def test_no_module_side_effects_on_construct(tmp_path):
    _, d, base = make(tmp_path)
    assert not base.exists() and d.starts == [] and d.queries == []

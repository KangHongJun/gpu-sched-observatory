import subprocess
from pathlib import Path

import pytest

from gso.providers import mps
from gso.providers.mps import MpsStatus, parse_pid_list, query

FIX = Path(__file__).parent / "fixtures"
SERVER_TXT = (FIX / "mps_server_list.txt").read_text()
CLIENT_TXT = (FIX / "mps_client_list.txt").read_text()
EMPTY_TXT = (FIX / "mps_server_list_empty.txt").read_text()
SERVER = int(SERVER_TXT.strip())
CLIENTS = [int(x) for x in CLIENT_TXT.split()]
ALLOWED = ("get_server_list", "get_client_list", "get_active_thread_percentage")

ALL_RUNS: list = []


def scripted(replies):
    """replies: callable(command) -> (returncode, stdout) | Exception, or a list consumed in order."""

    def run(argv, *args, **kwargs):
        run.argvs.append(list(argv))
        run.kwargs.append(kwargs)
        cmd = kwargs["input"].strip()
        run.commands.append(cmd)
        r = replies(cmd) if callable(replies) else replies[len(run.commands) - 1]
        if isinstance(r, BaseException) or (isinstance(r, type) and issubclass(r, BaseException)):
            raise r() if isinstance(r, type) else r
        rc, out = r
        return subprocess.CompletedProcess(argv, rc, out, "")

    run.argvs, run.kwargs, run.commands = [], [], []
    ALL_RUNS.append(run)
    return run


def happy(cmd):
    if cmd == "get_server_list":
        return 0, SERVER_TXT
    if cmd.startswith("get_client_list"):
        return 0, CLIENT_TXT
    return 0, "100.0\n"


def test_parse_fixtures():
    assert parse_pid_list(SERVER_TXT) == [SERVER]
    assert parse_pid_list(CLIENT_TXT) == CLIENTS
    assert parse_pid_list(EMPTY_TXT) == []


def test_parse_ignores_noise_and_dedups():
    assert parse_pid_list("\n  12  \n\n12 34\npid 12\n7\n12\n") == [12, 7]


def test_happy_path():
    run = scripted(happy)
    st = query(run=run)
    assert st == MpsStatus(True, [SERVER], CLIENTS, 100.0)
    assert run.commands == [
        "get_server_list",
        f"get_client_list {SERVER}",
        f"get_active_thread_percentage {SERVER}",
    ]
    assert all(a == ["nvidia-cuda-mps-control"] for a in run.argvs)


def test_no_daemon():
    run = scripted(lambda c: (1, EMPTY_TXT))
    assert query(run=run) == MpsStatus(False, [], [], None)
    assert run.commands == ["get_server_list"]


@pytest.mark.parametrize("exc", [FileNotFoundError("x"), subprocess.TimeoutExpired("c", 5), OSError("x")])
def test_exceptions_inactive(exc):
    run = scripted(lambda c: exc)
    assert query(run=run) == MpsStatus(False, [], [], None)
    assert run.commands == ["get_server_list"]


def test_daemon_without_server():
    run = scripted(lambda c: (0, ""))
    assert query(run=run) == MpsStatus(True, [], [], None)
    assert run.commands == ["get_server_list"]


def test_two_servers():
    def r(cmd):
        if cmd == "get_server_list":
            return 0, "11\n22\n"
        if cmd == "get_client_list 11":
            return 0, "1\n2\n"
        if cmd == "get_client_list 22":
            return 0, "2\n3\n"
        return 0, "50.0\n"

    run = scripted(r)
    assert query(run=run) == MpsStatus(True, [11, 22], [1, 2, 3], 50.0)
    pct = [c for c in run.commands if c.startswith("get_active")]
    assert pct == ["get_active_thread_percentage 11"]


def test_client_list_failure_tolerated():
    def r(cmd):
        if cmd == "get_server_list":
            return 0, "11\n22\n"
        if cmd == "get_client_list 11":
            return 1, "error\n"
        if cmd == "get_client_list 22":
            return 0, "5\n"
        return 0, "100\n"

    run = scripted(r)
    assert query(run=run) == MpsStatus(True, [11, 22], [5], 100.0)


@pytest.mark.parametrize("out", ["N/A\n", "nan\n", "inf\n", ""])
def test_bad_percentage(out):
    def r(cmd):
        return happy(cmd) if not cmd.startswith("get_active") else (0, out)

    run = scripted(r)
    st = query(run=run)
    assert st.active and st.thread_pct is None
    assert st.client_pids == CLIENTS


def test_pipe_dir_env():
    env = {"A": "1"}
    run = scripted(happy)
    query(pipe_dir="/tmp/x", run=run, environ=env)
    for kw in run.kwargs:
        assert kw["env"]["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/x"
        assert kw["env"]["A"] == "1"
        assert kw["timeout"] == 5
    assert env == {"A": "1"}


def test_no_pipe_dir_passthrough():
    run = scripted(happy)
    query(run=run, environ={"A": "1"})
    for kw in run.kwargs:
        assert "CUDA_MPS_PIPE_DIRECTORY" not in kw["env"]


def test_send_rejects_disallowed():
    run = scripted(happy)
    for bad in ("quit", "start_server -uid 0", "set_default_active_thread_percentage 50"):
        with pytest.raises(ValueError):
            mps._send(bad, run, {})
    assert run.commands == []


def test_only_allowed_commands_sent_across_scenarios():
    assert ALL_RUNS
    for run in ALL_RUNS:
        for argv in run.argvs:
            assert argv == ["nvidia-cuda-mps-control"]
        for c in run.commands:
            assert c.split()[0] in ALLOWED


def test_static_no_forbidden_words():
    src = Path(mps.__file__).read_text()
    for w in ("quit", "start_server", "terminate", "shutdown_server", "set_"):
        assert w not in src, w

import re
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

_STDIN_OK = re.compile(
    r"(get_server_list|get_client_list [1-9][0-9]*|get_active_thread_percentage [1-9][0-9]*)\n"
)


_CURRENT: list[str] | None = None


@pytest.fixture(autouse=True)
def violations():
    """Fresh violation list per test; fails the test at teardown if any fake call broke the
    read-only contract (even if the module under test swallowed the failure)."""
    global _CURRENT
    _CURRENT = []
    yield _CURRENT
    found, _CURRENT = list(_CURRENT), None
    assert not found, "read-only violations recorded by fake run: " + "; ".join(found)


def scripted(replies):
    """Fake subprocess.run. Checks the read-only guarantee on EVERY call, in every test:
    bare argv, stdin is exactly one allowed command line, a timeout is given.

    replies: callable(command_without_newline) -> (returncode, stdout) | exception."""

    def run(argv, *args, **kwargs):
        # Record (never raise) so a failure swallowed by the code under test still fails the test.
        if _CURRENT is None:
            raise RuntimeError("fake run used outside the autouse violations fixture")
        stdin = kwargs.get("input")
        if list(argv) != ["nvidia-cuda-mps-control"]:
            _CURRENT.append(f"argv {list(argv)!r}")
        if args:
            _CURRENT.append(f"extra positional args {args!r}")
        if not isinstance(stdin, str) or not _STDIN_OK.fullmatch(stdin):
            _CURRENT.append(f"stdin {stdin!r}")
        if not kwargs.get("timeout"):
            _CURRENT.append("missing timeout")
        stdin = stdin if isinstance(stdin, str) else ""
        run.argvs.append(list(argv))
        run.kwargs.append(kwargs)
        run.commands.append(stdin)
        r = replies(stdin[:-1])
        if isinstance(r, BaseException) or (
            isinstance(r, type) and issubclass(r, BaseException)
        ):
            raise r() if isinstance(r, type) else r
        rc, out = r
        return subprocess.CompletedProcess(argv, rc, out, "")

    run.argvs, run.kwargs, run.commands = [], [], []
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


def test_parse_ignores_zero():
    assert parse_pid_list("0\n00\n5\n") == [5]


def test_happy_path():
    run = scripted(happy)
    st = query(run=run)
    assert st == MpsStatus(True, [SERVER], CLIENTS, 100.0)
    assert run.commands == [
        "get_server_list\n",
        f"get_client_list {SERVER}\n",
        f"get_active_thread_percentage {SERVER}\n",
    ]


def test_no_daemon():
    run = scripted(lambda c: (1, EMPTY_TXT))
    assert query(run=run) == MpsStatus(False, [], [], None)
    assert run.commands == ["get_server_list\n"]


@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError("x"),
        subprocess.TimeoutExpired("c", 5),
        OSError("x"),
        PermissionError("x"),
        ValueError("embedded null byte"),
        TypeError("bad env"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"),
    ],
)
def test_exceptions_inactive(exc):
    run = scripted(lambda c: exc)
    assert query(run=run) == MpsStatus(False, [], [], None)
    assert run.commands == ["get_server_list\n"]


def test_exception_on_later_commands_tolerated():
    def r(cmd):
        if cmd == "get_server_list":
            return 0, SERVER_TXT
        return ValueError("boom")

    st = query(run=scripted(r))
    assert st == MpsStatus(True, [SERVER], [], None)


def test_stdout_none_handled():
    st = query(run=scripted(lambda c: (0, None)))
    assert st == MpsStatus(True, [], [], None)
    st = query(
        run=scripted(lambda c: (0, SERVER_TXT) if c == "get_server_list" else (0, None))
    )
    assert st == MpsStatus(True, [SERVER], [], None)


def test_daemon_without_server():
    run = scripted(lambda c: (0, ""))
    assert query(run=run) == MpsStatus(True, [], [], None)
    assert run.commands == ["get_server_list\n"]


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
    assert pct == ["get_active_thread_percentage 11\n"]


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

    st = query(run=scripted(r))
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
        assert kw["encoding"] == "utf-8" and kw["errors"] == "replace"
    assert env == {"A": "1"}


def test_no_pipe_dir_passthrough():
    run = scripted(happy)
    query(run=run, environ={"A": "1"})
    for kw in run.kwargs:
        assert "CUDA_MPS_PIPE_DIRECTORY" not in kw["env"]


@pytest.mark.parametrize(
    "verb,pid",
    [
        ("quit", None),
        ("start_server", None),
        ("set_default_active_thread_percentage", 50),
        ("get_server_list quit", None),
        ("get_client_list\nquit", 5),
        (" get_server_list", None),
        ("get_server_list", 5),
        ("get_client_list", None),
        ("get_client_list", "5"),
        ("get_client_list", True),
        ("get_client_list", 0),
        ("get_client_list", -1),
        ("get_active_thread_percentage", None),
    ],
)
def test_send_rejects_invalid(verb, pid):
    calls = []
    with pytest.raises(ValueError):
        mps._send(verb, pid, lambda *a, **k: calls.append(a), {})
    assert calls == []


def test_send_builds_exact_stdin():
    run = scripted(lambda c: (0, ""))
    mps._send("get_client_list", 5, run, {})
    mps._send("get_server_list", None, run, {})
    assert run.commands == ["get_client_list 5\n", "get_server_list\n"]


def test_static_no_forbidden_words():
    src = Path(mps.__file__).read_text()
    for w in ("quit", "start_server", "terminate", "shutdown_server", "set_"):
        assert w not in src, w


def _evil_send_now(verb, pid, run, env):
    return run(
        ["nvidia-cuda-mps-control", "--evil"],
        input="quit\n",
        capture_output=True,
        timeout=5,
    )


def test_guard_catches_violation_on_inactive_result(monkeypatch, violations):
    run = scripted(lambda c: OSError("refused"))
    monkeypatch.setattr(mps, "_send", _evil_send_now)
    assert query(run=run) == MpsStatus(False, [], [], None)  # failure was swallowed...
    assert violations  # ...but the fake recorded it
    violations.clear()


def test_guard_catches_violation_on_later_command(monkeypatch, violations):
    run = scripted(happy)
    real = mps._send

    def send(verb, pid, run_, env):
        if verb == "get_client_list":
            return _evil_send_now(verb, pid, run_, env)
        return real(verb, pid, run_, env)

    monkeypatch.setattr(mps, "_send", send)
    st = query(run=run)
    assert st.active and st.client_pids == []
    assert violations
    violations.clear()


def test_send_rejects_str_subclass():
    class Sneaky(str):
        def __format__(self, spec):
            return "quit"

    calls = []
    with pytest.raises(ValueError):
        mps._send(Sneaky("get_server_list"), None, lambda *a, **k: calls.append(a), {})
    assert calls == []

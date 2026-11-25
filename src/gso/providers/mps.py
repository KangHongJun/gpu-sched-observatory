"""Read-only CUDA MPS status query.

Talks to ``nvidia-cuda-mps-control`` over stdin and only ever sends the commands in
``_ALLOWED``. It never passes arguments to the binary (so it cannot launch a daemon) and never
raises: any failure means "MPS not active".
"""

from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass

_BINARY = "nvidia-cuda-mps-control"
_SERVER_LIST = "get_server_list"
_ALLOWED = (_SERVER_LIST, "get_client_list", "get_active_thread_percentage")
_PIPE_ENV = "CUDA_MPS_PIPE_DIRECTORY"
_TIMEOUT_S = 5


@dataclass(frozen=True)
class MpsStatus:
    active: bool
    server_pids: list[int]
    client_pids: list[int]
    thread_pct: float | None


def parse_pid_list(text: str) -> list[int]:
    """One int per all-digit line; everything else is ignored. Order kept, duplicates dropped."""
    out: list[int] = []
    for line in text.splitlines():
        s = line.strip()
        if s.isascii() and s.isdigit():
            n = int(s)
            if n > 0 and n not in out:
                out.append(n)
    return out


def _send(verb: str, pid: int | None, run, env: Mapping[str, str]) -> subprocess.CompletedProcess:
    """The only place a control command leaves this module.

    The stdin text is built here from an exact allowed verb and an optional positive int, so
    extra tokens or newlines cannot be expressed. Anything else raises ValueError."""
    if type(verb) is not str or verb not in _ALLOWED:
        raise ValueError(f"MPS command not allowed: {verb!r}")
    verb = _ALLOWED[_ALLOWED.index(verb)]  # use our own literal, never the caller's object
    if verb == _SERVER_LIST:
        if pid is not None:
            raise ValueError(f"{verb} takes no pid")
        text = f"{verb}\n"
    else:
        if type(pid) is not int or pid <= 0:
            raise ValueError(f"{verb} needs a positive int pid, got {pid!r}")
        text = f"{verb} {pid}\n"
    return run(
        [_BINARY],
        input=text,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=_TIMEOUT_S,
        env=dict(env),
    )


def _parse_pct(text: str) -> float | None:
    for line in text.splitlines():
        try:
            v = float(line.strip())
        except ValueError:
            continue
        return v if math.isfinite(v) else None
    return None


def _ask(verb: str, pid: int | None, run, env: Mapping[str, str]) -> str | None:
    """stdout of a successful command, or None on non-zero exit or any failure of ``run``."""
    try:
        r = _send(verb, pid, run, env)
        return (r.stdout or "") if r.returncode == 0 else None
    except Exception:  # noqa: BLE001 - query must never raise
        return None


def query(pipe_dir: str | None = None, run=subprocess.run, environ=None) -> MpsStatus:
    env = dict(os.environ if environ is None else environ)
    if pipe_dir is not None:
        env[_PIPE_ENV] = pipe_dir

    out = _ask(_SERVER_LIST, None, run, env)
    if out is None:
        return MpsStatus(False, [], [], None)
    servers = parse_pid_list(out)  # may be empty: daemon up, no server yet

    clients: list[int] = []
    for pid in servers:
        text = _ask("get_client_list", pid, run, env)
        for c in parse_pid_list(text or ""):
            if c not in clients:
                clients.append(c)

    pct = None
    if servers:
        text = _ask("get_active_thread_percentage", servers[0], run, env)
        pct = _parse_pct(text or "")
    return MpsStatus(True, servers, clients, pct)

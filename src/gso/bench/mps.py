"""User-scoped CUDA MPS daemon lifecycle for one GPU, with safety checks.

Runs on shared machines: the module only ever starts, talks to and stops a daemon it created
itself (private pipe/log directory), and refuses to run when the GPU or the machine is not in a
state it can use safely (fail closed). No GPU library is imported here except through the
injected ``nvml`` (``pynvml`` is imported lazily). Nothing happens at import time.

How our daemon is recognised
----------------------------
``nvidia-cuda-mps-control -d`` forks and the parent exits, so there is no child handle and no PID
file. Instead we scan ``/proc`` for processes of THIS user whose ``comm``/command line is an MPS
control daemon or server AND whose environment has ``CUDA_MPS_PIPE_DIRECTORY`` equal to our pipe
directory (a process's environ is readable only by its owner). The control daemon found that way
is recorded with its start time (``gso.providers.nvml.proc_identity``) so that at exit a PID that
was reused by another process is never signalled. MPS server PIDs are additionally taken from
``get_server_list`` over our pipe directory.

Marking the daemon as own for ``ensure_safe``
---------------------------------------------
``session.own_daemon_pids`` is the set of PIDs (control daemon, servers) of this session. Pass it,
together with the PIDs of your own client processes, as ``ensure_safe(gpu, own_pids=...)``; the
same set is used both for the GPU compute-process check and for the "any MPS daemon on the
machine" check.

Session lock and device order
-----------------------------
One session per user at a time: ``base_dir/session.lock`` is flock'ed exclusively (opened with
``O_NOFOLLOW``) from before the machine-wide MPS check until after the daemon has been stopped.
The daemon is started with ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` forced (never inherited) so that
``CUDA_VISIBLE_DEVICES=<gpu_index>`` means the same device as the NVML index checked by
``ensure_safe``. ``quit`` is only sent for a daemon that provably started during our own start
attempt (our pipe answered, or a control process with our pipe dir in its environment whose start
time is after our ``-d`` call began).

Recording thread percentages
----------------------------
The per-client ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`` is a requested upper limit set at client
launch; it cannot be read back through ``gso.providers.mps.query``, whose ``thread_pct`` is the
server's own setting. Record both, labeled "requested (client)" and "server setting".
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import signal
import stat
import subprocess
import time
import warnings
from collections.abc import Callable, Collection
from pathlib import Path

from gso.providers import mps as _provider
from gso.providers.nvml import proc_identity

_BINARY = "nvidia-cuda-mps-control"
_PIPE_ENV = "CUDA_MPS_PIPE_DIRECTORY"
_LOG_ENV = "CUDA_MPS_LOG_DIRECTORY"
_THREAD_ENV = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
_CMDRE = r"(^|/)nvidia-cuda-mps-(control|server)( |$)"
_CMD_RE = re.compile(_CMDRE)
_CONTROL_ALLOWED = ("quit",)
_CTL_TIMEOUT_S = 5
_POLL_S = 0.1


class UnsafeGpuError(Exception):
    """The GPU or machine is not in a state this tool may use. The message names the GPU and
    counts of what was found, never other users' PIDs or process names."""


# ---- safety check -------------------------------------------------------------------------


def _pgrep(args: list[str]) -> list[str]:
    r = subprocess.run(
        ["pgrep", *args],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
    )
    if r.returncode == 1:
        return []
    if r.returncode != 0:
        raise RuntimeError(f"pgrep failed with exit {r.returncode}")
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def list_mps_processes() -> list[str]:
    """Every MPS control daemon or server on the machine (any user), as ``"<pid> <cmdline>"``.

    ``pgrep`` compares a 15-character truncated name, so ``pgrep nvidia-cuda-mps-control`` never
    matches; ``-x nvidia-cuda-mps`` matches both programs and ``-f`` catches the command line.
    False positives are acceptable, misses are not. Raises if ``pgrep`` itself fails."""
    seen: dict[int, str] = {}
    for line in _pgrep(["-a", "-x", "nvidia-cuda-mps"]) + _pgrep(["-a", "-f", _CMDRE]):
        seen.setdefault(int(line.split(None, 1)[0]), line)
    return [seen[p] for p in sorted(seen)]


def _entry_pid(entry) -> int:
    if isinstance(entry, int):
        return entry
    return int(str(entry).split(None, 1)[0])


def ensure_safe(
    gpu_index: int,
    nvml=None,
    own_pids: Collection[int] = frozenset(),
    mps_listing: Callable[[], list] | None = None,
) -> None:
    """Raise ``UnsafeGpuError`` unless the GPU and the machine are safe to use.

    Refused when: ``gpu_index`` is not a valid device; the GPU has a compute process (plain or
    MPS) whose PID is not in ``own_pids``; or an MPS control daemon/server exists anywhere on
    the machine whose PID is not in ``own_pids``. ``own_pids`` is normally
    ``session.own_daemon_pids | your client PIDs``. If safety cannot be established (NVML or
    process listing error) it refuses too. That includes a driver that does not support the MPS
    compute-process query (``NotSupported``): without it a process running under another user's
    MPS server cannot be ruled out, so the GPU is not provably free."""
    own = set(own_pids)
    listing = mps_listing if mps_listing is not None else list_mps_processes

    if nvml is None:
        try:
            import pynvml as nvml
        except Exception as e:
            raise UnsafeGpuError(f"cannot import NVML to check GPU {gpu_index}: {e}") from e

    foreign = _foreign_gpu_processes(gpu_index, nvml, own)
    if foreign:
        raise UnsafeGpuError(
            f"GPU {gpu_index} has {foreign} compute process(es) not started by this session"
        )

    try:
        entries = list(listing())
        others = sum(1 for e in entries if _entry_pid(e) not in own)
    except Exception as e:
        raise UnsafeGpuError(f"cannot list MPS processes ({e}); refusing") from e
    if others:
        raise UnsafeGpuError(
            f"{others} MPS control daemon/server process(es) already exist on this machine "
            "that this session did not start"
        )


def _foreign_gpu_processes(gpu_index, nvml, own: set[int]) -> int:
    inited = False
    try:
        try:
            nvml.nvmlInit()
            inited = True
            if type(gpu_index) is not int:
                raise UnsafeGpuError(f"invalid GPU index {gpu_index!r}")
            count = nvml.nvmlDeviceGetCount()
            if not 0 <= gpu_index < count:
                raise UnsafeGpuError(f"invalid GPU index {gpu_index} (machine has {count})")
            handle = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
            pids: list[int] = []
            for fn in (
                nvml.nvmlDeviceGetComputeRunningProcesses,
                nvml.nvmlDeviceGetMPSComputeRunningProcesses,
            ):
                pids += [p.pid for p in fn(handle)]
            return sum(1 for p in pids if p not in own)
        except UnsafeGpuError:
            raise
        except Exception as e:
            raise UnsafeGpuError(f"cannot verify GPU {gpu_index} is idle: {e}") from e
    finally:
        if inited:
            with contextlib.suppress(Exception):
                nvml.nvmlShutdown()


# ---- control channel (quit) ---------------------------------------------------------------


def _send_control(command: str, pipe_dir: str, log_dir: str, run) -> subprocess.CompletedProcess:
    """The only place ``quit`` leaves this module; anything but the exact string ``quit`` is
    refused. Kept apart from the read-only provider's allow-list on purpose."""
    if type(command) is not str or command not in _CONTROL_ALLOWED:
        raise ValueError(f"MPS control command not allowed: {command!r}")
    command = _CONTROL_ALLOWED[_CONTROL_ALLOWED.index(command)]
    env = {**os.environ, _PIPE_ENV: pipe_dir, _LOG_ENV: log_dir}
    return run(
        [_BINARY],
        input=f"{command}\n",
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=_CTL_TIMEOUT_S,
        env=env,
    )


# ---- process discovery --------------------------------------------------------------------


def find_own_processes(
    pipe_dir: str, uid: int | None = None, proc_root: str = "/proc"
) -> list[tuple[int, str]]:
    """``(pid, "control"|"server")`` of this user's MPS processes whose environment points at
    ``pipe_dir``. Unreadable or vanished processes are skipped (they are not provably ours)."""
    uid = os.getuid() if uid is None else uid
    out: list[tuple[int, str]] = []
    try:
        names = os.listdir(proc_root)
    except OSError:
        return out
    for name in names:
        if not name.isdigit():
            continue
        base = f"{proc_root}/{name}"
        try:
            if os.stat(base).st_uid != uid:
                continue
            with open(f"{base}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            if not _CMD_RE.search(cmd):
                continue
            with open(f"{base}/environ", "rb") as f:
                env = f.read().split(b"\0")
        except OSError:
            continue
        if f"{_PIPE_ENV}={pipe_dir}".encode() not in env:
            continue
        out.append((int(name), "server" if "mps-server" in cmd else "control"))
    return out


# ---- session ------------------------------------------------------------------------------


def _owned_by_me(path: Path) -> bool:
    return path.lstat().st_uid == os.getuid()


class MpsSession:
    """Start a user-scoped MPS daemon for ONE GPU and stop exactly that daemon afterwards.

    ``with MpsSession(2) as s:`` -> ``s.client_env(pct)`` and ``--gpu s.client_gpu_arg`` for the
    clients. Never touches the system default pipe directory or GPU compute mode, never uses
    sudo, never signals a process it cannot prove is its own."""

    def __init__(
        self,
        gpu_index: int,
        base_dir: Path | None = None,
        run=subprocess.run,
        nvml=None,
        mps_listing: Callable[[], list] | None = None,
        start_timeout_s: float = 10.0,
        *,
        stop_timeout_s: float = 10.0,
        find_own: Callable[[], list[tuple[int, str]]] | None = None,
        owner_check: Callable[[Path], bool] = _owned_by_me,
        kill: Callable[[int, int], None] = os.kill,
    ):
        self.gpu_index = gpu_index
        self.base_dir = (
            Path(base_dir) if base_dir is not None else Path(f"/tmp/gso-mps-{os.getuid()}")
        )
        self._run = run
        self._nvml = nvml
        self._listing = mps_listing
        self._start_timeout_s = start_timeout_s
        self._stop_timeout_s = stop_timeout_s
        self._owner_check = owner_check
        self._kill = kill
        self._find_own = find_own or (lambda: find_own_processes(self.pipe_dir))
        self._identity = proc_identity
        self._start_time = process_start_time
        self._lock_fd: int | None = None
        self._started = False
        self._control: dict[int, tuple[int, int] | None] = {}  # pid -> identity at start
        self._servers: dict[int, tuple[int, int] | None] = {}

    # -- paths / env --
    @property
    def pipe_dir(self) -> str:
        return str(self.base_dir / "pipe")

    @property
    def log_dir(self) -> str:
        return str(self.base_dir / "log")

    @property
    def client_gpu_arg(self) -> int:
        """Device index a client must pass as ``--gpu``. The daemon is started with
        ``CUDA_VISIBLE_DEVICES=<host gpu>``, so inside every client its single visible GPU is
        device 0 (measured, Task 6: ``--gpu 1`` failed with "No CUDA GPUs are available"; ``--gpu 0``
        ran on the intended host GPU, shown as ``M+C``)."""
        return 0

    def client_env(self, thread_pct: int | None) -> dict[str, str]:
        env = {_PIPE_ENV: self.pipe_dir, _LOG_ENV: self.log_dir}
        if thread_pct is not None:
            if type(thread_pct) is not int or not 1 <= thread_pct <= 100:
                raise ValueError(f"thread_pct must be an int in 1..100, got {thread_pct!r}")
            env[_THREAD_ENV] = str(thread_pct)
        return env

    @property
    def own_daemon_pids(self) -> frozenset[int]:
        """PIDs of this session's control daemon and servers, for ``ensure_safe(own_pids=...)``."""
        pids = set(self._control) | set(self._servers)
        if self._started:
            pids |= set(self._query().server_pids)
        return frozenset(pids)

    # -- helpers --
    def _query(self):
        return _provider.query(pipe_dir=self.pipe_dir, run=self._run)

    def _prepare_base(self) -> None:
        b = self.base_dir
        if b.is_symlink():
            raise UnsafeGpuError(f"refusing {b}: it is a symlink")
        if b.exists() and (not b.is_dir() or not self._owner_check(b)):
            raise UnsafeGpuError(f"refusing {b}: not a directory owned by this user")
        b.mkdir(mode=0o700, parents=True, exist_ok=True)
        if b.is_symlink() or not stat.S_ISDIR(b.lstat().st_mode) or not self._owner_check(b):
            raise UnsafeGpuError(f"refusing {b}: not a directory owned by this user")
        os.chmod(b, 0o700)
        for sub in (self.pipe_dir, self.log_dir):
            Path(sub).mkdir(mode=0o700, exist_ok=True)

    def _env(self) -> dict[str, str]:
        return {
            **os.environ,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",  # override: index must match NVML's
            "CUDA_VISIBLE_DEVICES": str(self.gpu_index),
            _PIPE_ENV: self.pipe_dir,
            _LOG_ENV: self.log_dir,
        }

    # -- lifecycle --
    def __enter__(self) -> MpsSession:  # noqa: PYI034
        try:
            self._prepare_base()
        except OSError as e:
            raise UnsafeGpuError(f"cannot prepare {self.base_dir}: {e}") from e
        self._acquire_lock()
        try:
            ensure_safe(self.gpu_index, nvml=self._nvml, mps_listing=self._listing)
            self._start()
        except BaseException:
            try:
                self._stop(raise_on_leftover=False)
            finally:
                self._release_lock()
            raise
        return self

    def _acquire_lock(self) -> None:
        path = self.base_dir / "session.lock"
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        except OSError as e:
            raise UnsafeGpuError(f"cannot open session lock {path}: {e}") from e
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise UnsafeGpuError("another gso MPS session of this user is active") from None
        self._lock_fd = fd

    def _release_lock(self) -> None:
        fd, self._lock_fd = self._lock_fd, None
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)  # closing drops the flock

    def _adopt_if_started_by_us(self, t0_wall: float) -> None:
        """After a failed start: take responsibility for a control process of ours (pipe dir in
        its environment) that began after our ``-d`` call did. Anything older is not ours."""
        for pid, kind in self._find_own():
            started = self._start_time(pid)
            if kind == "control" and started is not None and started >= t0_wall - 0.05:
                self._control[pid] = self._identity(pid)
                self._started = True

    def _start(self) -> None:
        if self._query().active:
            raise UnsafeGpuError(f"an MPS daemon already answers on {self.pipe_dir}; refusing")
        t0_wall = time.time()
        try:
            r = self._run(
                [_BINARY, "-d"],
                env=self._env(),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._start_timeout_s,
            )
        except (OSError, subprocess.SubprocessError) as e:
            self._adopt_if_started_by_us(t0_wall)
            raise UnsafeGpuError(f"could not start the MPS daemon: {e}") from e
        if r.returncode != 0:
            self._adopt_if_started_by_us(t0_wall)
            detail = (r.stderr or r.stdout or "").strip()[:200]
            raise UnsafeGpuError(f"MPS daemon failed to start (exit {r.returncode}): {detail}")
        deadline = time.monotonic() + self._start_timeout_s
        while not self._query().active:
            if time.monotonic() >= deadline:
                self._adopt_if_started_by_us(t0_wall)
                raise UnsafeGpuError(
                    f"MPS daemon did not become active within {self._start_timeout_s:g}s"
                )
            time.sleep(_POLL_S)
        self._started = True  # -d returned 0 and our pipe answers: cleanup is now ours
        for pid, kind in self._find_own():
            if kind == "control":
                self._control[pid] = self._identity(pid)
        if not self._control:
            raise UnsafeGpuError(
                "MPS daemon is up but its process could not be identified through our pipe dir"
            )

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self._stop(raise_on_leftover=exc_type is None)
        finally:
            self._release_lock()
        return False

    def _stop(self, raise_on_leftover: bool) -> None:
        if not self._started:
            return
        self._started = False
        for pid in self._query().server_pids:  # also answers "is the daemon even up"
            self._servers.setdefault(pid, self._identity(pid))
        # leftovers are checked below, whatever happened to the quit call
        with contextlib.suppress(Exception):
            _send_control("quit", self.pipe_dir, self.log_dir, self._run)
        remaining = self._wait_gone(self._stop_timeout_s)
        if remaining:
            self._terminate_control_if_ours()
            remaining = self._wait_gone(min(2.0, self._stop_timeout_s))
        if remaining:
            msg = (
                "MPS daemon of this session is still running after quit: "
                + ", ".join(remaining)
                + f" (pipe dir {self.pipe_dir}); not killed because it could not be proven ours"
            )
            if raise_on_leftover:
                raise RuntimeError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=3)

    def _wait_gone(self, timeout_s: float) -> list[str]:
        deadline = time.monotonic() + timeout_s
        while True:
            left = self._remaining()
            if not left or time.monotonic() >= deadline:
                return left
            time.sleep(min(_POLL_S, timeout_s / 5 or _POLL_S))

    def _remaining(self) -> list[str]:
        left = [f"{kind} pid {pid}" for pid, kind in self._find_own()]
        known = {**self._control, **self._servers}
        for pid, ident in known.items():
            now = self._identity(pid)
            if ident is not None and now == ident:
                tag = f"pid {pid}"
                if not any(s.endswith(tag) for s in left):
                    left.append(f"process {tag}")
        if self._query().active:
            left.append("control daemon still answers on our pipe")
        return left

    def _terminate_control_if_ours(self) -> None:
        """SIGTERM the recorded control daemon, only if it is still the same process (same start
        time as when we started it) and still provably ours (found through our pipe dir).
        With ``pidfd_open`` the identity is checked on the process the descriptor pins, so the
        PID cannot be reused between check and signal; otherwise the check is best-effort."""
        live = {pid for pid, kind in self._find_own() if kind == "control"}
        for pid, ident in self._control.items():
            if ident is None or pid not in live:
                continue
            try:
                if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                    fd = os.pidfd_open(pid)
                    try:
                        if self._identity(pid) == ident:
                            signal.pidfd_send_signal(fd, signal.SIGTERM)
                    finally:
                        os.close(fd)
                elif self._identity(pid) == ident:
                    self._kill(pid, signal.SIGTERM)
            except OSError:
                pass


def process_start_time(pid: int, proc_root: str = "/proc") -> float | None:
    """Wall-clock start time of ``pid`` (seconds since the epoch), or None if unreadable.

    ``/proc/<pid>/stat`` field 22 is the start time in clock ticks since boot; boot time is
    ``now - /proc/uptime``."""
    ident = proc_identity(pid, proc_root)
    if ident is None:
        return None
    try:
        with open(f"{proc_root}/uptime", encoding="utf-8") as f:
            uptime = float(f.read().split()[0])
        return time.time() - uptime + ident[1] / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None

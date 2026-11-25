"""nsys on-demand 캡처 래퍼: 5~10초 캡처 후 SQLite로 내보낸다.

관측만 한다. 이 모듈은 nsys를 실행해 ``<name>.nsys-rep``와 ``<name>.sqlite``를
``out_dir`` 안에 만들 뿐이며, 적재와 분석은 하지 않는다. sudo를 쓰지 않는다.

두 모드(docs/verification/RESULTS.md 결정 1):
- global:   대상이 ``sleep <D>``. 모든 GPU의 외부 프로세스 컨텍스트 전환 이벤트를 본다.
            CUDA 커널 테이블은 없다.
- launched: 호출자가 준 명령을 nsys가 직접 띄운다. 그 프로세스의 CUDA 커널/런타임
            테이블을 본다. 컨텍스트 전환 이벤트는 거의 없다.
CPU 샘플링과 OS 스레드 추적은 끈다(perf_event_paranoid 때문에 비루트는 쓸 수 없다).
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from gso.capability import Capabilities

MIN_DURATION_S = 5
MAX_DURATION_S = 10
PROFILE_TIMEOUT_MARGIN_S = 60
EXPORT_TIMEOUT_S = 120
_TAIL_LINES = 5
_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


class CaptureUnavailable(Exception):
    """nsys를 찾지 못했다."""


class CaptureError(Exception):
    """nsys가 실행됐지만 실패했다(종료 코드, 시간 초과, 출력 파일 없음)."""


@dataclass(frozen=True)
class CaptureResult:
    sqlite_path: Path
    rep_path: Path
    mode: str  # "global" | "launched"
    masked: bool
    duration_s: int
    started_ns: int
    wall_s: float
    warnings: tuple[str, ...]


def _check_duration(duration_s) -> None:
    if (
        isinstance(duration_s, bool)
        or not isinstance(duration_s, int)
        or not MIN_DURATION_S <= duration_s <= MAX_DURATION_S
    ):
        raise ValueError(
            f"duration_s must be an int in {MIN_DURATION_S}..{MAX_DURATION_S}, got {duration_s!r}"
        )


def build_command(
    nsys: str,
    duration_s: int,
    out: Path,
    gpuctxsw: bool = True,
    target: Sequence[str] | None = None,
) -> list[str]:
    """``nsys profile`` argv. ``out``은 확장자 없는 경로(nsys가 .nsys-rep을 붙인다)."""
    _check_duration(duration_s)
    if target is not None and len(target) == 0:
        raise ValueError("target must be a non-empty command or None")
    argv = [nsys, "profile"]
    if gpuctxsw:
        argv.append("--gpuctxsw=true")
    argv += ["--sample=none", "--cpuctxsw=none"]
    if target is not None:
        # 기본 trace 집합(cuda,nvtx,osrt,opengl) 대신 CUDA만: 커널/런타임 테이블에 필요한 최소.
        argv.append("--trace=cuda")
    argv += [f"--duration={duration_s}", "--force-overwrite=true", "-o", str(out)]
    argv += list(target) if target is not None else ["sleep", str(duration_s)]
    return argv


def build_export_command(nsys: str, rep: Path, sqlite: Path) -> list[str]:
    return [nsys, "export", "--type=sqlite", "--force-overwrite=true", "-o", str(sqlite), str(rep)]


# 기본 runner. nsys는 보조 프로세스를 띄우므로 subprocess.run(timeout=)만으로는 시간 초과 때
# 손자 프로세스가 남을 수 있다. 그래서 자체 세션(=프로세스 그룹)에서 시작하고, 시간 초과나
# 중단 시 "우리가 만든 그 그룹"만 SIGTERM, 잠시 뒤 SIGKILL 한다. 이름으로 찾아 죽이지 않는다.
def _run_in_group(argv, *, capture_output=True, text=True, timeout=None, env=None, errors="replace"):
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=text,
        errors=errors if text else None,
        env=env,
        start_new_session=True,
    )
    pgid = proc.pid  # start_new_session: 이 프로세스가 새 그룹의 리더

    def kill_group(sig):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, sig)

    try:
        out, err = proc.communicate(timeout=timeout)
    except BaseException:  # TimeoutExpired, KeyboardInterrupt 포함
        kill_group(signal.SIGTERM)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            kill_group(signal.SIGKILL)
            proc.communicate()
        raise
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def _tail(*streams) -> str:
    lines = [ln.strip() for s in streams if s for ln in str(s).splitlines() if ln.strip()]
    return "\n".join(lines[-_TAIL_LINES:])


def _remove(*paths: Path) -> None:
    for p in paths:
        with contextlib.suppress(FileNotFoundError):
            p.unlink()


def _validate_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or name.startswith(".")
        or not set(name) <= _NAME_CHARS
    ):
        raise ValueError(f"name must be a plain file stem (letters, digits, - _ .), got {name!r}")


def _exec(run, argv, timeout, env, what: str, tail_note: str = ""):
    try:
        res = run(argv, capture_output=True, text=True, timeout=timeout, env=env, errors="replace")
    except subprocess.TimeoutExpired as e:
        raise CaptureError(f"nsys {what} 시간 초과 (timeout {timeout}s)") from e
    except OSError as e:
        raise CaptureError(f"nsys {what} 실행 실패: {e}") from e
    if res.returncode != 0:
        raise CaptureError(
            f"nsys {what} 실패 (exit code {res.returncode}):\n{_tail(res.stdout, res.stderr)}"
        )
    return res


def capture(
    duration_s: int,
    out_dir: Path,
    caps: Capabilities,
    run=None,
    target: Sequence[str] | None = None,
    name: str = "capture",
    env: Mapping[str, str] | None = None,
    clock_ns=time.time_ns,
) -> CaptureResult:
    """on-demand 캡처 한 번: nsys profile, 이어서 nsys export. 실패 시 이번 캡처의 산출물을 지운다.

    run은 subprocess.run 호환 호출 가능 객체이며, 생략하면 프로세스 그룹을 정리하는
    기본 runner를 쓴다.
    """
    if caps.nsys_path is None:
        raise CaptureUnavailable(
            "nsys not found: set the NSYS environment variable to its path or put nsys on PATH"
        )
    _check_duration(duration_s)
    _validate_name(name)
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise ValueError(f"out_dir is not an existing directory: {out_dir}")
    if run is None:
        run = _run_in_group

    base = out_dir / name
    rep = out_dir / f"{name}.nsys-rep"
    sqlite = out_dir / f"{name}.sqlite"
    profile_argv = build_command(caps.nsys_path, duration_s, base, target=target)
    export_argv = build_export_command(caps.nsys_path, rep, sqlite)

    _remove(rep, sqlite)  # 같은 이름의 옛 산출물을 이번 결과로 오인하지 않도록
    started_ns = clock_ns()
    t0 = time.monotonic()
    try:
        _exec(run, profile_argv, duration_s + PROFILE_TIMEOUT_MARGIN_S, env, "profile")
        if not rep.is_file():
            raise CaptureError(f"nsys profile exited 0 but produced no {rep.name}")
        _exec(run, export_argv, EXPORT_TIMEOUT_S, None, "export")
        if not sqlite.is_file():
            raise CaptureError(f"nsys export exited 0 but produced no {sqlite.name}")
    except BaseException:
        _remove(rep, sqlite)
        raise
    wall_s = time.monotonic() - t0

    masked = not caps.is_root
    warnings: list[str] = []
    if masked:
        warnings.append("root가 아니어서 다른 사용자 프로세스는 마스킹된 PID로 나타납니다.")
    if target is None:
        warnings.append("외부 프로세스의 CUDA 커널 구간은 이 캡처에 기록되지 않습니다.")
        mode = "global"
    else:
        warnings.append("컨텍스트 전환 이벤트는 직접 실행한 프로세스 주변에서만 기록됩니다.")
        mode = "launched"
    return CaptureResult(
        sqlite_path=sqlite,
        rep_path=rep,
        mode=mode,
        masked=masked,
        duration_s=duration_s,
        started_ns=started_ns,
        wall_s=wall_s,
        warnings=tuple(warnings),
    )

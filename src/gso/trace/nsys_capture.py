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
import functools
import os
import signal
import subprocess
import tempfile
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
_CANCEL_POLL_S = 0.2
_SIGTERM_EXIT = 143  # 128 + SIGTERM
_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
)


class CaptureUnavailable(Exception):
    """nsys를 찾지 못했다."""


class CaptureError(Exception):
    """nsys가 실행됐지만 실패했다(종료 코드, 시간 초과, 출력 파일 없음)."""


class CaptureCancelled(Exception):
    """다른 스레드가 ``cancellable_runner``의 취소 이벤트를 세워 캡처를 중단했다."""


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
    target_terminated: bool = False  # end_target=True 요청대로 캡처가 대상을 끝냈다


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
    end_target: bool = False,
) -> list[str]:
    """``nsys profile`` argv. ``out``은 확장자 없는 경로(nsys가 .nsys-rep을 붙인다).

    launched 모드는 항상 ``--kill``을 명시한다(nsys 기본은 sigterm이라 기간 끝에 대상을 죽인다):
    ``end_target=False``면 ``--kill=none``, True면 ``--kill=sigterm``.
    """
    _check_duration(duration_s)
    if end_target and target is None:
        raise ValueError("end_target=True requires a launched target")
    if target is not None and len(target) == 0:
        raise ValueError("target must be a non-empty command or None")
    argv = [nsys, "profile"]
    if gpuctxsw:
        argv.append("--gpuctxsw=true")
    argv += ["--sample=none", "--cpuctxsw=none"]
    if target is not None:
        # 기본 trace 집합(cuda,nvtx,osrt,opengl) 대신 CUDA만: 커널/런타임 테이블에 필요한 최소.
        argv.append("--trace=cuda")
        argv.append("--kill=sigterm" if end_target else "--kill=none")
    argv += [f"--duration={duration_s}", "--force-overwrite=true", "-o", str(out)]
    argv += list(target) if target is not None else ["sleep", str(duration_s)]
    return argv


def build_export_command(nsys: str, rep: Path, sqlite: Path) -> list[str]:
    return [
        nsys,
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        "-o",
        str(sqlite),
        str(rep),
    ]


# 기본 runner. nsys는 보조 프로세스를 띄우므로 subprocess.run(timeout=)만으로는 시간 초과 때
# 손자 프로세스가 남을 수 있다. 그래서 자체 세션(=프로세스 그룹)에서 시작하고, 시간 초과나
# 중단 시 "우리가 만든 그 그룹"만 SIGTERM, 잠시 뒤 SIGKILL 한다. 이름으로 찾아 죽이지 않는다.
# 정상 반환에서는 그룹에 신호를 보내지 않는다(--kill=none으로 살려 둔 대상이 계속 실행된다).
# 출력은 파이프 대신 임시 파일로 받는다: 살아남은 대상이 상속한 파이프를 계속 잡고 있으면
# communicate()가 EOF를 기다리며 nsys가 끝난 뒤에도 돌아오지 않기 때문이다.
def _wait_cancellable(proc, timeout, cancel, argv) -> None:
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        if cancel.is_set():
            raise CaptureCancelled("capture cancelled")
        left = None if deadline is None else deadline - time.monotonic()
        if left is not None and left <= 0:
            raise subprocess.TimeoutExpired(argv, timeout)
        try:
            proc.wait(
                timeout=_CANCEL_POLL_S if left is None else min(_CANCEL_POLL_S, left)
            )
            return
        except subprocess.TimeoutExpired:
            continue


def _run_in_group(
    argv,
    *,
    capture_output=True,
    text=True,
    timeout=None,
    env=None,
    errors="replace",
    cancel=None,
):
    with tempfile.TemporaryFile() as fout, tempfile.TemporaryFile() as ferr:
        proc = subprocess.Popen(
            argv,
            stdout=fout if capture_output else None,
            stderr=ferr if capture_output else None,
            env=env,
            start_new_session=True,
        )
        pgid = proc.pid  # start_new_session: 이 프로세스가 새 그룹의 리더

        def kill_group(sig):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, sig)

        try:
            if cancel is None:
                proc.wait(timeout=timeout)
            else:
                _wait_cancellable(proc, timeout, cancel, argv)
        except BaseException:  # TimeoutExpired, KeyboardInterrupt 포함
            kill_group(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                kill_group(signal.SIGKILL)
                proc.wait()
            raise
        out = err = None
        if capture_output:
            fout.seek(0)
            ferr.seek(0)
            out, err = fout.read(), ferr.read()
            if text:
                out, err = out.decode(errors=errors), err.decode(errors=errors)
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def cancellable_runner(cancel):
    """기본 runner와 같되, 다른 스레드가 ``cancel``(threading.Event)을 세우면 자신이 만든
    프로세스 그룹을 종료하고 ``CaptureCancelled``를 일으킨다. ``capture(run=...)``에 넘긴다.
    ``capture``는 그 예외에서도 이번 호출이 만든 파일을 지우고 다시 던진다."""
    return functools.partial(_run_in_group, cancel=cancel)


def _tail(*streams) -> str:
    lines = [
        ln.strip() for s in streams if s for ln in str(s).splitlines() if ln.strip()
    ]
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
        raise ValueError(
            f"name must be a plain file stem (letters, digits, - _ .), got {name!r}"
        )


def _exec(run, argv, timeout, env, what: str, ok_codes=(0,)):
    try:
        res = run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            errors="replace",
        )
    except subprocess.TimeoutExpired as e:
        raise CaptureError(f"nsys {what} 시간 초과 (timeout {timeout}s)") from e
    except OSError as e:
        raise CaptureError(f"nsys {what} 실행 실패: {e}") from e
    if ok_codes is not None and res.returncode not in ok_codes:
        raise CaptureError(
            f"nsys {what} 실패 (exit code {res.returncode}):\n{_tail(res.stdout, res.stderr)}"
        )
    return res


def _signature(path: Path):
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def _fresh_nonempty(path: Path, before) -> bool:
    """이번 호출이 만들었거나 다시 쓴 비어 있지 않은 파일인가."""
    sig = _signature(path)
    return sig is not None and sig != before and sig[1] > 0


def capture(
    duration_s: int,
    out_dir: Path,
    caps: Capabilities,
    run=None,
    target: Sequence[str] | None = None,
    name: str = "capture",
    env: Mapping[str, str] | None = None,
    clock_ns=time.time_ns,
    end_target: bool = False,
    monotonic=time.monotonic,
) -> CaptureResult:
    """on-demand 캡처 한 번: nsys profile, 이어서 nsys export.

    산출물은 ``<out_dir>/<name>.nsys-rep``와 ``.sqlite``뿐이다. 같은 이름의 기존 파일은
    미리 지우지 않고 nsys가 덮어쓰게 한다. 실패하면 이번 호출이 새로 만들었거나 다시 쓴
    파일(inode/크기/mtime이 달라진 것)만 지우고, 그대로인 기존 파일은 남긴다. 성공 판정도
    그런 "새로 쓴 비어 있지 않은 파일"만 인정한다.

    launched 모드(``target`` 지정): ``end_target=False``(기본)이면 ``--kill=none``이라 캡처
    기간이 끝나도 nsys가 대상을 끝내지 않는다. 이때 ``capture`` 반환 시점에 대상이 아직
    실행 중일 수 있고, 그 수명은 호출자 몫이다(실측: 대상은 nsys 종료 후에도 계속 돌고 스스로
    끝난다). 대상이 기간보다 먼저 끝나면 nsys도 그때 끝나므로 실제 캡처는 ``duration_s``보다
    짧을 수 있다. ``end_target=True``면 ``--kill=sigterm``: 기간 끝에 nsys가 대상을 끝내고
    대상의 종료 코드(143, 130, 1 등)로 끝나며, 보고서가 비어 있지 않고 프로파일 경과 시간이 ``duration_s``
    이상일 때만 성공(``target_terminated=True``)으로 본다. 그 밖의 비정상 종료는
    모두 CaptureError다. ``end_target=True``는 global 모드와 함께 쓸 수 없다(ValueError).

    시간 초과나 중단(KeyboardInterrupt)이면 기본 runner가 자신이 만든 프로세스 그룹 전체를,
    살아 있는 대상까지 포함해 종료한다. 정상 반환에서는 그룹에 신호를 보내지 않는다.

    run은 subprocess.run 호환 호출 가능 객체이며 생략하면 위 기본 runner를 쓴다. 주입하는
    run은 nsys를 자체 세션/프로세스 그룹에서 시작해야 한다: ``end_target=True``면 nsys가
    대상의 프로세스 그룹에 신호를 보내므로, 호출자와 같은 그룹이면 호출자까지 맞을 수 있다.
    """
    if caps.nsys_path is None:
        raise CaptureUnavailable(
            "nsys not found: set the NSYS environment variable to its path or put nsys on PATH"
        )
    _check_duration(duration_s)
    _validate_name(name)
    if end_target and target is None:
        raise ValueError("end_target=True requires a launched target")
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise ValueError(f"out_dir is not an existing directory: {out_dir}")
    if run is None:
        run = _run_in_group

    base = out_dir / name
    rep = out_dir / f"{name}.nsys-rep"
    sqlite = out_dir / f"{name}.sqlite"
    profile_argv = build_command(
        caps.nsys_path, duration_s, base, target=target, end_target=end_target
    )
    export_argv = build_export_command(caps.nsys_path, rep, sqlite)

    before_rep, before_sqlite = _signature(rep), _signature(sqlite)
    started_ns = clock_ns()
    t0 = monotonic()
    terminated = False
    target_rc = None
    try:
        # end_target: nsys는 대상의 종료 코드로 끝난다(143, 130, 1 ...). 코드 자체로는 실패가 아니고
        # 아래의 새 보고서/기간 검사가 진짜 실패를 가른다.
        ok = None if end_target else (0,)
        prof = _exec(
            run, profile_argv, duration_s + PROFILE_TIMEOUT_MARGIN_S, env, "profile", ok
        )
        profile_s = monotonic() - t0
        if not _fresh_nonempty(rep, before_rep):
            raise CaptureError(
                f"nsys profile (exit code {prof.returncode}) produced no new non-empty {rep.name}:\n"
                f"{_tail(prof.stdout, prof.stderr)}"
            )
        if end_target and prof.returncode != 0:
            # 기간 끝에 우리가 대상을 끝내면 nsys는 대상의 종료 코드로 끝난다(143, 130, 1 ...).
            # 기간을 못 채웠다면 대상이 스스로 죽었거나 다른 원인이다.
            if profile_s < duration_s:
                raise CaptureError(
                    f"nsys profile exit code {prof.returncode} after {profile_s:.1f}s, "
                    f"before the {duration_s}s duration ended:\n{_tail(prof.stdout, prof.stderr)}"
                )
            terminated = True
            target_rc = prof.returncode
        _exec(run, export_argv, EXPORT_TIMEOUT_S, None, "export")
        if not _fresh_nonempty(sqlite, before_sqlite):
            raise CaptureError(
                f"nsys export exited 0 but produced no new non-empty {sqlite.name}"
            )
    except BaseException:
        for path, before in ((rep, before_rep), (sqlite, before_sqlite)):
            if _signature(path) != before:
                _remove(path)
        raise
    wall_s = monotonic() - t0

    masked = not caps.is_root
    warnings: list[str] = []
    if masked:
        warnings.append(
            "root가 아니어서 다른 사용자 프로세스는 마스킹된 PID로 나타납니다."
        )
    if target is None:
        warnings.append("외부 프로세스의 CUDA 커널 구간은 이 캡처에 기록되지 않습니다.")
        mode = "global"
    else:
        warnings.append(
            "컨텍스트 전환 이벤트는 직접 실행한 프로세스 주변에서만 기록됩니다."
        )
        warnings.append(
            "대상이 기간보다 먼저 종료되면 캡처도 그때 끝나므로 실제 캡처 길이는"
            " duration_s보다 짧을 수 있습니다."
        )
        if terminated:
            warnings.append(
                "end_target 요청대로 캡처 기간 끝에 직접 실행한 대상을 종료시켰습니다."
                " 대상의 자체 결과 파일은 만들어지지 않을 수 있습니다."
                f" (대상 종료 코드 {target_rc})"
            )
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
        target_terminated=terminated,
    )

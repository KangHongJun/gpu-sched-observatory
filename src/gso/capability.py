"""시작 시 한 번, 이 머신에서 무엇을 측정할 수 있는지 감지한다.

GPU 목록, DCGM 프로파일링 필드 지원, nsys 설치, root 여부, MPS 활성 여부.
감지 실패는 예외가 아니라 이유가 붙은 "사용 불가"로 돌려준다.
"""

from __future__ import annotations

import contextlib
import glob as _glob
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

DCGM_FIELD_IDS = (1001, 1002, 1003, 1004, 1005)
RUN_TIMEOUT_S = 5
_PROFILE_ROW = re.compile(r"^\|\s*\S+\s*\|\s*(\d+)\s*\|", re.MULTILINE)


@dataclass
class GpuInfo:
    index: int
    uuid: str
    name: str
    driver: str
    mig_enabled: bool


@dataclass
class DcgmStatus:
    available: bool
    reason: str
    fields: dict[int, bool]


@dataclass
class Capabilities:
    gpus: list[GpuInfo]
    dcgm: DcgmStatus
    nsys_path: str | None
    is_root: bool
    mps_active: bool
    nvml_error: str | None = None


def parse_profile_fields(text: str) -> dict[int, bool]:
    """`dcgmi profile -l` 출력에서 1001~1005 각각의 지원 여부를 돌려준다."""
    present = {int(m) for m in _PROFILE_ROW.findall(text)}
    return {fid: fid in present for fid in DCGM_FIELD_IDS}


def _s(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _query_nvml(nvml) -> tuple[list[GpuInfo], bool, str | None]:
    """(gpus, nvml 기준 MPS 활성, 오류 문자열). 항상 init/shutdown을 짝지어 호출한다."""
    try:
        nvml.nvmlInit()
    except Exception as e:  # noqa: BLE001 - NVMLError 계열 전부
        return [], False, f"NVML init failed: {e}"
    try:
        driver = _s(nvml.nvmlSystemGetDriverVersion())
        gpus: list[GpuInfo] = []
        mps = False
        for i in range(nvml.nvmlDeviceGetCount()):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            try:
                mig = nvml.nvmlDeviceGetMigMode(h)[0] == 1
            except nvml.NVMLError:
                mig = False
            try:
                if nvml.nvmlDeviceGetMPSComputeRunningProcesses(h):
                    mps = True
            except nvml.NVMLError:
                pass
            gpus.append(
                GpuInfo(
                    i,
                    _s(nvml.nvmlDeviceGetUUID(h)),
                    _s(nvml.nvmlDeviceGetName(h)),
                    driver,
                    mig,
                )
            )
        return gpus, mps, None
    except Exception as e:  # noqa: BLE001
        return [], False, f"NVML query failed: {e}"
    finally:
        with contextlib.suppress(Exception):
            nvml.nvmlShutdown()


def _detect_dcgm(cmd, index, run, which) -> DcgmStatus:
    none = {fid: False for fid in DCGM_FIELD_IDS}

    def no(reason: str) -> DcgmStatus:
        return DcgmStatus(False, reason, dict(none))

    if which(cmd[0]) is None:
        hint = (
            "install DCGM or pass --dcgmi-cmd"
            if cmd[0] == "dcgmi"
            else "install it or fix --dcgmi-cmd"
        )
        return no(f"{cmd[0]} not found in PATH; {hint}.")
    argv = [*cmd, "profile", "-l", "-i", str(index)]
    try:
        r = run(argv, capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return no(
            f"dcgmi timed out after {RUN_TIMEOUT_S}s; the DCGM host engine "
            "may be hung or unreachable."
        )
    except OSError as e:
        return no(f"could not run {' '.join(cmd)}: {e}")
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        detail = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
        return no(
            "dcgmi could not reach the DCGM host engine"
            + (f" ({detail})" if detail else "")
            + "; start nv-hostengine (or the DCGM container) and retry."
        )
    fields = parse_profile_fields(out)
    if not any(fields.values()):
        return no(
            "dcgmi profile -l listed none of the profiling fields 1001-1005 "
            f"for GPU index {index}; check --dcgm-gpu-index."
        )
    return DcgmStatus(True, "ok", fields)


def _is_exec(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _find_nsys(environ, which, glob) -> str | None:
    env = environ.get("NSYS")
    if env and _is_exec(env):
        return env
    found = which("nsys")
    if found:
        return found
    matches = glob(os.path.expanduser("~/opt/nsys/**/bin/nsys"))
    if matches:
        return max(matches, key=lambda p: (_mtime(p), p))
    return None


def _default_mps_listing() -> bool:
    # pgrep은 15자로 잘린 comm과 비교하므로 전체 이름 nvidia-cuda-mps-control은
    # 매칭되지 않는다. 잘린 이름 nvidia-cuda-mps(control·server 공통)를 정확 일치로 찾는다.
    r = subprocess.run(
        ["pgrep", "-x", "nvidia-cuda-mps"],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        check=False,
    )
    return r.returncode == 0


def detect(
    nvml=None,
    run: Callable = subprocess.run,
    which: Callable = shutil.which,
    dcgmi_cmd: list[str] | None = None,
    dcgm_gpu_index: int | None = None,
    *,
    environ=None,
    glob: Callable | None = None,
    is_root: Callable[[], bool] | None = None,
    mps_listing: Callable[[], bool] | None = None,
) -> Capabilities:
    if nvml is None:
        import pynvml as nvml  # 지연 import: 모듈 import 시 GPU를 건드리지 않는다
    cmd = list(dcgmi_cmd) if dcgmi_cmd else ["dcgmi"]
    environ = os.environ if environ is None else environ
    glob = glob or (lambda p: _glob.glob(p, recursive=True))
    is_root = is_root or (lambda: os.geteuid() == 0)
    mps_listing = mps_listing or _default_mps_listing

    gpus, nvml_mps, nvml_error = _query_nvml(nvml)
    index = dcgm_gpu_index
    if index is None:
        index = gpus[0].index if gpus else 0
    try:
        listing_mps = bool(mps_listing())
    except Exception:  # noqa: BLE001
        listing_mps = False
    return Capabilities(
        gpus=gpus,
        dcgm=_detect_dcgm(cmd, index, run, which),
        nsys_path=_find_nsys(environ, which, glob),
        is_root=bool(is_root()),
        mps_active=nvml_mps or listing_mps,
        nvml_error=nvml_error,
    )

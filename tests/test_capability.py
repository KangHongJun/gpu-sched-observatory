import subprocess
from pathlib import Path

import pytest

from gso import capability, cli
from gso.capability import (
    Capabilities,
    DcgmStatus,
    GpuInfo,
    detect,
    parse_profile_fields,
)
from tests.fakes import FakeNvml, fake_run

FIXTURE = Path(__file__).parent / "fixtures" / "dcgmi_profile_l_a5000.txt"
PROFILE_TEXT = FIXTURE.read_text()
DOCKER_CMD = ["docker", "exec", "gso-dcgm", "dcgmi"]


@pytest.fixture(autouse=True)
def hermetic(monkeypatch, tmp_path):
    """기본 인자(실제 환경 변수, ~/opt/nsys 글롭, pgrep)가 호스트 상태에 좌우되지 않게 한다."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NSYS", raising=False)
    monkeypatch.setattr(capability, "_default_mps_listing", lambda: False)


def any_which(n):
    return "/usr/bin/" + n


def no_mps():
    return False


def det(**kw):
    kw.setdefault("nvml", FakeNvml())
    kw.setdefault("which", lambda _: None)
    kw.setdefault("environ", {})
    kw.setdefault("glob", lambda _: [])
    kw.setdefault("is_root", lambda: False)
    kw.setdefault("mps_listing", no_mps)
    return detect(**kw)


# --- 브리프 테스트 -------------------------------------------------------
def test_dcgmi_missing():
    caps = detect(nvml=FakeNvml(), which=lambda _: None)
    assert caps.dcgm.available is False and "dcgmi not found" in caps.dcgm.reason


def test_hostengine_down():
    caps = detect(
        nvml=FakeNvml(),
        which=lambda n: "/usr/bin/" + n,
        run=fake_run(returncode=1, stdout="Unable to connect to host engine"),
    )
    assert caps.dcgm.available is False and "host engine" in caps.dcgm.reason


def test_nsys_missing_is_not_an_error():
    assert detect(nvml=FakeNvml(), which=lambda _: None).nsys_path is None


# --- DCGM ---------------------------------------------------------------
def test_parse_profile_fields_fixture():
    fields = parse_profile_fields(PROFILE_TEXT)
    assert all(fields[i] for i in (1001, 1002, 1003, 1004, 1005))
    assert 1006 not in fields


def test_parse_profile_fields_negative_cases():
    no_1004 = "| A.1 | 1002 | sm_active |\n| D.0 | 1001 | gr_engine_active |\n"
    assert parse_profile_fields(no_1004)[1004] is False
    lookalike = "| A.1 | 11001 | something |\n"
    fields = parse_profile_fields(lookalike)
    assert fields[1001] is False and fields[1004] is False


def test_dcgm_available_path():
    run = fake_run(stdout=PROFILE_TEXT)
    caps = det(which=any_which, run=run)
    assert caps.dcgm.available is True
    assert caps.dcgm.fields == {i: True for i in (1001, 1002, 1003, 1004, 1005)}
    assert run.argvs[0] == ["dcgmi", "profile", "-l", "-i", "0"]
    assert run.kwargs[0]["timeout"] == 5


def test_dcgm_partial_fields():
    text = "| A.1 | 1002 | sm_active |\n| D.0 | 1001 | gr_engine_active |\n"
    caps = det(which=any_which, run=fake_run(stdout=text))
    assert caps.dcgm.fields[1001] and caps.dcgm.fields[1002]
    assert not caps.dcgm.fields[1004]


def test_dcgm_unavailable_fields_all_false():
    caps = det()
    assert caps.dcgm.fields == {i: False for i in (1001, 1002, 1003, 1004, 1005)}


def test_docker_missing_names_docker():
    caps = det(dcgmi_cmd=DOCKER_CMD, run=fake_run(stdout=PROFILE_TEXT))
    assert caps.dcgm.available is False
    assert "docker" in caps.dcgm.reason


def test_dcgm_gpu_index_passed():
    run = fake_run(stdout=PROFILE_TEXT)
    det(
        nvml=FakeNvml(gpus=[{}, {}]),
        which=any_which,
        run=run,
        dcgmi_cmd=DOCKER_CMD,
        dcgm_gpu_index=0,
    )
    argv = run.argvs[0]
    assert argv[:4] == DOCKER_CMD
    assert argv[argv.index("-i") + 1] == "0"


def test_dcgm_default_index_is_first_gpu_host_index():
    run = fake_run(stdout=PROFILE_TEXT)
    det(nvml=FakeNvml(gpus=[{}, {}]), which=any_which, run=run)
    assert run.argvs[0][-1] == "0"


@pytest.mark.parametrize("exc", [FileNotFoundError, OSError])
def test_run_oserror_is_unavailable(exc):
    caps = det(which=any_which, run=fake_run(raises=exc))
    assert caps.dcgm.available is False and caps.dcgm.reason


def test_run_timeout_is_unavailable():
    exc = subprocess.TimeoutExpired(cmd="dcgmi", timeout=5)
    caps = det(which=any_which, run=fake_run(raises=exc))
    assert caps.dcgm.available is False and "timed out" in caps.dcgm.reason


@pytest.mark.parametrize(
    "out, needle",
    [
        ("Error: unable to find GPU id 9", "GPU id 9"),
        (
            "Error response from daemon: No such container: gso-dcgm",
            "No such container",
        ),
    ],
)
def test_nonzero_exit_not_host_engine(out, needle):
    caps = det(which=any_which, run=fake_run(returncode=1, stderr=out))
    reason = caps.dcgm.reason
    assert "exited with code 1" in reason and needle in reason
    assert "host engine" not in reason
    assert "--dcgm-gpu-index" in reason


def test_connection_failure_is_host_engine():
    caps = det(
        which=any_which,
        run=fake_run(returncode=1, stdout="Error: unable to establish a connection"),
    )
    assert "host engine" in caps.dcgm.reason


def test_zero_exit_no_fields_quotes_output():
    caps = det(which=any_which, run=fake_run(stdout="\nsomething odd happened\n"))
    assert "something odd happened" in caps.dcgm.reason


def test_zero_exit_but_no_fields_is_unavailable():
    caps = det(which=any_which, run=fake_run(stdout="garbage"))
    assert caps.dcgm.available is False


# --- NVML ---------------------------------------------------------------
def test_gpus_enumerated():
    caps = det(nvml=FakeNvml(gpus=[{}, {"name": "X"}], driver="550.1"))
    assert caps.gpus == [
        GpuInfo(0, "GPU-test-0000", "NVIDIA RTX A5000", "550.1", False),
        GpuInfo(1, "GPU-test-0001", "X", "550.1", False),
    ]
    assert caps.nvml_error is None


def test_bytes_normalised():
    caps = det(nvml=FakeNvml(as_bytes=True))
    g = caps.gpus[0]
    assert (g.name, g.uuid, g.driver) == (
        "NVIDIA RTX A5000",
        "GPU-test-0000",
        "535.00.00",
    )


def test_mig_enabled_and_not_supported():
    assert det(nvml=FakeNvml(gpus=[{"mig": 1}])).gpus[0].mig_enabled is True
    f = FakeNvml(errors={"nvmlDeviceGetMigMode": FakeNvml.NVMLError_NotSupported})
    assert det(nvml=f).gpus[0].mig_enabled is False


def test_nvml_init_failure():
    f = FakeNvml(errors={"nvmlInit": FakeNvml.NVMLError("no driver")})
    caps = det(nvml=f)
    assert caps.gpus == [] and caps.nvml_error


def test_init_shutdown_paired():
    f = FakeNvml()
    det(nvml=f)
    assert f.calls[0] == "nvmlInit" and f.calls[-1] == "nvmlShutdown"
    f2 = FakeNvml(errors={"nvmlDeviceGetCount": FakeNvml.NVMLError("x")})
    caps = det(nvml=f2)
    assert f2.calls[-1] == "nvmlShutdown" and caps.gpus == []


# --- MPS / root / nsys --------------------------------------------------
def test_mps_via_listing():
    assert det(mps_listing=lambda: True).mps_active is True


def test_mps_via_nvml():
    assert det(nvml=FakeNvml(mps_procs={0: [1]})).mps_active is True


def test_mps_false_when_both_empty():
    assert det().mps_active is False


def test_mps_listing_error_is_false():
    def boom():
        raise OSError("x")

    assert det(mps_listing=boom).mps_active is False


def test_is_root_injected():
    assert det(is_root=lambda: True).is_root is True
    assert det(is_root=lambda: False).is_root is False


def test_nsys_resolution_order(tmp_path):
    exe = tmp_path / "nsys_env"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    which = lambda n: "/usr/bin/nsys" if n == "nsys" else None
    globbed = lambda _: ["/o/1/bin/nsys", "/o/2/bin/nsys"]
    # env var wins
    assert det(environ={"NSYS": str(exe)}, which=which, glob=globbed).nsys_path == str(
        exe
    )
    # env var not executable -> PATH
    bad = tmp_path / "nope"
    assert (
        det(environ={"NSYS": str(bad)}, which=which, glob=globbed).nsys_path
        == "/usr/bin/nsys"
    )
    # glob (newest = last in sorted order)
    assert det(which=lambda _: None, glob=globbed).nsys_path == "/o/2/bin/nsys"
    # none
    assert det().nsys_path is None


def test_import_does_not_touch_gpu():
    assert callable(capability.detect)


# --- doctor -------------------------------------------------------------
def canned(dcgm_ok=False, nsys=None):
    return Capabilities(
        gpus=[GpuInfo(0, "GPU-test-0000", "NVIDIA RTX A5000", "535.00", False)],
        dcgm=DcgmStatus(
            dcgm_ok,
            "ok" if dcgm_ok else "host engine is not running; start it first",
            {i: dcgm_ok for i in (1001, 1002, 1003, 1004, 1005)},
        ),
        nsys_path=nsys,
        is_root=False,
        mps_active=False,
    )


def test_doctor_cli(monkeypatch, capsys):
    monkeypatch.setattr(cli, "detect", lambda **kw: canned())
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "NVIDIA RTX A5000" in out
    assert "host engine is not running" in out
    assert "N/A" in out
    assert "1001" not in out


def test_doctor_cli_available_shows_fields_and_options(monkeypatch, capsys):
    seen = {}

    def fake_detect(**kw):
        seen.update(kw)
        return canned(dcgm_ok=True, nsys="/opt/nsys/bin/nsys")

    monkeypatch.setattr(cli, "detect", fake_detect)
    rc = cli.main(
        ["doctor", "--dcgmi-cmd", "docker exec gso-dcgm dcgmi", "--dcgm-gpu-index", "0"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert seen["dcgmi_cmd"] == DOCKER_CMD and seen["dcgm_gpu_index"] == 0
    assert "1004" in out and "/opt/nsys/bin/nsys" in out


# --- 기본 MPS listing (autouse 스텁을 우회) ---------------------------------
REAL_MPS_LISTING = capability._default_mps_listing


@pytest.mark.parametrize("rc, expected", [(0, True), (1, False)])
def test_default_mps_listing_pgrep(monkeypatch, rc, expected):
    run = fake_run(returncode=rc)
    monkeypatch.setattr(capability.subprocess, "run", run)
    assert REAL_MPS_LISTING() is expected
    # 15자로 잘린 comm 이름에 대한 정확 일치여야 한다(전체 이름은 매칭되지 않는다).
    assert run.argvs == [["pgrep", "-x", "nvidia-cuda-mps"]]


def test_default_mps_listing_oserror_is_false_via_detect(monkeypatch):
    monkeypatch.setattr(capability.subprocess, "run", fake_run(raises=OSError))
    monkeypatch.setattr(capability, "_default_mps_listing", REAL_MPS_LISTING)
    assert det(mps_listing=None).mps_active is False


# --- 장치별 NVML 오류 --------------------------------------------------------
def test_per_device_error_keeps_others():
    f = FakeNvml(
        gpus=[{}, {}, {}],
        mps_procs={2: [1]},
        errors={"nvmlDeviceGetUUID": {1: FakeNvml.NVMLError("boom")}},
    )
    caps = det(nvml=f)
    assert [g.index for g in caps.gpus] == [0, 2]
    assert "1" in caps.nvml_error and "nvmlDeviceGetUUID" in caps.nvml_error
    assert caps.mps_active is True
    assert f.calls.count("nvmlShutdown") == 1


def test_per_device_non_nvml_exception_skips():
    f = FakeNvml(
        gpus=[{}, {}],
        errors={"nvmlDeviceGetName": {0: RuntimeError("weird")}},
    )
    caps = det(nvml=f)
    assert [g.index for g in caps.gpus] == [1]
    assert caps.nvml_error


def test_nvml_error_none_when_all_ok():
    assert det(nvml=FakeNvml(gpus=[{}, {}])).nvml_error is None

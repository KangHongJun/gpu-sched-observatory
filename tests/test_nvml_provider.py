import math

import pytest

from gso.providers.nvml import NvmlProvider, proc_identity
from gso.record import Quality, Scope
from tests.fakes import FakeNvml

NOW = 1_000_000_000
DEVICE_METRICS = [
    "gpu.util",
    "gpu.mem_used",
    "gpu.power",
    "gpu.temp",
    "gpu.enc_util",
    "gpu.dec_util",
]


def sample(fake, now=NOW, gpu=0):
    with NvmlProvider(gpu, nvml=fake) as p:
        return p.sample(now)


def dev(recs):
    return {r.metric: r for r in recs if r.scope is Scope.DEVICE}


def proc(recs, metric):
    return {r.pid: r for r in recs if r.metric == metric}


def test_device_metrics_measured_in_order_with_units():
    f = FakeNvml(
        util=(37, 5), mem_used=3 * 1048576, power_mw=150500, temp=61, enc=7, dec=9
    )
    recs = sample(f)
    assert [r.metric for r in recs] == DEVICE_METRICS
    assert all(r.quality is Quality.MEASURED for r in recs)
    assert [r.value for r in recs] == [37, 3.0, 150.5, 61, 7, 9]
    assert [r.unit for r in recs] == ["%", "MiB", "W", "C", "%", "%"]


def test_every_record_carries_ts_uuid_source():
    f = FakeNvml(procs={0: [(10, 1048576)]}, proc_util={0: [(10, 5, 1)]})
    recs = sample(f, now=42)
    assert len(recs) == 9
    assert {(r.ts_ns, r.gpu_uuid, r.source) for r in recs} == {
        (42, "GPU-test-0000", "nvml")
    }


def test_uses_given_gpu_index():
    f = FakeNvml(gpus=[{}, {}], util={0: (1, 0), 1: (88, 0)})
    recs = sample(f, gpu=1)
    assert recs[0].value == 88 and recs[0].gpu_uuid == "GPU-test-0001"


def test_enc_not_supported_is_unsupported_others_measured():
    f = FakeNvml(
        errors={"nvmlDeviceGetEncoderUtilization": FakeNvml.NVMLError_NotSupported}
    )
    d = dev(sample(f))
    assert d["gpu.enc_util"].value is None
    assert d["gpu.enc_util"].quality is Quality.UNSUPPORTED
    assert all(
        d[m].quality is Quality.MEASURED for m in DEVICE_METRICS if m != "gpu.enc_util"
    )


def test_other_nvml_error_is_dropped_only_for_that_metric():
    f = FakeNvml(errors={"nvmlDeviceGetPowerUsage": FakeNvml.NVMLError})
    d = dev(sample(f))
    assert d["gpu.power"].value is None and d["gpu.power"].quality is Quality.DROPPED
    assert sum(r.quality is Quality.MEASURED for r in d.values()) == 5


def test_per_device_error_form():
    f = FakeNvml(
        gpus=[{}, {}],
        errors={"nvmlDeviceGetTemperature": {1: FakeNvml.NVMLError_NotSupported}},
    )
    assert dev(sample(f, gpu=0))["gpu.temp"].quality is Quality.MEASURED
    assert dev(sample(f, gpu=1))["gpu.temp"].quality is Quality.UNSUPPORTED


@pytest.mark.parametrize("bad", [math.nan, math.inf, "n/a"])
def test_non_finite_or_non_numeric_is_dropped(bad):
    d = dev(sample(FakeNvml(temp=bad)))
    assert d["gpu.temp"].value is None and d["gpu.temp"].quality is Quality.DROPPED
    assert d["gpu.util"].quality is Quality.MEASURED


def test_process_metrics_and_mps_flag():
    f = FakeNvml(
        procs={0: [(10, 2 * 1048576), (11, 4 * 1048576)]},
        mps_procs={0: [(11, 4 * 1048576)]},
        proc_util={0: [(10, 30, 5), (11, 60, 5)]},
    )
    recs = sample(f)
    assert all(r.scope is Scope.PROCESS for r in recs if r.pid is not None)
    assert {p: r.value for p, r in proc(recs, "proc.mem_used").items()} == {
        10: 2.0,
        11: 4.0,
    }
    assert {p: r.value for p, r in proc(recs, "proc.is_mps").items()} == {
        10: 0.0,
        11: 1.0,
    }
    assert proc(recs, "proc.is_mps")[10].unit == ""
    assert {p: r.value for p, r in proc(recs, "proc.sm_util").items()} == {
        10: 30,
        11: 60,
    }
    assert len(recs) == 6 + 6


def test_mps_only_process_listed_and_deduplicated():
    f = FakeNvml(
        procs={0: [(10, 1048576)]},
        mps_procs={0: [(10, 1048576), (12, 1048576)]},
        proc_util={0: [(10, 1, 1), (12, 1, 1)]},
    )
    recs = sample(f)
    assert sorted(proc(recs, "proc.mem_used")) == [10, 12]
    assert len([r for r in recs if r.metric == "proc.mem_used"]) == 2
    assert {p: r.value for p, r in proc(recs, "proc.is_mps").items()} == {
        10: 1.0,
        12: 1.0,
    }


def test_vanished_process_skipped_entirely():
    f = FakeNvml(
        procs={0: [(10, 1048576), (11, 1048576)]},
        proc_util={0: [(10, 5, 1), (11, 5, 1)]},
    )
    # 프로세스 11의 메모리 조회가 NotFound: 레코드 객체의 속성 접근이 던지도록 만든다.
    orig = f.nvmlDeviceGetComputeRunningProcesses

    class Gone:
        pid = 11

        @property
        def usedGpuMemory(self):
            raise FakeNvml.NVMLError_NotFound("gone")

    f.nvmlDeviceGetComputeRunningProcesses = lambda h: [
        p if p.pid != 11 else Gone() for p in orig(h)
    ]
    recs = sample(f)
    pids = {r.pid for r in recs if r.pid is not None}
    assert pids == {10}
    assert len(recs) == 6 + 3


def test_sm_util_picks_most_recent_sample_per_pid():
    f = FakeNvml(
        procs={0: [(10, 1048576)]},
        proc_util={0: [(10, 11, 100), (10, 77, 300), (10, 22, 200)]},
    )
    assert proc(sample(f), "proc.sm_util")[10].value == 77


def test_listed_pid_without_sample_is_dropped_not_zero():
    f = FakeNvml(procs={0: [(10, 1048576), (11, 1048576)]}, proc_util={0: [(10, 5, 1)]})
    r = proc(sample(f), "proc.sm_util")
    assert r[10].quality is Quality.MEASURED
    assert r[11].value is None and r[11].quality is Quality.DROPPED


def test_util_notfound_drops_all_but_mem_still_reported():
    f = FakeNvml(
        procs={0: [(10, 1048576), (11, 2097152)]}
    )  # proc_util 비어 있음 -> NotFound
    recs = sample(f)
    sm = proc(recs, "proc.sm_util")
    assert {r.quality for r in sm.values()} == {Quality.DROPPED}
    assert sorted(sm) == [10, 11]
    assert {p: r.value for p, r in proc(recs, "proc.mem_used").items()} == {
        10: 1.0,
        11: 2.0,
    }


def test_util_not_supported_is_unsupported_for_all():
    f = FakeNvml(
        procs={0: [(10, 1048576)]},
        errors={"nvmlDeviceGetProcessUtilization": FakeNvml.NVMLError_NotSupported},
    )
    r = proc(sample(f), "proc.sm_util")[10]
    assert r.value is None and r.quality is Quality.UNSUPPORTED


def test_unknown_used_memory_is_unsupported():
    f = FakeNvml(procs={0: [10]}, proc_util={0: [(10, 5, 1)]})
    r = proc(sample(f), "proc.mem_used")[10]
    assert r.value is None and r.quality is Quality.UNSUPPORTED


@pytest.mark.parametrize("err", [FakeNvml.NVMLError_NotSupported, FakeNvml.NVMLError])
def test_process_list_failure_gives_device_records_only(err):
    f = FakeNvml(procs={0: [10]}, errors={"nvmlDeviceGetComputeRunningProcesses": err})
    recs = sample(f)
    assert [r.metric for r in recs] == DEVICE_METRICS


def test_mps_list_failure_keeps_compute_processes():
    f = FakeNvml(
        procs={0: [(10, 1048576)]},
        proc_util={0: [(10, 5, 1)]},
        errors={
            "nvmlDeviceGetMPSComputeRunningProcesses": FakeNvml.NVMLError_NotSupported
        },
    )
    assert proc(sample(f), "proc.is_mps")[10].value == 0.0


def test_sample_before_open_raises():
    with pytest.raises(RuntimeError, match="open"):
        NvmlProvider(0, nvml=FakeNvml()).sample(NOW)


def test_constructor_does_not_touch_nvml():
    f = FakeNvml()
    NvmlProvider(0, nvml=f)
    assert f.calls == []


def test_context_manager_inits_and_shuts_down_once():
    f = FakeNvml()
    with NvmlProvider(0, nvml=f) as p:
        p.sample(NOW)
        p.sample(NOW + 1)
    assert f.calls.count("nvmlInit") == 1
    assert f.calls.count("nvmlShutdown") == 1
    assert f.calls[-1] == "nvmlShutdown"


def test_last_seen_timestamp_passed_on_second_call():
    f = FakeNvml(
        procs={0: [(10, 1048576)]}, proc_util={0: [(10, 5, 111), (10, 6, 222)]}
    )
    with NvmlProvider(0, nvml=f) as p:
        p.sample(NOW)
        p.sample(NOW + 1)
    assert f.proc_util_args == [0, 222]


def test_bytes_uuid_normalised():
    recs = sample(FakeNvml(as_bytes=True))
    assert recs[0].gpu_uuid == "GPU-test-0000"


def test_proc_identity_parses_after_last_paren(tmp_path):
    d = tmp_path / "4242"
    d.mkdir()
    fields_after = ["S", "1", "2", "3", "4", "5"] + ["0"] * 13 + ["987654"] + ["0"] * 5
    (d / "stat").write_text("4242 (my (weird) cmd) " + " ".join(fields_after) + "\n")
    # field 3 = "S"; starttime is field 22 -> index 19 after the ")"
    assert proc_identity(4242, proc_root=str(tmp_path)) == (4242, 987654)


def test_proc_identity_missing_pid(tmp_path):
    assert proc_identity(99999, proc_root=str(tmp_path)) is None

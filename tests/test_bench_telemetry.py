import time

from gso.bench.telemetry import NvmlDcgmTelemetry, mean_in_window
from gso.record import Quality, Record, Scope, unsupported
from tests.fakes import FakeNvml


def _rec(ts, metric, v, quality=Quality.MEASURED):
    return Record(ts, "u", metric, v, "W", "t", Scope.DEVICE, quality)


def test_mean_in_window_uses_only_measured_samples_inside_the_window():
    recs = [_rec(5, "m", 100.0), _rec(10, "m", 2.0), _rec(15, "m", 4.0), _rec(20, "m", 6.0),
            _rec(25, "m", 999.0), _rec(12, "other", 77.0),
            _rec(13, "m", None, Quality.DROPPED), unsupported(14, "u", "m", "W", "t", Scope.DEVICE)]
    assert mean_in_window(recs, "m", 10, 20) == 4.0


def test_mean_in_window_without_samples_is_none_not_zero():
    assert mean_in_window([], "m", 0, 100) is None
    assert mean_in_window([_rec(500, "m", 1.0)], "m", 0, 100) is None


def test_samples_power_every_interval_and_probes_temperature_and_util():
    t = NvmlDcgmTelemetry(0, nvml=FakeNvml(power_mw=150000, temp=61, util=(42, 7)), interval_s=0.01)
    t.start()
    try:
        assert t.probe() == (61.0, 42.0)
        time.sleep(0.15)
    finally:
        t.stop()
    means = t.window_means(0, 2**62)
    assert means["power_w_mean"] == 150.0
    # DCGM not configured: unknown, never 0
    for k in ("sm_active_mean", "sm_occupancy_mean", "tensor_active_mean", "dram_active_mean"):
        assert means[k] is None


def test_window_outside_samples_gives_none():
    t = NvmlDcgmTelemetry(0, nvml=FakeNvml(), interval_s=0.01)
    t.start()
    time.sleep(0.05)
    t.stop()
    assert t.window_means(0, 1000)["power_w_mean"] is None


def test_nvml_failure_degrades_to_none():
    t = NvmlDcgmTelemetry(0, nvml=FakeNvml(errors={"nvmlInit": RuntimeError("no driver")}))
    t.start()
    assert t.probe() == (None, None)
    assert t.window_means(0, 2**62)["power_w_mean"] is None
    assert any("NVML" in e for e in t.errors)
    t.stop()


def test_dcgm_not_started_when_no_field_is_supported():
    popens = []
    t = NvmlDcgmTelemetry(0, dcgm_fields={1002: False}, nvml=FakeNvml(), interval_s=0.01,
                          popen=lambda *a, **k: popens.append(a))
    t.start()
    t.stop()
    assert popens == []


def test_sm_clock_mean_and_static_info():
    t = NvmlDcgmTelemetry(0, nvml=FakeNvml(sm_clock=1700, power_limit_mw=230000, pstate=2),
                          interval_s=0.01)
    t.start()
    try:
        time.sleep(0.12)
    finally:
        t.stop()
    assert t.window_means(0, 2**62)["sm_clock_mhz_mean"] == 1700.0
    assert t.static_info() == {"power_limit_w": 230.0, "perf_state": "P2"}


def test_static_info_none_when_unsupported():
    nv = FakeNvml(errors={"nvmlDeviceGetEnforcedPowerLimit": FakeNvml.NVMLError_NotSupported,
                          "nvmlDeviceGetPerformanceState": FakeNvml.NVMLError_NotSupported,
                          "nvmlDeviceGetClockInfo": FakeNvml.NVMLError_NotSupported})
    t = NvmlDcgmTelemetry(0, nvml=nv, interval_s=0.01)
    t.start()
    time.sleep(0.05)
    t.stop()
    assert t.static_info() == {"power_limit_w": None, "perf_state": None}
    assert t.window_means(0, 2**62)["sm_clock_mhz_mean"] is None


def test_foreign_processes_are_counted_never_listed():
    nv = FakeNvml(procs={0: [111, 222]}, mps_procs={0: [333]})
    t = NvmlDcgmTelemetry(0, nvml=nv, interval_s=0.01)
    t.set_own_filter(lambda pid: pid == 111)
    t.start()
    time.sleep(0.12)
    t.stop()
    assert t.foreign_in_window(0, 2**62) == 2  # 222 and 333
    assert t.foreign_in_window(0, 1000) == 0  # outside the window
    assert not any(isinstance(v, int) and v in (222, 333) for v in vars(t).values())


def test_no_filter_means_no_foreign_check():
    t = NvmlDcgmTelemetry(0, nvml=FakeNvml(procs={0: [5]}), interval_s=0.01)
    t.start()
    time.sleep(0.05)
    t.stop()
    assert t.foreign_in_window(0, 2**62) == 0

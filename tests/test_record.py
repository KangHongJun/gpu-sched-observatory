import pytest

from gso.record import (
    Quality,
    Record,
    Scope,
    format_value,
    from_json,
    to_json,
    unsupported,
)


def test_unsupported_renders_na_not_zero():
    r = unsupported(1, "GPU-x", "dcgm.tensor_active", "%", "dcgm", Scope.DEVICE)
    assert r.value is None and r.quality is Quality.UNSUPPORTED
    assert format_value(r) == "N/A"


def test_value_none_requires_non_measured_quality():
    with pytest.raises(ValueError):
        Record(1, "GPU-x", "m", None, "%", "nvml", Scope.DEVICE, Quality.MEASURED)


def test_json_roundtrip():
    r = Record(5, "GPU-x", "gpu.util", 78.0, "%", "nvml", Scope.PROCESS, Quality.MEASURED, pid=1421)
    assert from_json(to_json(r)) == r


def test_unsupported_with_value_raises():
    with pytest.raises(ValueError):
        Record(1, "GPU-x", "m", 1.0, "%", "nvml", Scope.DEVICE, Quality.UNSUPPORTED)


def test_dropped_without_value_is_allowed_and_formats_na():
    r = Record(1, "GPU-x", "m", None, "%", "nvml", Scope.DEVICE, Quality.DROPPED)
    assert format_value(r) == "N/A"


def test_format_value_digits():
    r = Record(1, "GPU-x", "m", 12.34, "%", "nvml", Scope.DEVICE, Quality.MEASURED)
    assert format_value(r, digits=1) == "12.3"
    assert format_value(r) == "12"


def test_json_roundtrip_all_optional_none():
    r = Record(1, "GPU-x", "m", 3.0, "%", "nvml", Scope.DEVICE, Quality.ESTIMATED)
    assert r.pid is None and r.context_id is None and r.stream_id is None
    assert from_json(to_json(r)) == r


def test_json_roundtrip_unsupported():
    r = unsupported(7, "GPU-x", "dcgm.tensor_active", "%", "dcgm", Scope.DEVICE)
    assert from_json(to_json(r)) == r


def test_to_json_is_single_line():
    r = Record(5, "GPU-x", "gpu.util", 78.0, "%", "nvml", Scope.PROCESS, Quality.MEASURED, pid=1421)
    assert "\n" not in to_json(r)

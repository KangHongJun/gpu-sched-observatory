import json
import math
import sqlite3
from pathlib import Path

import pytest

from gso.trace.report import build_report, build_summary

FIX = Path(__file__).parent / "fixtures"
GLOBAL = FIX / "nsys_ctxsw_min.sqlite"
LAUNCHED = FIX / "nsys_launched_min.sqlite"
KEYS = {
    "schema_version", "mode", "gpu", "masked", "window_s", "ctx", "kernel",
    "interprocess_overlap", "dropped_hints", "raw_counts", "notes",
}
PCT = {"n", "p50_ms", "p95_ms", "p99_ms"}


def g(gpu=1):
    return build_summary(GLOBAL, gpu=gpu, mode="global", masked=True)


def test_key_set_is_exact():
    s = g()
    assert set(s) == KEYS
    assert set(s["ctx"]) == {
        "switch_rate_per_s", "switch_rate_is_lower_bound", "masked_restore_events", "per_context"
    }
    assert set(s["kernel"]) == {
        "count", "duration_ms", "launch_to_start_ms", "ambiguous_api_matches", "pids"
    }
    assert s["schema_version"] == 1


def test_global_fixture_gpu1():
    s = g()
    assert s["mode"] == "global" and s["gpu"] == 1 and s["masked"] is True
    assert s["window_s"] > 0
    assert isinstance(s["ctx"]["switch_rate_per_s"], float)
    assert s["ctx"]["switch_rate_is_lower_bound"] is False
    pcs = s["ctx"]["per_context"]
    assert len(pcs) == 2
    for pc in pcs:
        assert pc["restores_per_s"] > 0
        rs, hold = pc["restore_save"], pc["hold"]
        assert 0 <= rs["share"] <= 1 and 0 <= hold["share"] <= 1
        assert rs["share"] < hold["share"]
        assert set(rs["slice_ms"]) == PCT
    assert s["kernel"]["count"] == 0
    assert s["kernel"]["duration_ms"]["p50_ms"] is None
    assert s["kernel"]["launch_to_start_ms"]["p50_ms"] is None
    assert s["interprocess_overlap"] is None
    joined = " ".join(s["notes"])
    assert "restore_save" in joined and "hold" in joined
    assert "SM" in joined  # overlap is not simultaneous SM execution
    assert "마스킹" in joined
    assert any("커널" in n and "외부" in n for n in s["notes"])


def test_global_fixture_masked_only_gpu0():
    s = g(0)
    assert s["ctx"]["switch_rate_per_s"] is None
    assert s["ctx"]["per_context"] == []
    assert s["ctx"]["masked_restore_events"] > 0


def test_launched_fixture():
    s = build_summary(LAUNCHED, gpu=1, mode="launched", masked=True)
    assert s["kernel"]["count"] == 2000
    assert isinstance(s["kernel"]["launch_to_start_ms"]["p50_ms"], float)
    assert isinstance(s["kernel"]["duration_ms"]["p50_ms"], float)
    assert len(s["kernel"]["pids"]) == 1
    assert s["interprocess_overlap"] is None
    assert any("실행한 프로세스" in n for n in s["notes"])


def test_launched_other_gpu_excludes_kernels():
    s = build_summary(LAUNCHED, gpu=0, mode="launched", masked=True)
    assert s["kernel"]["count"] == 0
    assert s["kernel"]["pids"] == []


def test_empty_sqlite_is_all_null(tmp_path):
    p = tmp_path / "e.sqlite"
    sqlite3.connect(p).close()
    s = build_summary(p, gpu=1, mode="global", masked=True)
    assert s["ctx"]["switch_rate_per_s"] is None
    assert s["ctx"]["per_context"] == []
    assert s["window_s"] is None
    assert s["interprocess_overlap"] is None
    assert s["kernel"]["duration_ms"]["p50_ms"] is None
    json.dumps(s, allow_nan=False)


@pytest.mark.parametrize("args", [(GLOBAL, 1, "global"), (GLOBAL, 0, "global"),
                                  (LAUNCHED, 1, "launched")])
def test_json_clean(args):
    path, gpu, mode = args
    s = build_summary(path, gpu=gpu, mode=mode, masked=False)
    text = json.dumps(s, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    assert not any(isinstance(v, float) and not math.isfinite(v) for v in _walk(s))
    assert "real-time" not in text and "실시간" not in text


def _walk(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)
    else:
        yield o


def test_lower_bound_note_when_masked_rows_mix(monkeypatch):
    import gso.trace.report as r

    monkeypatch.setattr(r, "switch_rate_is_lower_bound_by_gpu", lambda *a, **k: {1: True})
    s = g()
    assert s["ctx"]["switch_rate_is_lower_bound"] is True
    assert any("하한" in n for n in s["notes"])


def test_timeline_text_global():
    _, tl = build_report(GLOBAL, gpu=1, mode="global", masked=True)
    lines = tl.splitlines()
    assert "GPU 1" in lines[0] and "global" in lines[0] and "on-demand capture" in lines[0]
    assert "[restore_save]" in tl and "[hold]" in tl
    assert max(len(ln) for ln in lines) <= 120
    assert "real-time" not in tl and "실시간" not in tl
    rows = [ln for ln in lines if "|" in ln and "=" in ln]
    assert len(rows) >= 4  # two contexts x two constructions


def test_timeline_text_launched_has_kernel_block():
    _, tl = build_report(LAUNCHED, gpu=1, mode="launched", masked=True)
    assert "[kernels]" in tl and "[hold]" not in tl


def test_timeline_note_when_nothing_to_draw(tmp_path):
    p = tmp_path / "e.sqlite"
    sqlite3.connect(p).close()
    _, tl = build_report(p, gpu=1, mode="global", masked=True)
    assert "N/A" in tl

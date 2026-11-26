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
    k = s["kernel"]  # a global capture measures no kernels: unknown, not zero
    assert k["count"] is None and k["ambiguous_api_matches"] is None and k["pids"] is None
    assert k["duration_ms"]["n"] is None and k["duration_ms"]["p50_ms"] is None
    assert k["launch_to_start_ms"]["p50_ms"] is None
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


def test_launched_without_kernel_table_is_null(tmp_path):
    p = tmp_path / "e.sqlite"
    sqlite3.connect(p).close()
    k = build_summary(p, gpu=1, mode="launched", masked=True)["kernel"]
    assert k["count"] is None and k["pids"] is None and k["ambiguous_api_matches"] is None


def _zoom_rows(tl, name):
    lines = tl.splitlines()
    i = next(n for n, ln in enumerate(lines) if ln.startswith(f"[{name} zoom]"))
    rows = []
    for ln in lines[i + 1 :]:
        if ln.startswith("[") or not ln.strip():
            break
        if "|" in ln:
            rows.append(ln.split("|")[1])
    return rows


def test_zoomed_blocks_alternate_on_global_fixture():
    _, tl = build_report(GLOBAL, gpu=1, mode="global", masked=True)
    assert "zoom: +" in tl and "1 cell = 50 µs" in tl
    assert tl.index("[restore_save zoom]") < tl.index("[restore_save]")  # zoom first
    rows = _zoom_rows(tl, "restore_save")
    assert len(rows) == 2 and all(len(r) == 100 for r in rows)
    for r in rows:
        assert "=" in r and " " in r
    assert rows[0] != rows[1]
    assert _zoom_rows(tl, "hold")


def test_zoomed_kernel_block_in_launched_mode():
    _, tl = build_report(LAUNCHED, gpu=1, mode="launched", masked=True)
    (row,) = _zoom_rows(tl, "kernels")
    assert "=" in row


def test_masked_only_gpu_timeline_says_masked():
    _, tl = build_report(GLOBAL, gpu=0, mode="global", masked=True)
    assert "masked" in tl and "no intervals" not in tl


def test_mixed_masked_header_line(monkeypatch):
    import gso.trace.report as r

    real = r.load_ctx_events

    def with_masked(path, gpu_id=None):
        ev = real(path, gpu_id=gpu_id)
        from dataclasses import replace

        first = next(e for e in ev if e.tag == 8)
        masked = [replace(first, pid=None, global_pid=1 << 48, ts_ns=first.ts_ns + 1)]
        return ev + masked

    monkeypatch.setattr(r, "load_ctx_events", with_masked)
    s, tl = build_report(GLOBAL, gpu=1, mode="global", masked=True)
    n = s["ctx"]["masked_restore_events"]
    assert n >= 1
    assert f"masked contexts not shown: {n} restore events" in tl

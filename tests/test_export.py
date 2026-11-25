import csv
import os
import threading

import pytest

from gso.buffer import RingBuffer
from gso.export import (
    CSV_COLUMNS,
    CsvStream,
    Drainer,
    JsonlStream,
    read_jsonl,
    write_csv,
    write_jsonl,
    write_records,
)
from gso.record import Quality, Record, Scope, unsupported


def rec(ts=1, metric="gpu.util", value=50.0, unit="%", pid=None, ctx=None, stream=None,
        quality=Quality.MEASURED, scope=Scope.DEVICE):
    return Record(ts, "GPU-test-0001", metric, value, unit, "nvml", scope, quality,
                  pid=pid, context_id=ctx, stream_id=stream)


SAMPLE = [
    rec(1, value=12.5),
    rec(1, "proc.mem_used", 1024.0, "MiB", pid=42, scope=Scope.PROCESS),
    unsupported(2, "GPU-test-0001", "dcgm.sm_active", "%", "dcgm", Scope.DEVICE),
    rec(3, "k.dur", 7.0, "us", pid=1, ctx=2, stream=3, quality=Quality.ESTIMATED,
        scope=Scope.CONTEXT),
]


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_jsonl_round_trip(tmp_path):
    p = tmp_path / "o.jsonl"
    assert write_jsonl(SAMPLE, p) == len(SAMPLE)
    assert read_jsonl(p) == SAMPLE
    text = p.read_text(encoding="utf-8")
    assert text.endswith("\n") and "\r" not in text and len(text.splitlines()) == len(SAMPLE)


def test_csv_header_order(tmp_path):
    p = tmp_path / "o.csv"
    write_csv(SAMPLE, p)
    assert CSV_COLUMNS == ("ts_ns", "gpu_uuid", "pid", "context_id", "stream_id", "metric",
                           "value", "unit", "source", "scope", "quality")
    assert p.read_text(encoding="utf-8").splitlines()[0] == ",".join(CSV_COLUMNS)


def test_csv_unsupported_has_empty_value(tmp_path):
    p = tmp_path / "o.csv"
    write_csv([SAMPLE[2]], p)
    (row,) = read_csv_rows(p)
    assert row["value"] == "" and row["quality"] == "unsupported"
    assert row["scope"] == "device" and row["pid"] == ""


def test_csv_none_ids_are_empty_and_ids_written(tmp_path):
    p = tmp_path / "o.csv"
    write_csv([SAMPLE[0], SAMPLE[3]], p)
    a, b = read_csv_rows(p)
    assert (a["pid"], a["context_id"], a["stream_id"]) == ("", "", "")
    assert (b["pid"], b["context_id"], b["stream_id"]) == ("1", "2", "3")


def test_csv_float_round_trips_exactly(tmp_path):
    p = tmp_path / "o.csv"
    vals = [0.1 + 0.2, 1 / 3, 1e-9, 123456789.123456789, 5.0]
    write_csv([rec(i, value=v) for i, v in enumerate(vals)], p)
    assert [float(r["value"]) for r in read_csv_rows(p)] == vals


def test_csv_quotes_commas_and_quotes(tmp_path):
    p = tmp_path / "o.csv"
    write_csv([rec(metric='a,"b"', unit="x,y")], p)
    (row,) = read_csv_rows(p)
    assert row["metric"] == 'a,"b"' and row["unit"] == "x,y"


def test_empty_input(tmp_path):
    j, c = tmp_path / "e.jsonl", tmp_path / "e.csv"
    assert write_jsonl([], j) == 0 and write_csv([], c) == 0
    assert j.read_bytes() == b""
    assert c.read_text(encoding="utf-8") == ",".join(CSV_COLUMNS) + "\n"


class Boom:
    """Looks enough like a Record for the first access, then raises during serialisation."""

    def __getattr__(self, name):
        raise RuntimeError("boom")


@pytest.mark.parametrize("fn,name", [(write_jsonl, "o.jsonl"), (write_csv, "o.csv")])
def test_atomic_replace_keeps_old_file_on_failure(tmp_path, fn, name):
    p = tmp_path / name
    p.write_text("OLD\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        fn([SAMPLE[0], Boom()], p)
    assert p.read_text(encoding="utf-8") == "OLD\n"
    assert os.listdir(tmp_path) == [name]


@pytest.mark.parametrize("fn,name", [(write_jsonl, "o.jsonl"), (write_csv, "o.csv")])
def test_atomic_failure_creates_no_file(tmp_path, fn, name):
    with pytest.raises(RuntimeError):
        fn([Boom()], tmp_path / name)
    assert os.listdir(tmp_path) == []


def test_write_records_dispatch(tmp_path):
    assert write_records(SAMPLE, tmp_path / "a.jsonl") == 4
    assert write_records(SAMPLE, tmp_path / "b.CSV") == 4
    assert write_records(SAMPLE, tmp_path / "c.JsonL") == 4
    assert read_jsonl(tmp_path / "a.jsonl") == SAMPLE
    assert read_csv_rows(tmp_path / "b.CSV")[0]["metric"] == "gpu.util"
    with pytest.raises(ValueError, match=r"\.jsonl.*\.csv"):
        write_records(SAMPLE, tmp_path / "d.txt")
    assert not (tmp_path / "d.txt").exists()


def test_read_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "o.jsonl"
    write_jsonl(SAMPLE[:2], p)
    p.write_text(p.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert read_jsonl(p) == SAMPLE[:2]


# -- streaming ---------------------------------------------------------------

def test_streams_append_and_flush(tmp_path):
    j, c = tmp_path / "s.jsonl", tmp_path / "s.csv"
    sj, sc = JsonlStream(j), CsvStream(c)
    assert sj.write(SAMPLE[:2]) == 2 and sc.write(SAMPLE[:2]) == 2
    # flushed without close: another reader already sees the data
    assert read_jsonl(sj.partial_path) == SAMPLE[:2]
    assert len(read_csv_rows(sc.partial_path)) == 2
    sj.write(SAMPLE[2:])
    sc.write(SAMPLE[2:])
    sj.commit()
    sc.commit()
    sj.commit()  # idempotent
    assert read_jsonl(j) == SAMPLE
    assert [r["metric"] for r in read_csv_rows(c)] == [r.metric for r in SAMPLE]
    assert sj.count == 4 and sc.count == 4


def test_stream_writes_to_partial_until_commit(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("OLD\n", encoding="utf-8")
    s = JsonlStream(p)
    s.write(SAMPLE[:2])
    assert s.partial_path == tmp_path / "s.jsonl.partial"
    assert p.read_text(encoding="utf-8") == "OLD\n"
    assert len(read_jsonl(s.partial_path)) == 2
    s.commit()
    assert read_jsonl(p) == SAMPLE[:2]
    assert not s.partial_path.exists()


def test_stream_stale_partial_is_overwritten(tmp_path):
    (tmp_path / "s.csv.partial").write_text("garbage\n", encoding="utf-8")
    s = CsvStream(tmp_path / "s.csv")
    s.write(SAMPLE[:1])
    s.commit()
    assert len(read_csv_rows(tmp_path / "s.csv")) == 1


@pytest.mark.parametrize("cls,name", [(JsonlStream, "s.jsonl"), (CsvStream, "s.csv")])
def test_empty_stream_commit_keeps_target_and_removes_partial(tmp_path, cls, name):
    p = tmp_path / name
    p.write_text("OLD\n", encoding="utf-8")
    s = cls(p)
    s.commit()
    assert p.read_text(encoding="utf-8") == "OLD\n"
    assert os.listdir(tmp_path) == [name]


def test_stream_close_keeps_data_in_partial_and_target_untouched(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("OLD\n", encoding="utf-8")
    s = JsonlStream(p)
    s.write(SAMPLE[:2])
    s.close()
    assert p.read_text(encoding="utf-8") == "OLD\n"
    assert read_jsonl(s.partial_path) == SAMPLE[:2]


def test_stream_dropped_without_close_leaves_target_and_flushed_partial(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("OLD\n", encoding="utf-8")
    s = JsonlStream(p)
    s.write(SAMPLE[:3])
    partial = s.partial_path
    del s
    assert p.read_text(encoding="utf-8") == "OLD\n"
    assert read_jsonl(partial) == SAMPLE[:3]


def test_empty_stream_close_removes_partial(tmp_path):
    s = JsonlStream(tmp_path / "s.jsonl")
    s.close()
    assert os.listdir(tmp_path) == []


class ListSink:
    """In-memory sink; ``fail_at`` (1-based record number) raises ``exc`` once, after k-1 writes."""

    def __init__(self, fail_at=None, exc=None):
        self.items = []
        self.fail_at = fail_at
        self.exc = exc

    def write(self, records):
        n = 0
        for r in records:
            if self.fail_at is not None and len(self.items) + 1 == self.fail_at:
                self.fail_at = None
                raise self.exc
            self.items.append(r)
            n += 1
        return n


def mkseq(n, ts=10, prefix="m"):
    return [rec(ts, metric=f"{prefix}{i}") for i in range(n)]


def test_drain_cursor_exactly_once_with_shared_timestamps():
    buf = RingBuffer(1000)
    sink = ListSink()
    d = Drainer(buf, sink)
    expected = []
    seq = 0
    for batch_ts in (10, 10, 10, 11, 11, 10, 12):  # repeated and even out-of-order timestamps
        for _ in range(3):
            r = rec(batch_ts, metric=f"m{seq}")
            seq += 1
            buf.push(r)
            expected.append(r)
        d.drain()
    d.drain()  # nothing new: no duplicates
    assert sink.items == expected and len(sink.items) == 21 and d.lost == 0


def test_drain_cursor_two_interleaving_producers():
    buf = RingBuffer(100_000)
    sink = ListSink()
    d = Drainer(buf, sink)
    pushed = {"a": [], "b": []}

    def producer(tag):
        for i in range(3000):
            r = rec(i // 5, metric=f"{tag}{i}")  # five records share each timestamp
            buf.push(r)
            pushed[tag].append(r)

    ts = [threading.Thread(target=producer, args=(t,)) for t in "ab"]
    for t in ts:
        t.start()
    while any(t.is_alive() for t in ts):
        d.drain()
    d.drain()
    assert d.lost == 0
    assert len(sink.items) == 6000 == len({r.metric for r in sink.items})
    for tag in "ab":  # per-producer order preserved
        assert [r for r in sink.items if r.metric.startswith(tag)] == pushed[tag]


def test_drain_reports_dropped_records():
    buf = RingBuffer(4)
    sink = ListSink()
    d = Drainer(buf, sink)
    for r in mkseq(3):
        buf.push(r)
    d.drain()
    assert d.lost == 0
    for r in mkseq(9, prefix="n"):  # 9 pushes, capacity 4: 5 overwritten unread
        buf.push(r)
    d.drain()
    assert d.lost == 5
    assert [r.metric for r in sink.items] == ["m0", "m1", "m2", "n5", "n6", "n7", "n8"]


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), OSError(28, "No space left on device")])
def test_drain_interrupted_mid_write_is_exactly_once(exc):
    buf = RingBuffer(100)
    sink = ListSink(fail_at=4, exc=exc)
    d = Drainer(buf, sink)
    recs = mkseq(10)
    for r in recs:
        buf.push(r)
    with pytest.raises(type(exc)):
        d.drain()
    assert len(sink.items) == 3
    d.drain()  # the handler's final drain
    d.drain()
    assert sink.items == recs and d.lost == 0


def test_drain_interrupted_after_overwrite_counts_lost_once():
    buf = RingBuffer(6)
    sink = ListSink(fail_at=3, exc=KeyboardInterrupt())
    d = Drainer(buf, sink)
    recs = mkseq(10)  # 4 overwritten before the first read
    for r in recs:
        buf.push(r)
    with pytest.raises(KeyboardInterrupt):
        d.drain()
    assert d.lost == 4
    d.drain()
    assert d.lost == 4  # not doubled
    assert sink.items == recs[4:]

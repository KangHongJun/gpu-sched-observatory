import threading

import pytest

from gso.buffer import RingBuffer
from gso.record import Quality, Record, Scope


def rec(ts_ns: int, metric: str = "m") -> Record:
    return Record(
        ts_ns=ts_ns,
        gpu_uuid="GPU-0",
        metric=metric,
        value=1.0,
        unit="u",
        source="test",
        scope=Scope.DEVICE,
        quality=Quality.MEASURED,
    )


def test_bounded_drops_oldest():
    buf = RingBuffer(3)
    for ts in range(1, 6):
        buf.push(rec(ts))
    assert len(buf) == 3
    assert buf.dropped == 2
    assert [r.ts_ns for r in buf.snapshot()] == [3, 4, 5]


def test_snapshot_filters_by_metric_in_insertion_order():
    buf = RingBuffer(10)
    buf.push(rec(1, "a"))
    buf.push(rec(2, "b"))
    buf.push(rec(3, "a"))
    buf.push(rec(4, "c"))
    buf.push(rec(5, "a"))
    got = buf.snapshot(metric="a")
    assert [r.ts_ns for r in got] == [1, 3, 5]
    assert all(r.metric == "a" for r in got)


def test_snapshot_since_ns_filter_and_combined_filters():
    buf = RingBuffer(10)
    buf.push(rec(1, "a"))
    buf.push(rec(2, "b"))
    buf.push(rec(3, "a"))
    buf.push(rec(4, "b"))
    buf.push(rec(5, "a"))
    assert [r.ts_ns for r in buf.snapshot(since_ns=3)] == [3, 4, 5]
    assert [r.ts_ns for r in buf.snapshot(metric="a", since_ns=3)] == [3, 5]


def test_snapshot_is_a_new_list():
    buf = RingBuffer(5)
    for ts in range(1, 4):
        buf.push(rec(ts))
    snap = buf.snapshot()
    snap.clear()
    assert len(buf) == 3
    assert len(buf.snapshot()) == 3


def test_capacity_must_be_positive_int():
    with pytest.raises(ValueError):
        RingBuffer(0)


def test_concurrent_pushes_account_for_every_record():
    buf = RingBuffer(500)
    errors: list[BaseException] = []

    def writer(base: int) -> None:
        try:
            for i in range(1000):
                buf.push(rec(base + i))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(k * 1000,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(buf) + buf.dropped == 4000


def test_concurrent_reader_never_sees_overfull_snapshot():
    buf = RingBuffer(100)
    errors: list[BaseException] = []
    stop = threading.Event()
    lengths: list[int] = []

    def writer(base: int) -> None:
        try:
            for i in range(2000):
                buf.push(rec(base + i))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        try:
            while not stop.is_set():
                lengths.append(len(buf.snapshot()))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    writers = [threading.Thread(target=writer, args=(k * 10000,)) for k in range(2)]
    rd = threading.Thread(target=reader)
    rd.start()
    for t in writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    rd.join()
    assert errors == []
    assert lengths, "reader made no snapshots"
    assert max(lengths) <= 100


def test_read_from_returns_each_record_once_in_order():
    buf = RingBuffer(10)
    for i in range(3):
        buf.push(rec(5, f"m{i}"))  # same timestamp
    recs, nxt, lost = buf.read_from(0)
    assert [r.metric for r in recs] == ["m0", "m1", "m2"] and nxt == 3 and lost == 0
    buf.push(rec(5, "m3"))
    recs, nxt, lost = buf.read_from(nxt)
    assert [r.metric for r in recs] == ["m3"] and nxt == 4 and lost == 0
    assert buf.read_from(nxt) == ([], 4, 0)


def test_read_from_reports_overwritten_records():
    buf = RingBuffer(3)
    for i in range(5):
        buf.push(rec(i, f"m{i}"))
    recs, nxt, lost = buf.read_from(0)
    assert [r.metric for r in recs] == ["m2", "m3", "m4"]
    assert nxt == 5 and lost == 2

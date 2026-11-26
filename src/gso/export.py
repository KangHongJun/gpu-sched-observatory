"""JSONL / CSV export of metric records. Standard library only (no Textual)."""

from __future__ import annotations

import contextlib
import csv
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

from gso.buffer import RingBuffer
from gso.record import Record, from_json, to_json

CSV_COLUMNS = (
    "ts_ns",
    "gpu_uuid",
    "pid",
    "context_id",
    "stream_id",
    "metric",
    "value",
    "unit",
    "source",
    "scope",
    "quality",
)
SUPPORTED_EXTENSIONS = (".jsonl", ".csv")


def _cell(v) -> str:
    return "" if v is None else str(v)


def csv_row(r: Record) -> list[str]:
    return [
        _cell(r.ts_ns),
        r.gpu_uuid,
        _cell(r.pid),
        _cell(r.context_id),
        _cell(r.stream_id),
        r.metric,
        ""
        if r.value is None
        else repr(float(r.value)),  # shortest string that round-trips
        r.unit,
        r.source,
        r.scope.value,
        r.quality.value,
    ]


def _format_of(path) -> str:
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"unsupported export extension {ext or '(none)'!r}; use one of "
            + ", ".join(SUPPORTED_EXTENSIONS)
        )
    return ext[1:]


def check_export_path(path) -> None:
    """Raise ValueError if the format cannot be chosen from the extension."""
    _format_of(path)


def _write_atomic(path, body: Callable[[object], int]) -> int:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            n = body(f)
        mode = path.stat().st_mode & 0o7777 if path.exists() else 0o644
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return n


def write_jsonl(records: Iterable[Record], path) -> int:
    def body(f) -> int:
        n = 0
        for r in records:
            f.write(to_json(r) + "\n")
            n += 1
        return n

    return _write_atomic(path, body)


def write_csv(records: Iterable[Record], path) -> int:
    def body(f) -> int:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(CSV_COLUMNS)
        n = 0
        for r in records:
            w.writerow(csv_row(r))
            n += 1
        return n

    return _write_atomic(path, body)


def write_records(records: Iterable[Record], path) -> int:
    """Write in the format chosen by the extension (.jsonl or .csv, case-insensitive)."""
    fmt = _format_of(path)
    return write_jsonl(records, path) if fmt == "jsonl" else write_csv(records, path)


def read_jsonl(path) -> list[Record]:
    with open(path, encoding="utf-8") as f:
        return [from_json(line) for line in f if line.strip()]


class _Stream:
    """Appending writer, flushed after every ``write``.

    Data goes to ``<path>.partial`` (a stale one is overwritten); the target is untouched until
    ``commit()`` atomically replaces it. ``commit()`` with zero records keeps the target and removes
    the partial. ``close()`` just closes, leaving the partial in place when it holds data (use it
    after an I/O error); a stream dropped without either leaves the flushed data in the partial.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.partial_path = self.path.with_name(self.path.name + ".partial")
        self.count = 0
        self._f = open(self.partial_path, "w", encoding="utf-8", newline="")  # noqa: SIM115
        try:
            self._start()
            self._f.flush()
        except BaseException:
            self._f.close()
            with contextlib.suppress(OSError):
                self.partial_path.unlink()
            raise

    def _start(self) -> None:
        pass

    def _emit(self, r: Record) -> None:
        raise NotImplementedError

    def write(self, records: Iterable[Record]) -> int:
        n = 0
        try:
            for r in records:
                self._emit(r)
                n += 1
        finally:
            self.count += n
            if not self._f.closed:
                with contextlib.suppress(OSError):
                    self._f.flush()
        return n

    def close(self) -> None:
        if not self._f.closed:
            with contextlib.suppress(OSError):
                self._f.close()
        if self.count == 0:
            with contextlib.suppress(OSError):
                self.partial_path.unlink()

    def commit(self) -> None:
        """Finish an orderly run: replace the target with the partial (if anything was written)."""
        if self._f.closed and not self.partial_path.exists():
            return  # already committed or discarded
        self._f.close()
        if self.count == 0:
            with contextlib.suppress(OSError):
                self.partial_path.unlink()
            return
        os.replace(self.partial_path, self.path)


class JsonlStream(_Stream):
    def _emit(self, r: Record) -> None:
        self._f.write(to_json(r) + "\n")


class CsvStream(_Stream):
    def _start(self) -> None:
        self._csv = csv.writer(self._f, lineterminator="\n")
        self._csv.writerow(CSV_COLUMNS)

    def _emit(self, r: Record) -> None:
        self._csv.writerow(csv_row(r))


def open_stream(path) -> _Stream:
    return (JsonlStream if _format_of(path) == "jsonl" else CsvStream)(path)


class Drainer:
    """Copies a ``RingBuffer`` into a sink exactly once and in push order.

    The cursor (a push sequence number) moves one record at a time, only after that record reached
    the sink, so an exception of any kind (Ctrl-C, I/O error) leaves it at the first unwritten
    record. ``lost`` totals records the ring overwrote before they could be read; it is added when
    first observed and never again for a re-read range.
    """

    def __init__(self, buffer: RingBuffer, sink) -> None:
        self._buffer = buffer
        self._sink = sink
        self.cursor = 0
        self.lost = 0

    def drain(self) -> int:
        records, nxt, lost = self._buffer.read_from(self.cursor)
        self.lost += lost
        self.cursor = nxt - len(
            records
        )  # sequence number of records[0]; skips the lost range
        n = 0
        for r in records:
            self._sink.write([r])
            self.cursor += 1
            n += 1
        return n

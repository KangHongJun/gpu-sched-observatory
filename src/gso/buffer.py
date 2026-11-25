"""Bounded, thread-safe ring buffer of metric records."""

from __future__ import annotations

import threading
from collections import deque
from itertools import islice

from gso.record import Record


class RingBuffer:
    """Keeps the newest ``capacity`` records; older ones are discarded and counted."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"capacity must be an int >= 1, got {capacity!r}")
        self._items: deque[Record] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._dropped = 0
        self._total = 0  # records ever pushed; the sequence number of the next record

    def push(self, r: Record) -> None:
        with self._lock:
            if len(self._items) == self._items.maxlen:
                self._dropped += 1
            self._items.append(r)
            self._total += 1

    def snapshot(
        self, metric: str | None = None, since_ns: int | None = None
    ) -> list[Record]:
        with self._lock:
            items = list(self._items)
        return [
            r
            for r in items
            if (metric is None or r.metric == metric)
            and (since_ns is None or r.ts_ns >= since_ns)
        ]

    def read_from(self, seq: int) -> tuple[list[Record], int, int]:
        """Records pushed at sequence number ``seq`` or later, exactly once and in push order.

        Returns ``(records, next_seq, lost)``: pass ``next_seq`` to the next call; ``lost`` is how
        many records in ``[seq, next_seq)`` were overwritten before they could be read. Atomic, so
        records sharing a timestamp or interleaved from several threads are never skipped or
        repeated (a timestamp cursor cannot promise that).
        """
        with self._lock:
            seq = min(max(seq, 0), self._total)
            first = self._total - len(self._items)  # sequence number of the oldest kept record
            start = max(seq, first)
            items = list(islice(self._items, start - first, None))
            return items, self._total, start - seq

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

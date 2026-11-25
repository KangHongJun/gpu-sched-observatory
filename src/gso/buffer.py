"""Bounded, thread-safe ring buffer of metric records."""

from __future__ import annotations

import threading
from collections import deque

from gso.record import Record


class RingBuffer:
    """Keeps the newest ``capacity`` records; older ones are discarded and counted."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"capacity must be an int >= 1, got {capacity!r}")
        self._items: deque[Record] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._dropped = 0

    def push(self, r: Record) -> None:
        with self._lock:
            if len(self._items) == self._items.maxlen:
                self._dropped += 1
            self._items.append(r)

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

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

"""Common metric record shared by all providers, the ring buffer, TUI and exporters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


class Quality(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNSUPPORTED = "unsupported"
    DROPPED = "dropped"


class Scope(str, Enum):
    DEVICE = "device"
    PROCESS = "process"
    CONTEXT = "context"


@dataclass(frozen=True)
class Record:
    ts_ns: int
    gpu_uuid: str
    metric: str
    value: float | None
    unit: str
    source: str
    scope: Scope
    quality: Quality
    pid: int | None = None
    context_id: int | None = None
    stream_id: int | None = None

    def __post_init__(self) -> None:
        if self.value is None:
            if self.quality not in (Quality.UNSUPPORTED, Quality.DROPPED):
                raise ValueError(
                    f"value None requires quality UNSUPPORTED or DROPPED, got {self.quality.value}"
                )
        elif self.quality is Quality.UNSUPPORTED:
            raise ValueError("UNSUPPORTED record must not carry a value")


def unsupported(ts_ns, gpu_uuid, metric, unit, source, scope, pid=None) -> Record:
    return Record(
        ts_ns, gpu_uuid, metric, None, unit, source, scope, Quality.UNSUPPORTED, pid=pid
    )


def format_value(r: Record, digits: int = 0) -> str:
    if r.value is None:
        return "N/A"
    return f"{r.value:.{digits}f}"


def to_json(r: Record) -> str:
    data = {
        "ts_ns": r.ts_ns,
        "gpu_uuid": r.gpu_uuid,
        "metric": r.metric,
        "value": r.value,
        "unit": r.unit,
        "source": r.source,
        "scope": r.scope.value,
        "quality": r.quality.value,
        "pid": r.pid,
        "context_id": r.context_id,
        "stream_id": r.stream_id,
    }
    return json.dumps(data)


def from_json(s: str) -> Record:
    d = json.loads(s)
    return Record(
        ts_ns=d["ts_ns"],
        gpu_uuid=d["gpu_uuid"],
        metric=d["metric"],
        value=d["value"],
        unit=d["unit"],
        source=d["source"],
        scope=Scope(d["scope"]),
        quality=Quality(d["quality"]),
        pid=d["pid"],
        context_id=d["context_id"],
        stream_id=d["stream_id"],
    )

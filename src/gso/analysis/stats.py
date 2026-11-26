"""지연 시간 통계 도우미."""

import math


def percentile(values: list[float], p: float) -> float | None:
    """nearest-rank 백분위수. 빈 입력은 None."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarize(latencies_ms: list[float], duration_s: float) -> dict:
    """반복 횟수, 처리량, p50/p95/p99 지연(ms)을 요약한다."""
    n = len(latencies_ms)
    return {
        "iterations": n,
        "throughput_per_s": n / duration_s if duration_s > 0 else None,
        "latency_ms": {
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
            "p99": percentile(latencies_ms, 99),
        },
    }

from gso.analysis.stats import percentile, summarize


def test_percentile_nearest_rank():
    v = [float(i) for i in range(1, 101)]
    assert percentile(v, 50) == 50.0
    assert percentile(v, 95) == 95.0
    assert percentile(v, 99) == 99.0
    assert percentile([], 50) is None


def test_summarize():
    s = summarize([10.0] * 200, duration_s=2.0)
    assert s["iterations"] == 200
    assert s["throughput_per_s"] == 100.0
    assert s["latency_ms"] == {"p50": 10.0, "p95": 10.0, "p99": 10.0}


def test_summarize_zero_elapsed_throughput_is_unknown():
    s = summarize([1.0, 2.0], duration_s=0.0)
    assert s["throughput_per_s"] is None  # unknown, not a measured 0.0
    assert s["iterations"] == 2

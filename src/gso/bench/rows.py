"""Row schema of ``results.jsonl``: one row per process per repetition.

Fields (``null`` = not measured / not applicable, never 0 for unknown):
  run_id, condition, rep (1-based), proc ("A"/"B"), workload, ok, exit_code, error,
  metric_kind ("latency"|"throughput_only"), iterations, throughput_per_s, p50_ms, p95_ms,
  p99_ms (null for throughput-only), gbps, fps, fps_min, fps_p5, frames_dropped, backend,
  build_s, mps (bool), client_thread_pct_requested (the REQUESTED upper limit per client),
  mps_server_active_thread_pct (the SERVER setting read through the control pipe),
  window_start_ns, window_end_ns (this process's measured window, epoch ns),
  window_overlap_s (overlap of both windows), fair (False: failed rep, missing window, overlap
  below 95 % of the duration, a TensorRT build inside the measured run or unknown build time, or
  a foreign GPU process seen during the repetition; applies to BOTH rows of a repetition), fair_reason, cooldown_timeout, gpu_temp_start_c, power_w_mean (W),
  sm_active_mean, sm_occupancy_mean, tensor_active_mean, dram_active_mean (DCGM %, means over the
  overlapped window), sm_clock_mhz_mean (NVML, same window), profiled, ctx_switch_rate (per s, this GPU; null under MPS), ctx_switch_note,
  overlap (always null: global captures have no kernels).
A failed repetition has ok=false on both rows and null metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from gso.bench import pair
from gso.bench.pair import ProcResult

MIN_OVERLAP = 0.95
MPS_NOTE = "N/A under MPS: no context-switch events are recorded (decision 4)"
_MEANS = (
    "sm_clock_mhz_mean",
    "power_w_mean",
    "sm_active_mean",
    "sm_occupancy_mean",
    "tensor_active_mean",
    "dram_active_mean",
)
_EXTRA = ("gbps", "fps", "fps_min", "fps_p5", "frames_dropped", "backend", "build_s")


def fairness(
    results: Sequence[ProcResult], duration_s: float, extra_reasons: Sequence[str] = ()
) -> tuple[float | None, list[str]]:
    overlap, reasons = _fairness(results, duration_s)
    return overlap, [*reasons, *extra_reasons]


def _fairness(
    results: Sequence[ProcResult], duration_s: float
) -> tuple[float | None, list[str]]:
    payloads = [r.payload for r in results]
    if any(p is None for p in payloads):
        return None, ["repetition failed"]
    overlap = pair.window_overlap_s(payloads)
    reasons = []
    if overlap is None:
        reasons.append("measured window missing")
    elif overlap < MIN_OVERLAP * duration_s:
        reasons.append(
            f"window overlap {overlap:.2f} s is below 95 % of {duration_s:g} s"
        )
    for r in results:
        if r.workload != "trt_infer":
            continue
        build = r.payload.get("build_s")
        if build is None:
            reasons.append(f"trt_infer {r.label} build time unknown (build_s missing)")
        elif build != 0:
            reasons.append(
                f"trt_infer {r.label} build_s={build} (engine built in a measured run)"
            )
    return overlap, reasons


def build_rows(
    *,
    run_id,
    condition,
    mps,
    pcts,
    rep,
    results,
    extra,
    cooldown_timeout,
    temp_start,
    duration_s,
    profile,
    window_means: Callable[[int, int], dict],
    extra_reasons: Sequence[str] = (),
) -> list[dict]:
    ok = all(r.payload is not None for r in results)
    overlap, reasons = fairness(results, duration_s, extra_reasons)
    means = dict.fromkeys(_MEANS)
    if overlap is not None:
        payloads = [r.payload for r in results]
        start = max(p["window_start_ns"] for p in payloads)
        end = min(p["window_end_ns"] for p in payloads)
        means.update(window_means(start, end))
    prof = extra.get("profile") or {}
    rows = []
    for r, pct in zip(results, pcts, strict=True):
        p = r.payload if ok else {}
        lat = p.get("latency_ms") or {}
        rows.append(
            {
                "run_id": run_id,
                "condition": condition,
                "rep": rep,
                "proc": r.label,
                "workload": r.workload,
                "ok": ok,
                "exit_code": r.exit_code,
                "error": r.error or (None if ok else "peer process failed"),
                "metric_kind": p.get("metric_kind"),
                "iterations": p.get("iterations"),
                "throughput_per_s": p.get("throughput_per_s"),
                "p50_ms": lat.get("p50"),
                "p95_ms": lat.get("p95"),
                "p99_ms": lat.get("p99"),
                **{k: p.get(k) for k in _EXTRA},
                "mps": mps,
                "client_thread_pct_requested": pct,
                "mps_server_active_thread_pct": extra.get("server_pct")
                if mps
                else None,
                "window_start_ns": (r.payload or {}).get("window_start_ns"),
                "window_end_ns": (r.payload or {}).get("window_end_ns"),
                "window_overlap_s": overlap,
                "fair": not reasons,
                "fair_reason": "; ".join(reasons) or None,
                "cooldown_timeout": cooldown_timeout,
                "gpu_temp_start_c": temp_start,
                **means,
                "profiled": profile,
                "ctx_switch_rate": None if mps else prof.get("ctx_switch_rate"),
                "ctx_switch_note": MPS_NOTE if profile and mps else prof.get("note"),
                "overlap": None,
            }
        )
    return rows

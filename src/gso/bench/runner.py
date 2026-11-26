"""MPS A/B benchmark runner: two workload processes together under five configurations.

For each condition (MPS off; MPS on with client thread-percentage limits 100/100, 50/50, 70/30,
30/70): one untimed warm-up repetition, then ``reps`` measured repetitions, each writing one row
per process to ``results.jsonl``. Fairness: both processes get the same ``--start-at`` barrier and
the same arguments; the runner stores the overlap of the two measured windows and flags
``fair: false`` below 95 % of the duration; the GPU cools down between repetitions; a TensorRT
engine is built once before anything is measured. Safety: ``ensure_safe`` before every
repetition, one ``MpsSession`` per MPS condition, only our own process groups are signalled.
Row fields: see ``gso.bench.rows``; run metadata: ``_Run.write_run_json``.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from gso import __version__
from gso.bench import pair
from gso.bench.guard import GuardedSession
from gso.bench.mps import MpsSession, UnsafeGpuError, ensure_safe
from gso.bench.pair import ProcResult, ProcSpec
from gso.bench.rows import build_rows
from gso.bench.workloads import WORKLOADS
from gso.providers.mps import query as _mps_query

CONDITIONS = [  # (name, MPS on, A's thread %, B's thread %)
    ("mps_off", False, None, None), ("mps_100_100", True, 100, 100),
    ("mps_50_50", True, 50, 50), ("mps_70_30", True, 70, 30), ("mps_30_70", True, 30, 70),
]
MIN_REPS = 3
COOLDOWN_MAX_S = 60.0
COOLDOWN_DELTA_C = 3.0
PROFILE_S = 10
MIN_PROFILE_DURATION_S = 20
PREBUILD_DEADLINE_S = 1800.0
SERVER_QUERY_AFTER_S = 2.0
WARM_BASELINE_C = 45.0
SAFE_RETRY_S = 5.0  # a refusal must persist this long (our own MPS server may still be exiting)


@dataclass
class RunOutcome:
    results_path: Path
    status: str  # complete | incomplete | aborted | interrupted
    message: str = ""

    @property
    def complete(self) -> bool:
        return self.status == "complete"


@dataclass
class Hooks:
    """Everything with side effects outside the runner; tests replace these."""

    telemetry: object | None = None
    ensure_safe: object = ensure_safe
    session_factory: object = MpsSession
    mps_query: object = _mps_query
    profiler: object | None = None  # (gpu, mps) -> {"ctx_switch_rate", "note", "error"}
    clock: object = time.time
    sleep: object = time.sleep
    monotonic: object = time.monotonic
    killpg: object = os.killpg
    progress: object = field(default=lambda line: print(line, file=sys.stderr, flush=True))
    environ: object | None = None


class _Abort(Exception):
    pass


class _SetupFailed(Exception):
    pass


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def _validate(workload_a, workload_b, out, reps, warmup_s, duration_s, profile, budget, names):
    for w in (workload_a, workload_b):
        if w not in WORKLOADS:
            raise ValueError(f"unknown workload {w!r}; available: {', '.join(WORKLOADS)}")
    if reps < MIN_REPS:
        raise ValueError(f"reps must be >= {MIN_REPS} (at least 3 repetitions), got {reps}")
    if not duration_s > 0 or not warmup_s > 0 or not budget > 0:
        raise ValueError("duration, warmup and setup budget must be > 0")
    if profile and duration_s < MIN_PROFILE_DURATION_S:
        raise ValueError(f"--profile needs a duration >= {MIN_PROFILE_DURATION_S} s")
    known = [c[0] for c in CONDITIONS]
    bad = [n for n in names if n not in known]
    if bad:
        raise ValueError(f"unknown condition(s) {', '.join(bad)}; available: {', '.join(known)}")
    if (Path(out) / "results.jsonl").exists():
        raise ValueError(f"{Path(out) / 'results.jsonl'} already exists; use a new --out")


class _Run:
    def __init__(self, gpu, wa, wb, out, reps, warmup_s, duration_s, profile, launch, names,
                 a_args, b_args, budget, hooks, env_info, run_id):
        self.gpu, self.wl, self.out = gpu, {"A": wa, "B": wb}, Path(out)
        self.args = {"A": list(a_args), "B": list(b_args)}
        self.reps, self.warmup_s, self.duration_s, self.profile = reps, warmup_s, duration_s, profile
        self.launch, self.budget, self.h = launch, budget, hooks
        self.env_info = dict(env_info or {})
        self.conditions = [c for c in CONDITIONS if c[0] in names]  # canonical order
        self.run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
        self.environ = dict(os.environ if hooks.environ is None else hooks.environ)
        self.warnings: list[str] = []
        self.failed: list[dict] = []
        self.done_conditions: list[str] = []
        self.rows_written = 0
        self.base_temp: float | None = None
        self.static: dict = {}
        self.launched: set[int] = set()
        self.cur_session = None
        self.seen_daemon: set[int] = set()  # daemon/server pids of the current MPS condition
        self.seen_lock = threading.Lock()
        self.contamination: list[dict] = []
        self.t_start = hooks.clock()
        self.mono_start = hooks.monotonic()
        self.runs_total = len(self.conditions) * (reps + 1)
        self.runs_done = 0
        self.warned_unverifiable = False
        self.results = self.out / "results.jsonl"

    # ---- output ----
    def say(self, line: str) -> None:
        self.h.progress(f"gso bench: {line}")

    def write_run_json(self, status: str, message: str = "") -> None:
        info = self.env_info
        data = {
            "run_id": self.run_id, "started_at": _iso(self.t_start),
            "ended_at": _iso(self.h.clock()) if status != "running" else None,
            "status": f"error: {message}" if status == "error" else status,
            "complete": status == "complete", "message": message,
            "gpu": self.gpu, "gpu_name": info.get("gpu_name"),
            "driver_version": info.get("driver_version"), "nsys_version": info.get("nsys_version"),
            "gso_version": __version__,
            "workloads": {k: {"name": self.wl[k], "args": self.args[k]} for k in "AB"},
            "duration_s": self.duration_s, "warmup_s": self.warmup_s, "reps": self.reps,
            "setup_budget_s": self.budget, "profile": self.profile,
            "conditions": [c[0] for c in self.conditions],
            "conditions_completed": self.done_conditions, "rows_written": self.rows_written,
            "failed_reps": self.failed, "warnings": self.warnings,
            "baseline_temp_c": self.base_temp,
            "dcgm_available": info.get("dcgm_available"), "dcgm_reason": info.get("dcgm_reason"),
            "telemetry_errors": list(getattr(self.h.telemetry, "errors", [])),
            "power_limit_w": self.static.get("power_limit_w"),
            "perf_state_at_start": self.static.get("perf_state"),
            "contamination_events": self.contamination,
        }
        tmp = self.out / "run.json.tmp"
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(tmp, self.out / "run.json")

    def append_rows(self, rows: list[dict]) -> None:
        with open(self.results, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
            f.flush()
            os.fsync(f.fileno())
        self.rows_written += len(rows)

    # ---- environment, safety, cooldown ----
    def env_for(self, session, pct) -> dict[str, str]:
        env = {k: v for k, v in self.environ.items() if not k.startswith("CUDA_MPS_")}
        if session is not None:
            env.update(session.client_env(pct))
        return env

    def is_own(self, pid: int) -> bool:
        """Is this compute process ours? Our launched process groups (workload and anything it
        spawns, e.g. ffmpeg) and the session's daemon/server. A vanished pid cannot be judged and
        is not counted."""
        if pid in self.launched:
            return True
        try:
            if os.getpgid(pid) in self.launched:
                return True
        except ProcessLookupError:
            return True
        except OSError:
            return False
        s = self.cur_session
        return s is not None and pid in self.own_pids(s)

    def own_pids(self, session) -> frozenset[int]:
        """The session's daemon/server pids, plus every one seen earlier in this condition (the
        MPS server is replaced between repetitions and the old one may still be exiting)."""
        if session is None:
            return frozenset()
        now = frozenset(session.own_daemon_pids)
        with self.seen_lock:
            self.seen_daemon |= now
            return frozenset(self.seen_daemon)

    def safe_problem(self, session) -> str | None:
        """ensure_safe, retried for SAFE_RETRY_S: the reason it keeps failing, or None."""
        deadline = self.h.monotonic() + SAFE_RETRY_S
        while True:
            try:
                self.h.ensure_safe(self.gpu, own_pids=self.own_pids(session))
                return None
            except UnsafeGpuError as e:
                msg = str(e)
            if self.h.monotonic() >= deadline:
                return msg
            self.h.sleep(1.0)

    def end_check(self, session) -> str | None:
        """Non-aborting safety check after a repetition."""
        return self.safe_problem(session)

    def check_safe(self, session) -> None:
        problem = self.safe_problem(session)
        if problem is not None:
            raise _Abort(problem)

    def cooldown(self) -> bool:
        """Wait until temperature <= baseline + 3 C and utilization is 0 %, at most 60 s."""
        t0 = self.h.monotonic()
        while True:
            temp, util = self.h.telemetry.probe()
            if temp is None and util is None and not self.warned_unverifiable:
                self.warned_unverifiable = True
                self.warnings.append("cooldown cannot be verified: GPU temperature/utilization unreadable")
            hot = temp is not None and self.base_temp is not None and temp > self.base_temp + COOLDOWN_DELTA_C
            busy = util is not None and util > 0
            if not hot and not busy:
                return False
            if self.h.monotonic() - t0 >= COOLDOWN_MAX_S:
                self.warnings.append(
                    f"cooldown timed out after {COOLDOWN_MAX_S:g} s (temperature {temp} C, baseline "
                    f"{self.base_temp} C, utilization {util} %)")
                return True
            self.h.sleep(1.0)

    # ---- running ----
    def prebuild(self) -> None:
        """Build each TensorRT engine once, alone on the GPU with MPS off, before measuring."""
        seen = []
        for k in "AB":
            if self.wl[k] != "trt_infer" or self.args[k] in seen:
                continue
            seen.append(self.args[k])
            self.say(f"pre-building the TensorRT engine ({' '.join(self.args[k]) or 'default options'})")
            self.check_safe(None)
            with _tmpdir() as d:
                out = Path(d) / "prebuild.json"
                spec = ProcSpec("A", "trt_infer", pair.workload_argv(
                    "trt_infer", self.gpu, 1, out, None, self.args[k]), self.env_for(None, None), out)
                (res,) = pair.run_procs(
                    [spec], launch=self.launch, clock=self.h.clock, sleep=self.h.sleep,
                    deadline_at=self.h.clock() + PREBUILD_DEADLINE_S, killpg=self.h.killpg)
            if res.payload is None:
                raise _SetupFailed(f"TensorRT engine pre-build failed: {res.error}")

    def run_condition(self, cond, first: bool) -> None:
        name, mps_on, pct_a, pct_b = cond
        session_cm = (GuardedSession(self.h.session_factory(self.gpu), self.warnings.append)
                      if mps_on else contextlib.nullcontext(None))
        with session_cm as session:
            self.cur_session = session
            self.seen_daemon = set()
            if not first:
                self.cooldown()
            self.check_safe(session)
            self.say(f"{name} warm-up ({self.warmup_s:g} s, results discarded)")
            warm = self.run_rep(name, session, (pct_a, pct_b), self.warmup_s, profile=False)
            self.runs_done += 1
            for r in warm:
                self.say(f"{name} warm-up {r.label}: " + ("ok" if r.payload else f"FAILED {r.error}"))
            for rep in range(1, self.reps + 1):
                timeout = self.cooldown()
                self.check_safe(session)
                temp, _ = self.h.telemetry.probe()
                self.say(f"{name} rep {rep}/{self.reps} {self.timing()}")
                self.measured_rep(name, session, (pct_a, pct_b), rep, timeout, temp)
                self.runs_done += 1
            self.cur_session = None
        self.cur_session = None
        self.done_conditions.append(name)

    def timing(self) -> str:
        elapsed = self.h.monotonic() - self.mono_start
        eta = elapsed / self.runs_done * (self.runs_total - self.runs_done) if self.runs_done else None
        eta_text = "unknown" if eta is None else f"{eta / 60:.0f} min"
        return f"elapsed {elapsed / 60:.1f} min, ETA {eta_text}"

    def run_rep(self, name, session, pcts, duration, profile, actions_out=None) -> list[ProcResult]:
        start_at = round(self.h.clock() + self.budget, 3)
        gpu_arg = session.client_gpu_arg if session is not None else self.gpu
        with _tmpdir() as d:
            specs = []
            for k, pct in zip("AB", pcts, strict=True):
                out = Path(d) / f"{k}.json"
                argv = pair.workload_argv(self.wl[k], gpu_arg, duration, out, start_at, self.args[k])
                specs.append(ProcSpec(k, self.wl[k], argv, self.env_for(session, pct), out))
            actions = []
            if session is not None:
                def ask():
                    st = self.h.mps_query(pipe_dir=session.pipe_dir)
                    if actions_out is not None:
                        actions_out["server_pct"] = getattr(st, "thread_pct", None)
                actions.append((start_at + SERVER_QUERY_AFTER_S, ask))
            if profile:
                def cap():
                    actions_out["profile"] = self.h.profiler(self.gpu, session is not None)
                actions.append((start_at + (duration - PROFILE_S) / 2, cap))
            return pair.run_procs(
                specs, launch=self.launch, clock=self.h.clock, sleep=self.h.sleep,
                deadline_at=start_at + duration + pair.DEADLINE_GRACE_S, actions=actions,
                killpg=self.h.killpg, on_launch=self.launched.add)

    def measured_rep(self, name, session, pcts, rep, timeout, temp) -> None:
        extra: dict = {}
        t0_ns = int(self.h.clock() * 1e9)
        results = self.run_rep(name, session, pcts, self.duration_s, self.profile, extra)
        t1_ns = int(self.h.clock() * 1e9)
        # Contamination: a foreign process seen during the repetition (per-second sampling) or
        # present right after it (non-aborting ensure_safe). The rows are still written, marked
        # unfair, and the run stops afterwards.
        why_foreign = []
        n_foreign = self.h.telemetry.foreign_in_window(t0_ns, t1_ns)
        if n_foreign:
            why_foreign.append("foreign GPU process during window")
            self.contamination.append({
                "condition": name, "rep": rep, "source": "telemetry",
                "foreign_process_count": n_foreign,
                "message": f"{n_foreign} foreign compute process(es) seen on GPU {self.gpu}"})
        end_problem = self.end_check(session)
        if end_problem is not None:
            why_foreign.append("foreign GPU process present at end of repetition")
            self.contamination.append({
                "condition": name, "rep": rep, "source": "end_check",
                "foreign_process_count": None, "message": end_problem})
        rows = self.rows(name, session, pcts, rep, results, extra, timeout, temp, why_foreign)
        self.append_rows(rows)
        if not all(r["ok"] for r in rows):
            self.failed.append({"condition": name, "rep": rep})
            why = "; ".join(f"{r['proc']}: {r['error']}" for r in rows if r["error"])
            self.warnings.append(f"{name} rep {rep} failed: {why}")
        elif not rows[0]["fair"]:
            self.warnings.append(f"{name} rep {rep} not fair: {rows[0]['fair_reason']}")
        if timeout:
            self.warnings.append(f"{name} rep {rep} started after a cooldown timeout")
        prof = extra.get("profile")
        if prof and prof.get("error"):
            self.warnings.append(f"{name} rep {rep} profile capture failed: {prof['error']}")
        if why_foreign:
            self.warnings.append(f"{name} rep {rep} contaminated: {'; '.join(why_foreign)}")
            raise _Abort(f"another process is using GPU {self.gpu} ({name} rep {rep} is marked "
                         "unfair); stopping")

    def rows(self, name, session, pcts, rep, results, extra, timeout, temp, foreign=()) -> list[dict]:
        return build_rows(
            run_id=self.run_id, condition=name, mps=session is not None, pcts=pcts, rep=rep,
            results=results, extra=extra, cooldown_timeout=timeout, temp_start=temp,
            duration_s=self.duration_s, profile=self.profile, window_means=self.h.telemetry.window_means,
            extra_reasons=foreign)

    def execute(self) -> RunOutcome:
        status, message = "error", "terminated unexpectedly"
        try:
            self.h.telemetry.set_own_filter(self.is_own)
            self.h.telemetry.start()
            self.static = self.h.telemetry.static_info()
            self.base_temp, _ = self.h.telemetry.probe()  # before any pre-build or warm-up
            if self.base_temp is not None and self.base_temp >= WARM_BASELINE_C:
                self.warnings.append(
                    f"cooldown baseline {self.base_temp:g} C is already >= {WARM_BASELINE_C:g} C: "
                    "the GPU was warm before this run, so 'cooled down' is relative to a warm start")
            self.prebuild()
            for i, cond in enumerate(self.conditions):
                self.run_condition(cond, first=i == 0)
            status, message = "complete", ""
            if self.failed:
                status = "incomplete"
                message = f"{len(self.failed)} repetition(s) failed"
        except _SetupFailed as e:
            status, message = "incomplete", str(e)
        except (_Abort, UnsafeGpuError) as e:
            status, message = "aborted", f"aborted for safety: {e}"
        except KeyboardInterrupt:
            status, message = "interrupted", "interrupted"
        except Exception as e:  # noqa: BLE001 - recorded, never reported as a complete run
            status, message = "error", f"{type(e).__name__}: {e}"
        finally:
            with contextlib.suppress(Exception):
                self.h.telemetry.stop()
            self.write_run_json(status, message)
        return RunOutcome(self.results, status, message)


def _tmpdir():
    import tempfile

    return tempfile.TemporaryDirectory(prefix="gso-bench-")


def run_matrix(gpu_index, workload_a, workload_b, out: Path, reps=3, warmup_s=10,
               duration_s=30, profile=False, launch=subprocess.Popen, *, conditions=None,
               a_args=(), b_args=(), setup_budget_s=60.0, hooks: Hooks | None = None,
               env_info: dict | None = None, run_id: str | None = None) -> RunOutcome:
    names = list(conditions) if conditions is not None else [c[0] for c in CONDITIONS]
    _validate(workload_a, workload_b, out, reps, warmup_s, duration_s, profile, setup_budget_s, names)
    hooks = hooks or Hooks()
    if hooks.telemetry is None:
        from gso.bench.telemetry import NvmlDcgmTelemetry

        hooks.telemetry = NvmlDcgmTelemetry(gpu_index)
    Path(out).mkdir(parents=True, exist_ok=True)
    run = _Run(gpu_index, workload_a, workload_b, out, reps, warmup_s, duration_s, profile,
               launch, names, a_args, b_args, setup_budget_s, hooks, env_info, run_id)
    run.results.touch()
    run.write_run_json("running")
    return run.execute()

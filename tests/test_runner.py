"""MPS A/B 벤치마크 러너: 가짜 launcher/세션/ensure_safe/텔레메트리, GPU와 실제 워크로드 없음."""

import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from gso.bench import runner
from gso.bench.mps import UnsafeGpuError
from gso.bench.runner import CONDITIONS, Hooks, run_matrix
from tests.fakes import FakeTime

T0 = 1_000_000.0
GPU = 2
MEANS = {
    "power_w_mean": 111.0, "sm_active_mean": 50.0, "sm_occupancy_mean": 20.0,
    "tensor_active_mean": 1.0, "dram_active_mean": 5.0,
}
ROW_KEYS = {
    "run_id", "condition", "rep", "proc", "workload", "ok", "exit_code", "error", "metric_kind",
    "iterations", "throughput_per_s", "p50_ms", "p95_ms", "p99_ms", "gbps", "fps", "fps_min",
    "fps_p5", "frames_dropped", "backend", "build_s", "mps", "client_thread_pct_requested",
    "mps_server_active_thread_pct", "window_start_ns", "window_end_ns", "window_overlap_s",
    "fair", "fair_reason", "cooldown_timeout", "gpu_temp_start_c", "power_w_mean",
    "sm_active_mean", "sm_occupancy_mean", "tensor_active_mean", "dram_active_mean", "profiled",
    "ctx_switch_rate", "ctx_switch_note", "overlap",
}


def _arg(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


class FakeProc:
    def __init__(self, launcher, argv, kw):
        self.launcher, self.argv, self.kw = launcher, argv, kw
        launcher.pids += 1
        self.pid = 5000 + launcher.pids
        self.workload = argv[argv.index("-m") + 1].rsplit(".", 1)[1]
        self.duration = float(_arg(argv, "--duration"))
        sa = _arg(argv, "--start-at")
        self.start_at = float(sa) if sa else None
        self.out = Path(_arg(argv, "--out"))
        self.returncode = None
        self.killed = False
        self.beh = launcher.behavior(self)
        base = self.start_at if self.start_at is not None else launcher.time.now
        self.done_at = base + self.duration + 0.5

    def payload(self):
        start = self.start_at if self.start_at is not None else self.launcher.time.now
        p = {
            "workload": self.workload, "pid": 1234, "metric_kind": "latency", "iterations": 1000,
            "throughput_per_s": 1000.0,
            "latency_ms": {"p50": 0.1, "p95": 0.2, "p99": 0.3},
            "window_start_ns": int(start * 1e9),
            "window_end_ns": int((start + self.duration) * 1e9),
        }
        if self.workload == "decode":
            p.update(metric_kind="throughput_only", fps=30.0, fps_min=28.0, fps_p5=29.0,
                     frames_dropped=0, latency_ms={"p50": None, "p95": None, "p99": None})
        if self.workload == "trt_infer":
            p.update(backend="tensorrt", trt_version="x", build_s=0.0, build_waited_s=0.0)
        p.update(self.beh.get("payload", {}))
        return p

    def _finish(self):
        code = self.beh.get("exit_code", 0)
        if code == 0:
            self.out.write_text(json.dumps(self.payload()))
        else:
            self.kw["stderr"].write(self.beh.get("stderr", "boom").encode())
            self.kw["stderr"].flush()
        self.returncode = code

    def poll(self):
        if self.returncode is None:
            if self.killed:
                self.returncode = -signal.SIGTERM
            elif self.launcher.time.now >= self.done_at - self.beh.get("early", 0):
                self._finish()
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            if self.killed:
                self.returncode = -signal.SIGTERM
            else:
                import subprocess

                raise subprocess.TimeoutExpired(self.argv, timeout)
        return self.returncode


class FakeLauncher:
    def __init__(self, time, events, behavior=None):
        self.time, self.events = time, events
        self.behavior = behavior or (lambda proc: {})
        self.procs = []
        self.pids = 0
        self.killpg_calls = []
        self.raise_on = None  # nth launch (1-based) raises KeyboardInterrupt

    def __call__(self, argv, **kw):
        n = len(self.procs) + 1
        if self.raise_on == n:
            raise KeyboardInterrupt
        proc = FakeProc(self, list(argv), kw)
        self.procs.append(proc)
        self.events.append(("launch", proc.workload, proc.duration))
        return proc

    def killpg(self, pgid, sig):
        self.killpg_calls.append((pgid, sig))
        for p in self.procs:
            if p.pid == pgid:
                p.killed = True


class FakeSession:
    def __init__(self, events):
        self.events = events
        self.pipe_dir = "/fake/pipe"
        self.client_gpu_arg = 0
        self.own_daemon_pids = frozenset({9001, 9002})

    def client_env(self, pct):
        return {
            "CUDA_MPS_PIPE_DIRECTORY": self.pipe_dir,
            "CUDA_MPS_LOG_DIRECTORY": "/fake/log",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(pct),
        }

    def __enter__(self):
        self.events.append(("enter",))
        return self

    def __exit__(self, *exc):
        self.events.append(("exit",))
        return False


class FakeTelemetry:
    def __init__(self, probe_fn=None, means=None):
        self.probe_fn = probe_fn or (lambda n: (40.0, 0.0))
        self.means = MEANS if means is None else means
        self.probes = 0
        self.windows = []
        self.started = self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def probe(self):
        self.probes += 1
        return self.probe_fn(self.probes)

    def window_means(self, start_ns, end_ns):
        self.windows.append((start_ns, end_ns))
        return dict(self.means)


class Rig:
    def __init__(self, tmp_path, behavior=None, refuse_at=None, probe_fn=None, means=None,
                 environ=None, profiler=None, query_pct=100.0):
        self.time = FakeTime()
        self.time.now = T0
        self.events = []
        self.launcher = FakeLauncher(self.time, self.events, behavior)
        self.safe_calls = []
        self.refuse_at = refuse_at
        self.telemetry = FakeTelemetry(probe_fn, means)
        self.sessions = []
        self.queries = []
        self.progress = []
        self.profile_calls = []

        def ensure_safe(gpu, nvml=None, own_pids=frozenset(), mps_listing=None):
            self.safe_calls.append((gpu, frozenset(own_pids)))
            self.events.append(("safe", frozenset(own_pids)))
            if self.refuse_at == len(self.safe_calls):
                raise UnsafeGpuError("another user's process appeared")

        def session_factory(gpu):
            assert gpu == GPU
            s = FakeSession(self.events)
            self.sessions.append(s)
            return s

        def mps_query(pipe_dir=None, **kw):
            self.queries.append((pipe_dir, self.time.now))
            return SimpleNamespace(thread_pct=query_pct, active=True)

        def default_profiler(gpu, mps):
            self.profile_calls.append((gpu, mps, self.time.now))
            return {"ctx_switch_rate": 12.5, "note": None, "error": None}

        self.hooks = Hooks(
            telemetry=self.telemetry, ensure_safe=ensure_safe, session_factory=session_factory,
            mps_query=mps_query, profiler=profiler or default_profiler,
            clock=self.time.clock, sleep=self.time.sleep, monotonic=self.time.clock,
            killpg=self.launcher.killpg, progress=self.progress.append,
            environ=environ if environ is not None else {"PATH": "/bin"},
        )
        self.out = tmp_path / "out"

    def run(self, wa="small_kernels", wb="small_kernels", **kw):
        kw.setdefault("reps", 3)
        return run_matrix(GPU, wa, wb, self.out, launch=self.launcher, hooks=self.hooks, **kw)

    def rows(self):
        p = self.out / "results.jsonl"
        return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []

    def run_json(self):
        return json.loads((self.out / "run.json").read_text())


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


# ---- matrix shape and environments ---------------------------------------------------------


def test_conditions_are_the_five_specified():
    assert CONDITIONS == [
        ("mps_off", False, None, None), ("mps_100_100", True, 100, 100),
        ("mps_50_50", True, 50, 50), ("mps_70_30", True, 70, 30), ("mps_30_70", True, 30, 70),
    ]


def test_full_matrix_is_30_rows_with_the_full_schema(rig):
    out = rig.run()
    rows = rig.rows()
    assert out.status == "complete" and out.complete
    assert out.results_path == rig.out / "results.jsonl"
    assert len(rows) == 30
    for r in rows:
        assert set(r) == ROW_KEYS
        assert r["ok"] is True and r["fair"] is True
    assert {r["condition"] for r in rows} == {c[0] for c in CONDITIONS}
    assert [r["proc"] for r in rows[:4]] == ["A", "B", "A", "B"]
    assert [r["rep"] for r in rows[:6]] == [1, 1, 2, 2, 3, 3]


def test_no_pid_or_host_info_in_rows_or_run_json(rig):
    rig.run(conditions=["mps_off"])
    for r in rig.rows():
        assert "pid" not in r
    text = (rig.out / "run.json").read_text()
    assert "hostname" not in text and "pid" not in json.loads(text)


def _env_of(proc):
    return proc.kw["env"]


def test_mps_70_30_gives_a_70_and_b_30(rig):
    rig.run(conditions=["mps_70_30"])
    pa, pb = rig.launcher.procs[-2:]  # last pair = a measured rep of this condition
    key = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
    assert _env_of(pa)[key] == "70" and _env_of(pb)[key] == "30"
    assert _env_of(pa)["CUDA_MPS_PIPE_DIRECTORY"] == "/fake/pipe"


def test_mps_off_env_has_no_cuda_mps_even_if_inherited(tmp_path):
    inherited = {"PATH": "/bin", "CUDA_MPS_PIPE_DIRECTORY": "/system/mps",
                 "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "10", "CUDA_MPS_LOG_DIRECTORY": "/x"}
    r = Rig(tmp_path, environ=inherited)
    r.run(conditions=["mps_off"])
    assert r.launcher.procs
    for p in r.launcher.procs:
        assert not [k for k in _env_of(p) if k.startswith("CUDA_MPS_")]
        assert _env_of(p)["PATH"] == "/bin"


def test_mps_clients_get_gpu_0_and_mps_off_gets_host_gpu(rig):
    rig.run(conditions=["mps_off", "mps_50_50"])
    for p in rig.launcher.procs:
        want = "0" if "CUDA_MPS_PIPE_DIRECTORY" in _env_of(p) else str(GPU)
        assert _arg(p.argv, "--gpu") == want
    assert any(_arg(p.argv, "--gpu") == "0" for p in rig.launcher.procs)
    assert any(_arg(p.argv, "--gpu") == str(GPU) for p in rig.launcher.procs)


def test_same_args_for_every_condition_and_both_processes(rig):
    rig.run(a_args=["--size", "512"], b_args=["--size", "512"])
    meas = [p for p in rig.launcher.procs if p.duration == 30.0]
    assert len(meas) == 30
    for p in meas:
        assert p.argv[-2:] == ["--size", "512"]
        assert _arg(p.argv, "--duration") == "30"
    assert all(p.kw["start_new_session"] is True for p in rig.launcher.procs)


def test_conditions_subset_in_listed_order(rig):
    rig.run(conditions=["mps_30_70", "mps_off"])
    assert [r["condition"] for r in rig.rows()][::6] == ["mps_off", "mps_30_70"]
    assert len(rig.rows()) == 12


def test_unknown_condition_is_value_error(rig):
    with pytest.raises(ValueError, match="mps_99"):
        rig.run(conditions=["mps_99"])
    assert rig.launcher.procs == []


# ---- validation -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [{"reps": 2}, {"duration_s": 0}, {"duration_s": -1}, {"warmup_s": 0},
     {"profile": True, "duration_s": 19}, {"setup_budget_s": 0}],
)
def test_invalid_parameters_raise_before_anything_runs(rig, kw):
    with pytest.raises(ValueError):
        rig.run(**kw)
    assert rig.launcher.procs == [] and rig.events == []
    assert not (rig.out / "results.jsonl").exists()


def test_unknown_workload_raises(rig):
    with pytest.raises(ValueError, match="nope"):
        rig.run(wa="nope")
    assert rig.launcher.procs == []


def test_existing_results_file_is_refused(rig):
    rig.out.mkdir()
    (rig.out / "results.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="results.jsonl"):
        rig.run()
    assert (rig.out / "results.jsonl").read_text() == "{}\n"
    assert rig.launcher.procs == []


# ---- sessions and safety --------------------------------------------------------------------


def test_one_session_per_mps_condition_around_warmup_and_reps(rig):
    rig.run(conditions=["mps_off", "mps_50_50", "mps_70_30"])
    ev = [e[0] for e in rig.events]
    assert ev.count("enter") == 2 and ev.count("exit") == 2
    assert len(rig.sessions) == 2
    # enter -> safe -> warm-up launch(es) ... 3 measured pairs ... exit
    seg = rig.events[ev.index("enter") : ev.index("exit") + 1]
    kinds = [e[0] for e in seg]
    assert kinds[:3] == ["enter", "safe", "launch"]
    assert [e[2] for e in seg if e[0] == "launch"] == [10.0, 10.0] + [30.0, 30.0] * 3
    assert kinds[-1] == "exit"
    # mps_off part has no enter before its launches
    assert ev.index("launch") < ev.index("enter")


def test_ensure_safe_before_every_warmup_and_rep_with_right_own_pids(rig):
    rig.run(conditions=["mps_off", "mps_50_50"])
    # per condition: 1 warm-up + 3 reps
    assert len(rig.safe_calls) == 8
    assert all(g == GPU for g, _ in rig.safe_calls)
    assert [own for _, own in rig.safe_calls[:4]] == [frozenset()] * 4
    assert [own for _, own in rig.safe_calls[4:]] == [frozenset({9001, 9002})] * 4
    # each launch pair is immediately preceded by a safe check
    flat = [e[0] for e in rig.events if e[0] in ("safe", "launch")]
    pairs = "".join("S" if k == "safe" else "L" for k in flat)
    assert pairs == "SLL" * 8


def test_safety_refusal_aborts_whole_run_keeps_rows_and_exits_session(rig):
    rig.refuse_at = 7  # 2nd rep of the 2nd condition (mps_off: 4 checks, mps_100_100: 3rd = warm-up+rep2)
    out = rig.run(conditions=["mps_off", "mps_100_100", "mps_50_50"])
    assert out.status == "aborted" and not out.complete
    assert "another user's process" in out.message
    rows = rig.rows()
    assert len(rows) == 3 * 2 + 1 * 2  # mps_off 3 reps + mps_100_100 rep 1
    assert not any(r["condition"] == "mps_50_50" for r in rows)
    assert [e[0] for e in rig.events].count("enter") == [e[0] for e in rig.events].count("exit") == 1
    rj = rig.run_json()
    assert rj["complete"] is False and rj["status"] == "aborted"
    assert rig.telemetry.stopped == 1


def test_session_enter_refusal_aborts(rig):
    def boom(gpu):
        raise UnsafeGpuError("another gso MPS session of this user is active")

    rig.hooks.session_factory = boom
    out = rig.run(conditions=["mps_50_50"])
    assert out.status == "aborted" and rig.rows() == []
    assert rig.run_json()["complete"] is False


def test_session_leftover_daemon_at_exit_is_an_abort(rig):
    class Leaky(FakeSession):
        def __exit__(self, *exc):
            raise RuntimeError("MPS daemon of this session is still running")

    rig.hooks.session_factory = lambda gpu: Leaky(rig.events)
    out = rig.run(conditions=["mps_50_50"])
    assert out.status == "aborted" and "still running" in out.message


# ---- failures, interrupts -------------------------------------------------------------------


def test_nonzero_exit_marks_rows_failed_session_still_exited_run_continues(tmp_path):
    def beh(proc):
        if proc.workload == "small_kernels" and proc.start_at and proc.duration == 30.0:
            beh.n += 1
            if beh.n == 1:  # first measured launch overall = rep 1 proc A of mps_off
                return {"exit_code": 3, "stderr": "CUDA error: boom\n"}
        return {}

    beh.n = 0
    r = Rig(tmp_path, behavior=beh)
    out = r.run(conditions=["mps_50_50"])
    assert out.status == "incomplete" and not out.complete
    rows = r.rows()
    assert len(rows) == 6
    a1, b1 = rows[0], rows[1]
    assert a1["ok"] is False and a1["exit_code"] == 3 and "boom" in a1["error"]
    assert a1["throughput_per_s"] is None and a1["fair"] is False
    assert b1["ok"] is False  # the peer was terminated: the rep is not a measurement
    assert all(x["ok"] for x in rows[2:])
    assert [e[0] for e in r.events].count("exit") == 1
    assert r.run_json()["failed_reps"] == [{"condition": "mps_50_50", "rep": 1}]
    assert r.run_json()["complete"] is False


def test_failed_process_makes_the_peer_get_terminated_by_group(tmp_path):
    def beh(proc):
        return {"exit_code": 1, "stderr": "x", "early": 100} if "--start-at" in proc.argv and \
            proc.duration == 30.0 and proc.pid % 2 == 1 else {}

    r = Rig(tmp_path, behavior=beh)
    r.run(conditions=["mps_off"], reps=3)
    assert r.launcher.killpg_calls  # only groups the runner created
    created = {p.pid for p in r.launcher.procs}
    assert {pgid for pgid, _ in r.launcher.killpg_calls} <= created
    assert all(sig == signal.SIGTERM for _, sig in r.launcher.killpg_calls)


def test_missed_start_barrier_fails_that_rep_loudly(tmp_path):
    def beh(proc):
        if proc.duration == 30.0 and proc.pid == 5005:  # rep 2, proc A
            return {"exit_code": 1, "stderr": "small_kernels: start barrier missed by 2.000 s"}
        return {}

    r = Rig(tmp_path, behavior=beh)
    out = r.run(conditions=["mps_off"])
    bad = [x for x in r.rows() if not x["ok"]]
    assert bad and any("start barrier missed" in x["error"] for x in bad)
    assert out.status == "incomplete"
    assert any("barrier" in w for w in r.run_json()["warnings"])


def test_keyboard_interrupt_in_launch_terminates_children_and_exits_session(rig):
    rig.launcher.raise_on = 6  # warm-up pair (1, 2), rep 1 pair (3, 4), rep 2: A is the 5th, B raises
    out = rig.run(conditions=["mps_50_50"])
    assert out.status == "interrupted" and not out.complete
    assert len(rig.rows()) == 2  # rep 1 only
    assert [e[0] for e in rig.events].count("exit") == 1
    # the already-launched A of the interrupted rep is terminated through its own group
    assert rig.launcher.killpg_calls == [(rig.launcher.procs[-1].pid, signal.SIGTERM)]
    rj = rig.run_json()
    assert rj["complete"] is False and rj["status"] == "interrupted"
    assert rig.telemetry.stopped == 1


def test_keyboard_interrupt_while_waiting_terminates_both_groups(rig):
    orig = rig.time.sleep
    state = {"n": 0}

    def sleeper(s):
        state["n"] += 1
        if state["n"] == 50:
            raise KeyboardInterrupt
        orig(s)

    rig.hooks.sleep = sleeper
    out = rig.run(conditions=["mps_off"])
    assert out.status == "interrupted"
    assert {pgid for pgid, _ in rig.launcher.killpg_calls} == {p.pid for p in rig.launcher.procs}


def test_workload_that_never_finishes_is_terminated_at_the_deadline(tmp_path):
    def beh(proc):
        return {"early": -10_000} if proc.duration == 30.0 else {}

    r = Rig(tmp_path, behavior=beh)
    out = r.run(conditions=["mps_off"])
    assert out.status == "incomplete"
    assert all(not x["ok"] and "timed out" in x["error"] for x in r.rows())
    assert r.launcher.killpg_calls


# ---- fairness -------------------------------------------------------------------------------


def test_both_processes_get_the_same_start_at_in_the_future(rig):
    rig.run(conditions=["mps_off"])
    procs = rig.launcher.procs
    for i in range(0, len(procs), 2):
        a, b = procs[i], procs[i + 1]
        assert a.start_at == b.start_at and a.start_at is not None
        assert a.start_at >= T0 + 60  # setup budget


def test_window_overlap_and_fair_flag(tmp_path):
    def beh(proc):
        if proc.duration == 30.0 and proc.pid == 5004:  # B of the first measured rep
            s = proc.start_at + 3.0  # 3 s late: overlap 27 s = 90 %
            return {"payload": {"window_start_ns": int(s * 1e9),
                                "window_end_ns": int((s + 30.0) * 1e9)}}
        return {}

    r = Rig(tmp_path, behavior=beh)
    r.run(conditions=["mps_off"])
    rows = r.rows()
    assert rows[0]["window_overlap_s"] == pytest.approx(27.0)
    assert rows[0]["fair"] is False and rows[1]["fair"] is False
    assert "overlap" in rows[0]["fair_reason"]
    assert rows[2]["window_overlap_s"] == pytest.approx(30.0) and rows[2]["fair"] is True
    # telemetry means are taken over the overlapped window only
    assert r.telemetry.windows[0] == (max(rows[0]["window_start_ns"], rows[1]["window_start_ns"]),
                                      min(rows[0]["window_end_ns"], rows[1]["window_end_ns"]))


def test_overlap_just_above_95_percent_is_fair(tmp_path):
    def beh(proc):
        if proc.duration == 30.0 and proc.pid == 5004:
            s = proc.start_at + 1.4  # overlap 28.6 s = 95.3 %
            return {"payload": {"window_start_ns": int(s * 1e9),
                                "window_end_ns": int((s + 30.0) * 1e9)}}
        return {}

    r = Rig(tmp_path, behavior=beh)
    r.run(conditions=["mps_off"])
    assert r.rows()[0]["fair"] is True


def test_missing_window_is_unfair(tmp_path):
    r = Rig(tmp_path, behavior=lambda p: {"payload": {"window_start_ns": None, "window_end_ns": None}}
            if p.duration == 30.0 else {})
    r.run(conditions=["mps_off"])
    row = r.rows()[0]
    assert row["fair"] is False and row["window_overlap_s"] is None
    assert row["power_w_mean"] is None and r.telemetry.windows == []


def test_warmup_results_are_not_written_but_are_logged(rig):
    rig.run(conditions=["mps_off"])
    assert len(rig.rows()) == 6
    warm = [p for p in rig.launcher.procs if p.duration == 10.0]
    assert len(warm) == 2
    assert any("warm-up" in line for line in rig.progress)
    assert len(rig.telemetry.windows) == 3  # only measured reps are summarised


def test_cooldown_timeout_flag_when_temperature_never_drops(tmp_path):
    r = Rig(tmp_path, probe_fn=lambda n: (40.0, 0.0) if n == 1 else (70.0, 0.0))
    r.run(conditions=["mps_off"])
    rows = r.rows()
    assert all(x["cooldown_timeout"] is True for x in rows)
    assert all(x["gpu_temp_start_c"] == 70.0 for x in rows)
    assert any("cooldown" in w for w in r.run_json()["warnings"])
    assert sum(1 for s in r.time.sleeps if s == 1.0) >= 60  # waited the 60 s bound


def test_cooldown_ok_when_temperature_and_util_settle(tmp_path):
    def probe(n):
        return (40.0, 0.0) if n == 1 else (41.5, 0.0)  # within +3 C

    r = Rig(tmp_path, probe_fn=probe)
    r.run(conditions=["mps_off"])
    assert all(x["cooldown_timeout"] is False for x in r.rows())


def test_cooldown_waits_for_gpu_util_zero(tmp_path):
    seen = {"n": 0}

    def probe(n):
        if n == 1:
            return 40.0, 0.0
        seen["n"] += 1
        return 40.0, (35.0 if seen["n"] <= 2 else 0.0)

    r = Rig(tmp_path, probe_fn=probe)
    r.run(conditions=["mps_off"], reps=3)
    assert all(x["cooldown_timeout"] is False for x in r.rows())
    assert 1.0 in r.time.sleeps


def test_no_cooldown_before_the_very_first_warmup(tmp_path):
    r = Rig(tmp_path, probe_fn=lambda n: (40.0, 99.0) if n == 1 else (40.0, 0.0))
    r.run(conditions=["mps_off"])
    # first probe is the baseline; the first launch is not delayed by it
    assert r.launcher.procs[0].start_at == pytest.approx(T0 + 60, abs=1.0)


def test_trt_prebuild_once_before_any_measured_rep(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "WORKLOADS", ("small_kernels", "trt_infer"))
    r = Rig(tmp_path)
    r.run(wa="trt_infer", wb="small_kernels", a_args=["--batch", "4"], conditions=["mps_off", "mps_50_50"])
    procs = r.launcher.procs
    pre = procs[0]
    assert pre.workload == "trt_infer" and pre.start_at is None
    assert _arg(pre.argv, "--gpu") == str(GPU) and "--batch" in pre.argv
    assert not [k for k in _env_of(pre) if k.startswith("CUDA_MPS_")]
    assert sum(1 for p in procs if p.start_at is None) == 1
    assert r.events[0][0] == "safe" and r.events[1][0] == "launch"
    assert len(r.rows()) == 12


def test_trt_prebuild_failure_stops_before_measuring(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "WORKLOADS", ("small_kernels", "trt_infer"))
    r = Rig(tmp_path, behavior=lambda p: {"exit_code": 1, "stderr": "no tensorrt"}
            if p.workload == "trt_infer" else {})
    out = r.run(wa="trt_infer", wb="small_kernels")
    assert out.status == "incomplete" and r.rows() == []
    assert "no tensorrt" in out.message
    assert len(r.launcher.procs) == 1


def test_no_prebuild_without_trt(rig):
    rig.run(conditions=["mps_off"])
    assert all(p.start_at is not None for p in rig.launcher.procs)


def test_trt_row_with_build_time_is_unfair(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "WORKLOADS", ("small_kernels", "trt_infer"))

    def beh(proc):
        if proc.workload == "trt_infer" and proc.duration == 30.0 and proc.pid == 5004:
            return {"payload": {"build_s": 12.0}}
        return {}

    r = Rig(tmp_path, behavior=beh)
    r.run(wa="trt_infer", wb="small_kernels", conditions=["mps_off"])
    rows = r.rows()
    assert rows[0]["build_s"] == 12.0 and rows[0]["fair"] is False
    assert "build" in rows[0]["fair_reason"]
    assert rows[2]["fair"] is True and rows[2]["build_s"] == 0.0


# ---- row contents ---------------------------------------------------------------------------


def test_throughput_only_rows_have_null_latencies_and_fps(tmp_path):
    r = Rig(tmp_path)
    r.run(wa="decode", wb="small_kernels", conditions=["mps_off"])
    a = r.rows()[0]
    assert a["metric_kind"] == "throughput_only" and a["workload"] == "decode"
    assert a["p50_ms"] is a["p95_ms"] is a["p99_ms"] is None
    assert a["fps"] == 30.0 and a["frames_dropped"] == 0 and a["throughput_per_s"] == 1000.0
    b = r.rows()[1]
    assert b["metric_kind"] == "latency" and b["p99_ms"] == 0.3
    assert b["fps"] is None and b["gbps"] is None and b["backend"] is None and b["build_s"] is None


def test_unavailable_dcgm_gives_null_means_never_zero(tmp_path):
    r = Rig(tmp_path, means={**MEANS, "sm_active_mean": None, "sm_occupancy_mean": None,
                             "tensor_active_mean": None, "dram_active_mean": None,
                             "power_w_mean": None})
    r.run(conditions=["mps_off"])
    for x in r.rows():
        for k in ("sm_active_mean", "sm_occupancy_mean", "tensor_active_mean",
                  "dram_active_mean", "power_w_mean"):
            assert x[k] is None
    assert r.rows()[0]["power_w_mean"] is None


def test_telemetry_means_flow_into_rows(rig):
    rig.run(conditions=["mps_off"])
    for k, v in MEANS.items():
        assert rig.rows()[0][k] == v


def test_requested_vs_server_thread_pct_recorded_separately(rig):
    rig.run(conditions=["mps_off", "mps_70_30"])
    off = [r for r in rig.rows() if r["condition"] == "mps_off"]
    on = [r for r in rig.rows() if r["condition"] == "mps_70_30"]
    assert all(r["mps"] is False and r["client_thread_pct_requested"] is None
               and r["mps_server_active_thread_pct"] is None for r in off)
    assert all(r["mps"] is True and r["mps_server_active_thread_pct"] == 100.0 for r in on)
    assert [r["client_thread_pct_requested"] for r in on[:2]] == [70, 30]
    # the server is asked while the clients run (after the barrier), never from a stale session
    assert rig.queries and all(p == "/fake/pipe" for p, _ in rig.queries)
    assert all(t >= T0 + 60 for _, t in rig.queries)


def test_server_pct_null_when_query_has_no_answer(tmp_path):
    r = Rig(tmp_path, query_pct=None)
    r.run(conditions=["mps_50_50"])
    assert all(x["mps_server_active_thread_pct"] is None for x in r.rows())


def test_gpu_temp_start_recorded(tmp_path):
    r = Rig(tmp_path, probe_fn=lambda n: (33.0, 0.0))
    r.run(conditions=["mps_off"])
    assert all(x["gpu_temp_start_c"] == 33.0 for x in r.rows())


def test_row_values_come_from_workload_json(rig):
    rig.run(conditions=["mps_off"])
    r = rig.rows()[0]
    assert r["throughput_per_s"] == 1000.0 and r["p50_ms"] == 0.1 and r["p95_ms"] == 0.2
    assert r["iterations"] == 1000 and r["workload"] == "small_kernels" and r["run_id"]
    assert r["window_start_ns"] < r["window_end_ns"] and r["window_overlap_s"] == pytest.approx(30.0)
    assert r["exit_code"] == 0 and r["error"] is None and r["profiled"] is False
    assert r["ctx_switch_rate"] is None and r["overlap"] is None


# ---- profiling ------------------------------------------------------------------------------


def test_profile_one_capture_per_measured_rep_inside_the_window(rig):
    rig.run(conditions=["mps_off"], duration_s=30, profile=True)
    assert len(rig.profile_calls) == 3  # not in the warm-up
    meas = [p for p in rig.launcher.procs if p.duration == 30.0]
    for k, (gpu, mps, t) in enumerate(rig.profile_calls):
        start_at = meas[2 * k].start_at
        assert gpu == GPU and mps is False
        assert t >= start_at + 10.0 and t < start_at + 10.0 + 1.0  # (30-10)/2
    for x in rig.rows():
        assert x["profiled"] is True and x["ctx_switch_rate"] == 12.5 and x["overlap"] is None


def test_profile_under_mps_has_null_rate_with_note(rig):
    rig.run(conditions=["mps_50_50"], profile=True)
    for x in rig.rows():
        assert x["profiled"] is True and x["ctx_switch_rate"] is None
        assert "MPS" in x["ctx_switch_note"]
    assert len(rig.profile_calls) == 3  # same capture load in every condition


def test_profile_failure_is_recorded_not_hidden(tmp_path):
    r = Rig(tmp_path, profiler=lambda gpu, mps: {"ctx_switch_rate": None, "note": None,
                                                 "error": "nsys export failed"})
    r.run(conditions=["mps_off"], profile=True)
    assert all(x["profiled"] is True and x["ctx_switch_rate"] is None for x in r.rows())
    assert any("nsys export failed" in w for w in r.run_json()["warnings"])


def test_no_profile_means_no_capture(rig):
    rig.run(conditions=["mps_off"])
    assert rig.profile_calls == []
    assert all(x["profiled"] is False for x in rig.rows())


# ---- run.json -------------------------------------------------------------------------------


def test_run_json_contents(rig):
    rig.run(conditions=["mps_off", "mps_50_50"], a_args=["--size", "128"],
            env_info={"gpu_name": "Test GPU", "driver_version": "1.2", "nsys_version": None})
    rj = rig.run_json()
    assert rj["complete"] is True and rj["status"] == "complete"
    assert rj["gpu"] == GPU and rj["gpu_name"] == "Test GPU" and rj["driver_version"] == "1.2"
    assert rj["workloads"] == {"A": {"name": "small_kernels", "args": ["--size", "128"]},
                               "B": {"name": "small_kernels", "args": []}}
    assert rj["duration_s"] == 30 and rj["warmup_s"] == 10 and rj["reps"] == 3
    assert rj["conditions"] == ["mps_off", "mps_50_50"]
    assert rj["conditions_completed"] == ["mps_off", "mps_50_50"]
    assert rj["profile"] is False and rj["setup_budget_s"] == 60
    assert rj["run_id"] == rig.rows()[0]["run_id"]
    assert rj["started_at"] < rj["ended_at"] or rj["started_at"] == rj["ended_at"]
    assert isinstance(rj["warnings"], list) and rj["gso_version"]
    assert rj["failed_reps"] == []


def test_run_json_marked_incomplete_before_the_run_finishes(rig):
    seen = {}
    orig = rig.hooks.ensure_safe

    def spy(*a, **kw):
        p = rig.out / "run.json"
        if p.exists() and "x" not in seen:
            seen["x"] = json.loads(p.read_text())["complete"]
        return orig(*a, **kw)

    rig.hooks.ensure_safe = spy
    rig.run(conditions=["mps_off"])
    assert seen["x"] is False


def test_progress_lines_name_condition_rep_and_eta(rig):
    rig.run(conditions=["mps_off"])
    text = "\n".join(rig.progress)
    assert "mps_off" in text and "rep 1/3" in text and "ETA" in text and "elapsed" in text

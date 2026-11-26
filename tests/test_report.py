import json

import pytest

from gso import cli
from gso.bench import report
from gso.bench.report import aggregate, compare, to_markdown


def row(cond="mps_off", proc="A", thr=100.0, p99=1.0, rep=1, **kw):
    r = {
        "run_id": "r",
        "condition": cond,
        "rep": rep,
        "proc": proc,
        "workload": "small_kernels",
        "ok": True,
        "metric_kind": "latency",
        "throughput_per_s": thr,
        "p50_ms": 0.5,
        "p95_ms": 0.9,
        "p99_ms": p99,
        "fair": True,
        "fair_reason": None,
        "profiled": False,
        "client_thread_pct_requested": None,
        "mps_server_active_thread_pct": None,
        "cooldown_timeout": False,
        "gpu_temp_start_c": 40.0,
    }
    r.update(kw)
    return r


def by(rows, cond, proc="A"):
    return next(
        c for c in compare(rows) if c["condition"] == cond and c["proc"] == proc
    )


# -- brief Step 1 -------------------------------------------------------------------------
def test_throughput_change_pct():
    rows = [row("mps_off", thr=100.0), row("mps_100_100", thr=120.0)]
    assert by(rows, "mps_100_100")["throughput_change_pct"] == pytest.approx(20.0)


def test_p99_change_pct():
    rows = [row("mps_off", p99=2.0), row("mps_100_100", p99=1.0)]
    assert by(rows, "mps_100_100")["p99_change_pct"] == pytest.approx(-50.0)


def test_cv_of_identical_values_is_zero():
    rows = [row(thr=10.0, rep=i) for i in (1, 2, 3)]
    st = aggregate(rows)[("mps_off", "A")]
    assert st.throughput.cv == 0.0
    assert st.throughput.mean == 10.0 and st.throughput.n == 3


def test_missing_baseline_is_none_and_na_in_markdown():
    rows = [row("mps_100_100", thr=120.0)]
    c = by(rows, "mps_100_100")
    assert c["throughput_change_pct"] is None and c["p99_change_pct"] is None
    md = to_markdown(rows)
    assert "N/A" in md
    assert "None" not in md and "nan" not in md.lower()


def test_profiled_rows_split_into_two_tables():
    rows = [row("mps_off", thr=100.0), row("mps_off", thr=50.0, profiled=True)]
    md = to_markdown(rows)
    assert "## Unprofiled runs" in md and "## Profiled runs" in md
    assert "on-demand capture" in md
    with pytest.raises(ValueError):
        aggregate(rows)
    with pytest.raises(ValueError):
        compare(rows)


def test_profiled_never_compared_with_unprofiled():
    rows = [row("mps_off", thr=100.0), row("mps_100_100", thr=50.0, profiled=True)]
    md = to_markdown(rows)
    # the profiled table has no mps_off baseline, so its change is N/A, not -50 %
    assert "-50.0" not in md


# -- ruling 1: fair/ok filter ----------------------------------------------------------------
def test_unfair_and_failed_rows_excluded_but_counted():
    rows = [
        row(thr=100.0, rep=1),
        row(thr=100.0, rep=2),
        row(thr=1.0, rep=3, fair=False, fair_reason="foreign process"),
        row(thr=None, rep=4, ok=False, fair=False),
    ]
    c = by(rows, "mps_off")
    st = aggregate(rows)[("mps_off", "A")]
    assert st.throughput.mean == 100.0 and st.n_used == 2 and st.n_excluded == 2
    assert c["n"] == 2 and c["n_excluded"] == 2
    md = to_markdown(rows)
    assert "foreign process" in md


def test_condition_with_only_excluded_rows_still_listed_as_na():
    rows = [row("mps_off"), row("mps_100_100", fair=False, fair_reason="x")]
    c = by(rows, "mps_100_100")
    assert c["n"] == 0 and c["n_excluded"] == 1 and c["throughput_mean"] is None
    assert c["throughput_change_pct"] is None
    assert "mps_100_100" in to_markdown(rows)


# -- ruling 2: degenerate statistics -----------------------------------------------------------
def test_single_rep_has_no_stdev_or_cv():
    st = aggregate([row(thr=10.0)])[("mps_off", "A")]
    assert (
        st.throughput.stdev is None
        and st.throughput.cv is None
        and st.throughput.n == 1
    )


def test_zero_mean_has_no_cv():
    rows = [row(thr=0.0, rep=1), row(thr=0.0, rep=2)]
    st = aggregate(rows)[("mps_off", "A")]
    assert (
        st.throughput.mean == 0.0
        and st.throughput.stdev == 0.0
        and st.throughput.cv is None
    )


def test_zero_baseline_gives_none_change():
    rows = [row("mps_off", thr=0.0), row("mps_100_100", thr=5.0)]
    assert by(rows, "mps_100_100")["throughput_change_pct"] is None


def test_baseline_change_of_baseline_is_zero():
    assert by([row("mps_off")], "mps_off")["throughput_change_pct"] == 0.0


# -- ruling 3/4 --------------------------------------------------------------------------------
def test_decode_throughput_only_has_na_p99():
    rows = [
        row(
            "mps_off",
            thr=400.0,
            p99=None,
            metric_kind="throughput_only",
            workload="decode",
        ),
        row(
            "mps_100_100",
            thr=440.0,
            p99=None,
            metric_kind="throughput_only",
            workload="decode",
        ),
    ]
    c = by(rows, "mps_100_100")
    assert c["throughput_change_pct"] == pytest.approx(10.0)
    assert c["p99_mean"] is None and c["p99_change_pct"] is None
    md = to_markdown(rows)
    assert "decode" in md and "N/A" in md


def test_per_proc_tables_labelled_with_workload_and_no_sum():
    rows = [
        row("mps_off", "A", workload="membound"),
        row("mps_off", "B", workload="decode", p99=None, metric_kind="throughput_only"),
    ]
    md = to_markdown(rows)
    assert "Proc A: membound" in md and "Proc B: decode" in md
    assert "sum" not in md.lower().replace("summary", "")


# -- ruling 5: notes ---------------------------------------------------------------------------
def meta(**kw):
    m = {
        "complete": True,
        "status": "complete",
        "dcgm_available": True,
        "dcgm_reason": None,
        "telemetry_errors": [],
        "contamination_events": [],
        "run_id": "r",
    }
    m.update(kw)
    return m


def notes_of(md):
    return md.split("## Notes", 1)[1]


def test_thermal_note_always_present_with_range():
    rows = [
        row("mps_off", gpu_temp_start_c=40.0),
        row("mps_100_100", gpu_temp_start_c=47.0),
    ]
    n = notes_of(to_markdown(rows, meta()))
    assert "thermal drift" in n and "confounded" in n and "40" in n and "47" in n


def test_clean_run_has_only_thermal_note():
    n = notes_of(to_markdown([row()], meta()))
    for word in (
        "cooldown",
        "dcgm",
        "contamina",
        "incomplete",
        "telemetry",
        "regardless",
    ):
        assert word not in n.lower()


def test_all_applicable_notes():
    rows = [
        row("mps_off", cooldown_timeout=True),
        row(
            "mps_50_50",
            "A",
            client_thread_pct_requested=50,
            mps_server_active_thread_pct=100.0,
        ),
        row(
            "mps_70_30",
            "A",
            client_thread_pct_requested=70,
            mps_server_active_thread_pct=100.0,
        ),
    ]
    m = meta(
        complete=False,
        status="aborted",
        message="stopped",
        dcgm_available=False,
        dcgm_reason="no host engine",
        telemetry_errors=["e1", "e2"],
        contamination_events=[{"x": 1}],
    )
    n = notes_of(to_markdown(rows, m, skipped_lines=2))
    assert (
        "INCOMPLETE" in to_markdown(rows, m).split("## Notes")[0]
    )  # prominent, before tables
    assert "aborted" in n
    assert "cooldown" in n.lower() and "1 repetition" in n
    assert "no host engine" in n
    assert "contamination" in n.lower()
    assert "telemetry" in n.lower() and "2" in n
    assert "server" in n and "100" in n and "regardless" in n
    assert "2 malformed" in n


def test_missing_run_json_is_reported_not_hidden():
    n = notes_of(to_markdown([row()], None))
    assert "run.json" in n and "N/A" in n


def test_requested_and_server_columns_labelled_not_applied():
    rows = [
        row(
            "mps_50_50",
            client_thread_pct_requested=50,
            mps_server_active_thread_pct=100.0,
        )
    ]
    md = to_markdown(rows, meta())
    assert "requested" in md and "server" in md and "applied" not in md.lower()


# -- loading + CLI (ruling 6) -----------------------------------------------------------------
def write_run(tmp_path, rows, extra_lines=(), meta_obj=None):
    d = tmp_path / "res"
    d.mkdir()
    lines = [json.dumps(r) for r in rows] + list(extra_lines)
    (d / "results.jsonl").write_text("\n".join(lines) + "\n")
    if meta_obj is not None:
        (d / "run.json").write_text(json.dumps(meta_obj))
    return d


def test_compare_cli_prints_and_writes_md(tmp_path, capsys):
    d = write_run(
        tmp_path, [row("mps_off"), row("mps_100_100", thr=120.0)], meta_obj=meta()
    )
    out = tmp_path / "report.md"
    assert cli.main(["compare", str(d), "--md", str(out)]) == 0
    stdout = capsys.readouterr().out
    assert "mps_100_100" in stdout and "+20.0" in stdout
    md = out.read_text()
    assert "## Notes" in md and "+20.0" in md


def test_compare_cli_without_md_writes_nothing(tmp_path, capsys):
    d = write_run(tmp_path, [row()], meta_obj=meta())
    assert cli.main(["compare", str(d)]) == 0
    assert not (d / "report.md").exists()


def test_compare_cli_missing_dir_exits_2(tmp_path, capsys):
    assert cli.main(["compare", str(tmp_path / "nope")]) == 2
    assert "nope" in capsys.readouterr().err


def test_compare_cli_missing_results_exits_2(tmp_path, capsys):
    assert cli.main(["compare", str(tmp_path)]) == 2
    assert "results.jsonl" in capsys.readouterr().err


@pytest.mark.parametrize("content", ["not json\n", "", "[1, 2]\n", '{"a": 1}\n'])
def test_compare_cli_unparsable_exits_2(tmp_path, capsys, content):
    (tmp_path / "results.jsonl").write_text(content)
    assert cli.main(["compare", str(tmp_path)]) == 2
    assert capsys.readouterr().err


def test_malformed_lines_skipped_counted_and_noted(tmp_path, capsys):
    d = write_run(
        tmp_path, [row()], extra_lines=["{broken", "", "[1]"], meta_obj=meta()
    )
    rows, skipped = report.load_results(d)
    assert len(rows) == 1 and skipped == 2  # blank line is not a malformed line
    assert cli.main(["compare", str(d), "--md", str(tmp_path / "r.md")]) == 0
    assert "2 malformed" in (tmp_path / "r.md").read_text()


def test_md_write_failure_exits_3(tmp_path, capsys):
    d = write_run(tmp_path, [row()], meta_obj=meta())
    assert (
        cli.main(["compare", str(d), "--md", str(tmp_path / "no" / "dir" / "r.md")])
        == 3
    )


def test_load_meta_bad_json_is_none(tmp_path):
    (tmp_path / "run.json").write_text("{nope")
    assert report.load_meta(tmp_path) is None

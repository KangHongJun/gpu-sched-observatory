import json

from gso import cli
from gso.tui.app import GsoApp
from tests.test_report import row
from tests.test_tui_live import NOW, caps, full_buffer, settle, text


def results_dir(tmp_path):
    rows = [row("mps_off", thr=100.0), row("mps_100_100", thr=120.0)]
    (tmp_path / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    return tmp_path


async def test_f5_shows_comparison_for_bench_dir(tmp_path):
    app = GsoApp(
        full_buffer(), caps(), 1, clock_ns=lambda: NOW, bench_dir=results_dir(tmp_path)
    )
    async with app.run_test(size=(200, 40)) as pilot:
        await pilot.press("f5")
        await settle(pilot)
        t = text(app, "compare-text")
        assert "mps_100_100" in t and "+20.0" in t and "Proc A: small_kernels" in t
        assert (
            app.query_one("#compare-view").display
            and not app.query_one("#metrics-view").display
        )
        await pilot.press("f2")
        await settle(pilot)
        assert (
            app.query_one("#metrics-view").display
            and not app.query_one("#compare-view").display
        )


async def test_f5_unreadable_bench_dir_shows_na(tmp_path):
    app = GsoApp(
        full_buffer(), caps(), 1, clock_ns=lambda: NOW, bench_dir=tmp_path / "nope"
    )
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("f5")
        await settle(pilot)
        assert text(app, "compare-text").startswith("N/A")


def test_live_bench_dir_validated_before_start(tmp_path, capsys):
    import pytest

    with pytest.raises(SystemExit) as e:
        cli.main(["live", "--gpu", "1", "--bench-dir", str(tmp_path / "nope")])
    assert e.value.code == 2
    assert "--bench-dir" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        cli.main(
            [
                "live",
                "--gpu",
                "1",
                "--bench-dir",
                str(tmp_path),
                "--headless",
                "--duration",
                "1",
                "--export",
                str(tmp_path / "x.jsonl"),
            ]
        )
    assert e.value.code == 2


async def test_f5_not_wrapped_at_120_columns_and_scrollable(tmp_path):
    from gso.bench import report

    d = results_dir(tmp_path)
    longest = max(len(line) for line in report.view_text(d).splitlines())
    assert longest > 120  # the case being guarded
    app = GsoApp(full_buffer(), caps(), 1, clock_ns=lambda: NOW, bench_dir=d)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("f5")
        await settle(pilot)
        w = app.query_one("#compare-text")
        view = app.query_one("#compare-view")
        assert w.size.width >= longest  # rows are not wrapped to the terminal width
        assert w.size.height <= len(report.view_text(d).splitlines()) + 1  # one per line
        assert view.max_scroll_x > 0  # the rest is reachable horizontally

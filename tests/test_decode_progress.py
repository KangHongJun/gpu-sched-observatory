from pathlib import Path

from gso.bench.workloads import decode

FIXTURE = Path(__file__).parent / "fixtures" / "ffmpeg_progress_sample.txt"


def test_parse_progress_blocks_from_real_sample():
    blocks = decode.parse_progress(FIXTURE.read_text().splitlines())
    assert len(blocks) >= 2
    assert all("frame" in b and "progress" in b for b in blocks)
    assert blocks[-1]["progress"] == "end"
    frames = [int(b["frame"]) for b in blocks]
    assert frames == sorted(frames) and frames[-1] > 0


def test_fixture_has_no_paths_or_hostnames():
    text = FIXTURE.read_text()
    assert "/" not in text.replace("N/A", "") and "home" not in text


def test_parse_progress_ignores_incomplete_trailing_block():
    blocks = decode.parse_progress(["frame=10", "progress=continue", "frame=20"])
    assert [b["frame"] for b in blocks] == ["10"]


def test_block_point_reports_drop_only_when_present():
    with_drop = decode.parse_progress(["frame=5", "drop_frames=2", "progress=end"])[0]
    without = decode.parse_progress(["frame=5", "progress=end"])[0]
    assert decode.block_point(1.0, with_drop) == (1.0, 5, 2)
    assert decode.block_point(1.0, without) == (1.0, 5, None)


def _points(rate=100, step=0.5, n=20, drop=0):
    return [(i * step, int(i * step * rate), drop) for i in range(n)]


def test_window_start_skips_warmup_seconds():
    pts = _points()
    assert decode.window_start(pts, warmup_s=2.0) == 4
    assert decode.window_start(pts[:3], warmup_s=2.0) is None
    assert decode.window_start([], warmup_s=2.0) is None


def test_window_metrics_fps_and_frames_exclude_warmup():
    pts = _points(rate=100, n=20)  # t 0..9.5
    i0 = decode.window_start(pts, 2.0)
    m = decode.window_metrics(pts, i0)
    assert m["frames"] == pts[-1][1] - pts[i0][1]
    assert m["window_s"] == pts[-1][0] - pts[i0][0]
    assert abs(m["fps"] - 100.0) < 1e-6
    assert m["frames_dropped"] == 0
    assert abs(m["fps_min"] - 100.0) < 1e-6 and abs(m["fps_p5"] - 100.0) < 1e-6


def test_window_metrics_fps_min_sees_a_stall():
    pts = _points(rate=100, n=20)
    # make the interval ending at index 10 slow (only 5 frames in 0.5 s)
    pts = [(t, f if i < 10 else f - 45, d) for i, (t, f, d) in enumerate(pts)]
    m = decode.window_metrics(pts, 4)
    assert abs(m["fps_min"] - 10.0) < 1e-6


def test_window_metrics_few_samples_give_null_percentile():
    pts = _points(n=4)
    m = decode.window_metrics(pts, 0)
    assert m["fps_p5"] is None and m["fps_min"] is None


def test_window_dropped_is_delta_and_null_when_unreported():
    pts = [(0.0, 0, 1), (1.0, 100, 3), (2.0, 200, 4)]
    assert decode.window_metrics(pts, 1)["frames_dropped"] == 1
    pts = [(0.0, 0, None), (1.0, 100, None), (2.0, 200, None)]
    assert decode.window_metrics(pts, 0)["frames_dropped"] is None


def test_window_metrics_needs_two_points():
    assert decode.window_metrics([(0.0, 1, 0)], 0) is None


def test_ffmpeg_command_is_one_continuous_process():
    cmd = decode.ffmpeg_cmd(Path("v.mp4"))
    assert cmd[cmd.index("-stream_loop") + 1] == "-1"
    assert cmd.index("-stream_loop") < cmd.index("-i")
    assert "-hwaccel_device" in cmd and "h264_cuvid" in cmd and "pipe:1" in cmd

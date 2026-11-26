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
    lines = ["frame=10", "progress=continue", "frame=20"]
    blocks = decode.parse_progress(lines)
    assert [b["frame"] for b in blocks] == ["10"]


def test_dropped_reported_in_sample():
    blocks = decode.parse_progress(FIXTURE.read_text().splitlines())
    assert decode.frames_dropped([blocks]) == 0


def test_dropped_none_when_key_missing():
    blocks = decode.parse_progress(["frame=5", "progress=end"])
    assert decode.frames_dropped([blocks]) is None


def test_dropped_sums_last_value_per_run():
    r1 = decode.parse_progress(["frame=5", "drop_frames=1", "progress=continue",
                                "frame=9", "drop_frames=2", "progress=end"])
    r2 = decode.parse_progress(["frame=4", "drop_frames=3", "progress=end"])
    assert decode.frames_dropped([r1, r2]) == 5


def test_samples_per_second_exclude_gaps_between_runs():
    # (wall_s, frame) 구간: run1 0.0->1.0s에 100프레임, 1.0->2.0s에 300프레임
    run1 = [(0.0, 0), (1.0, 100), (2.0, 400)]
    # run2는 3.5초에 시작(재시작 공백 1.5초는 샘플에 안 들어간다)
    run2 = [(3.5, 0), (4.0, 100)]
    samples = decode.fps_samples([run1, run2], bucket_s=1.0)
    # 버킷 [0,1): 100/1.0, [1,2): 300/1.0, [3,4): 100/0.5
    assert [round(f, 1) for f, _ in samples] == [100.0, 300.0, 200.0]
    assert [n for _, n in samples] == [100, 300, 100]


def test_frame_latencies_are_frame_weighted():
    lat = decode.frame_latencies_ms([(100.0, 2), (200.0, 3)])
    assert lat == [10.0, 10.0, 5.0, 5.0, 5.0]

"""NVIDIA 하드웨어 디코드(ffmpeg h264_cuvid) 워크로드. 처리량 전용(throughput_only).

CLI 계약: python -m gso.bench.workloads.decode --gpu N --duration S --out FILE.json
ffmpeg 프로세스 하나가 `-stream_loop -1`로 합성 영상을 계속 디코드하고(재시작·CUDA 컨텍스트 생성이
측정에 들어가지 않는다), 시작 후 WARMUP_S초의 진행 샘플은 버린다. 측정 창은 그 뒤 첫 진행
샘플부터 --duration초이며 끝나면 그 ffmpeg(자신이 띄운 PID)를 종료한다.
프레임은 `-hwaccel_output_format cuda`로 장치 메모리에 둔다(null 출력이라 호스트 복사 없음).

JSON: iterations = 창 안에서 디코드한 프레임 수, throughput_per_s = fps(프레임 / 창 시간),
latency_ms는 p50/p95/p99 모두 null(프레임별 지연을 측정하지 않는다), metric_kind = "throughput_only".
추가 키: fps, fps_min / fps_p5(0.5초 진행 샘플 구간별 FPS의 최솟값 / 하위 5%, 샘플이 5개 미만이면 null),
frames_dropped(창 안 drop_frames 증가량, ffmpeg가 보고하지 않았으면 null).
합성 영상(testsrc2 1920x1080@30, 20초, libx264)은 ~/.cache/gso/에 만들어 재사용한다.
"""

import argparse
import itertools
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from gso.analysis.stats import percentile
from gso.bench.workloads._common import (
    Measured,
    StopFlag,
    ThroughputOnly,
    WorkloadUnavailable,
    file_lock,
    run_cli,
)

WARMUP_S = 2.0
MIN_PCT_SAMPLES = 5
CACHE_DIR = Path("~/.cache/gso").expanduser()
VIDEO_PARAMS = "testsrc2_1920x1080_30fps_20s_h264_v1"


def parse_progress(lines) -> list[dict[str, str]]:
    """`-progress` 출력(key=value 줄, `progress=`로 블록 끝)을 블록 목록으로 바꾼다. 미완성 블록은 버린다."""
    blocks: list[dict[str, str]] = []
    cur: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        cur[key] = value
        if key == "progress":
            blocks.append(cur)
            cur = {}
    return blocks


def block_point(t: float, block: dict[str, str]) -> tuple[float, int, int | None] | None:
    """진행 블록 하나를 (벽시계 s, 누적 프레임, 누적 drop_frames 또는 None)으로."""
    if not block.get("frame", "").isdigit():
        return None
    drop = block.get("drop_frames", "")
    return (t, int(block["frame"]), int(drop) if drop.isdigit() else None)


def window_start(points: list[tuple[float, int, int | None]], warmup_s: float) -> int | None:
    """첫 샘플 시각 + warmup_s 이후의 첫 샘플 인덱스. 아직 없으면 None."""
    if not points:
        return None
    t0 = points[0][0]
    for i, p in enumerate(points):
        if p[0] >= t0 + warmup_s:
            return i
    return None


def window_metrics(points: list[tuple[float, int, int | None]], i0: int) -> dict | None:
    """points[i0:]의 측정 창 지표. 점이 둘 미만이면 None."""
    pts = points[i0:]
    if len(pts) < 2:
        return None
    window_s = pts[-1][0] - pts[0][0]
    frames = pts[-1][1] - pts[0][1]
    if window_s <= 0:
        return None
    rates = [
        (f1 - f0) / (t1 - t0)
        for (t0, f0, _), (t1, f1, _) in itertools.pairwise(pts)
        if t1 > t0
    ]
    enough = len(rates) >= MIN_PCT_SAMPLES
    d0, d1 = pts[0][2], pts[-1][2]
    return {
        "frames": frames,
        "window_s": window_s,
        "fps": frames / window_s,
        "fps_min": min(rates) if enough else None,
        "fps_p5": percentile(rates, 5) if enough else None,
        "frames_dropped": (d1 - d0) if d0 is not None and d1 is not None else None,
    }


def ffmpeg_cmd(video: Path) -> list[str]:
    return [
        "ffmpeg", "-nostdin", "-v", "error", "-nostats", "-progress", "pipe:1",
        "-stream_loop", "-1",
        "-hwaccel", "cuda", "-hwaccel_device", "0", "-hwaccel_output_format", "cuda",
        "-c:v", "h264_cuvid", "-i", str(video), "-f", "null", "-",
    ]


def ensure_video(stop: StopFlag, cache_dir: Path = CACHE_DIR) -> Path:
    path = cache_dir / f"{VIDEO_PARAMS}.mp4"
    if path.exists():
        return path
    note = lambda: print(
        "decode: waiting for another process to generate the test video", file=sys.stderr
    )
    with file_lock(cache_dir / f"{VIDEO_PARAMS}.lock", stop, note):
        if path.exists():
            return path
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.mp4")
        cmd = [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30", "-t", "20",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-g", "60",
            str(tmp),
        ]
        try:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            except FileNotFoundError as exc:
                raise WorkloadUnavailable("ffmpeg not found") from exc
            if r.returncode != 0:
                raise WorkloadUnavailable(f"cannot generate test video: {_last_line(r.stderr)}")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    stop.raise_if_set()  # 영상은 남기고, 중단되었으면 측정으로 넘어가지 않는다
    return path


def _last_line(text: str) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else "no output"


def _reader(stream, q: "queue.Queue[str | None]") -> None:
    for line in stream:
        q.put(line)
    q.put(None)


def _stop_child(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def measure(args: argparse.Namespace, stop: StopFlag) -> Measured | None:
    video = ensure_video(stop)
    stop.raise_if_set()
    try:
        proc = subprocess.Popen(  # 환경(CUDA_DEVICE_ORDER/CUDA_VISIBLE_DEVICES)은 상속
            ffmpeg_cmd(video), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError as exc:
        raise WorkloadUnavailable("ffmpeg not found") from exc
    try:
        q: queue.Queue[str | None] = queue.Queue()
        err_lines: list[str] = []
        threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True).start()
        threading.Thread(
            target=lambda: err_lines.extend(proc.stderr), daemon=True
        ).start()
        start = time.perf_counter()
        deadline = WARMUP_S + args.duration + 15.0  # 진행 샘플이 오지 않을 때의 상한
        pending: list[str] = []
        points: list[tuple[float, int, int | None]] = []
        i0: int | None = None
        while True:
            if stop.is_set():
                return None
            now = time.perf_counter() - start
            if now > deadline:
                raise WorkloadUnavailable("ffmpeg produced no progress in time")
            try:
                line = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                msg = _last_line("".join(err_lines))
                raise WorkloadUnavailable(f"ffmpeg h264_cuvid failed (exit {proc.wait()}): {msg}")
            pending.append(line)
            if not line.startswith("progress="):
                continue
            t = time.perf_counter() - start
            for block in parse_progress(pending):
                pt = block_point(t, block)
                if pt is not None:
                    points.append(pt)
            pending = []
            if i0 is None:
                i0 = window_start(points, WARMUP_S)
            if i0 is not None and points[-1][0] - points[i0][0] >= args.duration:
                break
    finally:
        _stop_child(proc)
    metrics = window_metrics(points, i0)
    if metrics is None:
        raise WorkloadUnavailable("ffmpeg reported no decoded frames")
    extra = {k: metrics[k] for k in ("fps", "fps_min", "fps_p5", "frames_dropped")}
    return ThroughputOnly(metrics["frames"], metrics["window_s"], extra)


def main(argv: list[str] | None = None) -> int:
    return run_cli(
        "decode", "ffmpeg h264_cuvid hardware decode workload", lambda p: None, measure, argv
    )


if __name__ == "__main__":
    raise SystemExit(main())

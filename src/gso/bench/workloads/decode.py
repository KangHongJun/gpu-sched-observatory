"""NVIDIA 하드웨어 디코드(ffmpeg h264_cuvid) 워크로드.

CLI 계약: python -m gso.bench.workloads.decode --gpu N --duration S --out FILE.json
추가 JSON 키: fps(프레임 수 / 경과 시간), frames_dropped(ffmpeg가 drop_frames를 보고했을 때만 값, 아니면 null)

"반복"은 프레임 하나다. 지연(ms)은 1초 구간마다 ffmpeg 진행 상황(-progress)에서 구한 FPS의
역수(1000/fps)이고, 그 구간에서 디코드된 프레임 수만큼 반복해 넣는다(프레임 가중).
따라서 p50/p95/p99는 "1초 평균 프레임 시간"의 분포이지 프레임별 실측이 아니다.
ffmpeg 프로세스 시작 공백은 구간 FPS 계산에서 빠지고, fps 키(전체 프레임 / 경과 시간)에는 포함된다.
합성 영상(testsrc2 1920x1080@30, 20초, libx264)은 ~/.cache/gso/에 만들어 재사용한다.
"""

import argparse
import itertools
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from gso.bench.workloads._common import Measured, StopFlag, WorkloadUnavailable, run_cli

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


def frames_dropped(runs: list[list[dict[str, str]]]) -> int | None:
    """실행(run)마다 마지막으로 보고된 drop_frames의 합. 어느 run에서도 보고되지 않았으면 None."""
    total = 0
    seen = False
    for blocks in runs:
        vals = [int(b["drop_frames"]) for b in blocks if b.get("drop_frames", "").lstrip("-").isdigit()]
        if vals:
            seen = True
            total += vals[-1]
    return total if seen else None


def fps_samples(
    runs: list[list[tuple[float, int]]], bucket_s: float = 1.0
) -> list[tuple[float, int]]:
    """run별 (벽시계 s, 누적 프레임) 점들에서 버킷별 (fps, 프레임 수)를 구한다.

    run 안의 인접한 두 점 사이 구간만 쓴다(run 사이 공백은 제외). 구간은 끝 시각이 속한 버킷에 넣는다.
    """
    acc: dict[int, list[float]] = {}
    for points in runs:
        for (t0, f0), (t1, f1) in itertools.pairwise(points):
            dt = t1 - t0
            if dt <= 0 or f1 < f0:
                continue
            b = int(t1 // bucket_s) if t1 % bucket_s else int(t1 // bucket_s) - 1
            slot = acc.setdefault(max(b, 0), [0.0, 0.0])
            slot[0] += f1 - f0
            slot[1] += dt
    out = []
    for b in sorted(acc):
        n, dt = acc[b]
        if n > 0 and dt > 0:
            out.append((n / dt, int(n)))
    return out


def frame_latencies_ms(samples: list[tuple[float, int]]) -> list[float]:
    return [1000.0 / fps for fps, n in samples for _ in range(n)]


def ensure_video(cache_dir: Path = CACHE_DIR) -> Path:
    path = cache_dir / f"{VIDEO_PARAMS}.mp4"
    if path.exists():
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.mp4")
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30", "-t", "20",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-g", "60",
        str(tmp),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise WorkloadUnavailable("ffmpeg not found") from exc
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise WorkloadUnavailable(f"cannot generate test video: {_last_line(r.stderr)}")
    os.replace(tmp, path)
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
    video = ensure_video()
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-nostats", "-progress", "pipe:1",
        "-hwaccel", "cuda", "-hwaccel_device", "0", "-c:v", "h264_cuvid",
        "-i", str(video), "-f", "null", "-",
    ]
    proc: subprocess.Popen | None = None
    runs_blocks: list[list[dict[str, str]]] = []
    runs_points: list[list[tuple[float, int]]] = []
    total_frames = 0
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < args.duration and not stop.is_set():
            try:
                proc = subprocess.Popen(  # 환경(CUDA_DEVICE_ORDER/CUDA_VISIBLE_DEVICES)은 상속
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
            except FileNotFoundError as exc:
                raise WorkloadUnavailable("ffmpeg not found") from exc
            q: queue.Queue[str | None] = queue.Queue()
            threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True).start()
            err_lines: list[str] = []
            threading.Thread(
                target=lambda p=proc, buf=err_lines: buf.extend(p.stderr), daemon=True
            ).start()
            pending: list[str] = []
            blocks: list[dict[str, str]] = []
            points: list[tuple[float, int]] = []
            prev_frames = 0
            while True:
                if stop.is_set() or time.perf_counter() - start >= args.duration:
                    break
                try:
                    line = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    break
                pending.append(line)
                if line.startswith("progress="):
                    now = time.perf_counter() - start
                    new = parse_progress(pending)
                    pending = []
                    blocks.extend(new)
                    for b in new:
                        if b.get("frame", "").isdigit():
                            f = int(b["frame"])
                            points.append((now, f))
                            total_frames += f - prev_frames
                            prev_frames = f
            _stop_child(proc)
            if stop.is_set():
                return None
            rc = proc.returncode
            finished = blocks and blocks[-1].get("progress") == "end"
            if not finished and time.perf_counter() - start < args.duration:
                msg = _last_line("".join(err_lines))
                raise WorkloadUnavailable(f"ffmpeg h264_cuvid failed (exit {rc}): {msg}")
            runs_blocks.append(blocks)
            runs_points.append(points)
    finally:
        if proc is not None:
            _stop_child(proc)
    elapsed = time.perf_counter() - start
    samples = fps_samples(runs_points)
    if not samples:
        raise WorkloadUnavailable("ffmpeg reported no decoded frames")
    extra = {
        "fps": total_frames / elapsed,
        "frames_dropped": frames_dropped(runs_blocks),
        "frames": total_frames,
    }
    return frame_latencies_ms(samples), elapsed, extra


def main(argv: list[str] | None = None) -> int:
    return run_cli(
        "decode", "ffmpeg h264_cuvid hardware decode workload", lambda p: None, measure, argv
    )


if __name__ == "__main__":
    raise SystemExit(main())

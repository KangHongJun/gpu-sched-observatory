import subprocess
from pathlib import Path

from gso.bench.workloads import _common, decode


def test_ensure_video_builds_once_with_per_pid_tmp(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        out = Path(cmd[-1])
        calls.append(out.name)
        out.write_bytes(b"video")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    stop = _common.StopFlag()
    p1 = decode.ensure_video(stop, tmp_path)
    p2 = decode.ensure_video(stop, tmp_path)
    assert p1 == p2 and p1.exists()
    assert len(calls) == 1
    import os
    assert str(os.getpid()) in calls[0]
    assert sorted(x.name for x in tmp_path.iterdir() if x.suffix != ".lock") == [p1.name]


def test_ensure_video_stopped_raises_interrupted(tmp_path, monkeypatch):
    import pytest

    def fake_run(cmd, **kw):
        Path(cmd[-1]).write_bytes(b"v")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    stop = _common.StopFlag()
    stop.set()
    with pytest.raises(_common.Interrupted):
        decode.ensure_video(stop, tmp_path)

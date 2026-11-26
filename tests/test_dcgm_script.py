import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "verify" / "03_dcgm.sh"


def _run(tmp_path, cmd, gpu=None):
    fake = tmp_path / "docker"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    if gpu is not None:
        env["GSO_DCGM_GPU"] = gpu
    return subprocess.run(["bash", str(SCRIPT), cmd], env=env, capture_output=True,
                          text=True, check=False)


def test_stop_and_status_do_not_need_gpu_choice(tmp_path):
    assert _run(tmp_path, "stop").returncode == 0
    assert _run(tmp_path, "status").returncode == 0


def test_start_and_dmon_require_gpu_choice(tmp_path):
    for cmd in ("start", "dmon"):
        r = _run(tmp_path, cmd)
        assert r.returncode != 0
        assert "GSO_DCGM_GPU" in r.stderr

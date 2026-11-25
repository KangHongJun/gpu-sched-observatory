import pytest

from gso import cli
from gso.buffer import RingBuffer
from gso.capability import Capabilities, DcgmStatus, GpuInfo
from gso.collector import Collector
from gso.tui.app import GsoApp


def caps():
    return Capabilities(
        gpus=[GpuInfo(1, "GPU-test-0001", "Test GPU", "535", False)],
        dcgm=DcgmStatus(False, "down", {}),
        nsys_path=None,
        is_root=False,
        mps_active=False,
    )


def test_gpu_is_required():
    with pytest.raises(SystemExit) as e:
        cli.main(["live"])
    assert e.value.code == 2


@pytest.mark.parametrize(
    "bad",
    [["--interval", "0"], ["--interval", "-1"], ["--dcgm-interval-ms", "50"],
     ["--interval", "nan"], ["--interval", "inf"]],
)
def test_bad_intervals_rejected(bad):
    with pytest.raises(SystemExit) as e:
        cli.main(["live", "--gpu", "1", *bad])
    assert e.value.code == 2


def test_build_live_wires_collector_without_starting(monkeypatch):
    seen = {}

    def fake_detect(**kw):
        seen.update(kw)
        return caps()

    monkeypatch.setattr(cli, "detect", fake_detect)
    args = cli.build_parser().parse_args(
        ["live", "--gpu", "1", "--interval", "0.5", "--dcgm-interval-ms", "200",
         "--dcgmi-cmd", "docker exec gso-dcgm dcgmi", "--dcgm-gpu-index", "0"]
    )
    app = cli.build_live(args)
    assert isinstance(app, GsoApp)
    assert seen == {"dcgmi_cmd": ["docker", "exec", "gso-dcgm", "dcgmi"], "dcgm_gpu_index": 0}
    c = app._collector
    assert isinstance(c, Collector)
    assert c._interval_s == 0.5
    assert c._thread is None  # not started
    assert app._interval_s == 0.5
    assert isinstance(app._buffer, RingBuffer)


def test_buffer_capacity_covers_five_minutes():
    # 6 device + (3 per process, say 8 processes) + 5 dcgm per tick; DCGM at the chosen rate
    assert cli.live_buffer_capacity(1.0, 1000) >= 300 * (6 + 24) + 300 * 5
    assert cli.live_buffer_capacity(1.0, 100) >= 300 * (6 + 24) + 3000 * 5
    assert cli.live_buffer_capacity(0.1, 1000) >= 3000 * (6 + 24) + 300 * 5

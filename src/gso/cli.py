"""gso 명령행 진입점."""

import argparse
import math
import shlex

from gso import __version__
from gso.capability import DCGM_FIELD_IDS, Capabilities, detect

_FIELD_NAMES = {
    1001: "gr_engine_active",
    1002: "sm_active",
    1003: "sm_occupancy",
    1004: "tensor_active",
    1005: "dram_active",
}


def format_doctor(caps: Capabilities) -> str:
    lines = ["GPUs"]
    if caps.nvml_error:
        lines.append(f"  (none: {caps.nvml_error})")
    elif not caps.gpus:
        lines.append("  (none found)")
    else:
        lines.append(f"  {'idx':<4}{'name':<28}{'driver':<14}MIG")
        for g in caps.gpus:
            mig = "on" if g.mig_enabled else "off"
            lines.append(f"  {g.index:<4}{g.name:<28}{g.driver:<14}{mig}")
    d = caps.dcgm
    if d.available:
        lines.append("DCGM: available")
        for fid in DCGM_FIELD_IDS:
            mark = "✓" if d.fields.get(fid) else "✗"
            lines.append(f"  {mark} {fid} {_FIELD_NAMES[fid]}")
    else:
        lines.append(f"DCGM: unavailable - {d.reason}")
    lines.append(f"nsys: {caps.nsys_path or 'N/A'}")
    root = "yes" if caps.is_root else "no (captures mask other users' PIDs)"
    lines.append(f"root: {root}")
    lines.append(f"MPS active: {'yes' if caps.mps_active else 'no'}")
    return "\n".join(lines)


def _positive_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return v


def _dcgm_interval_ms(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if v < 100:
        raise argparse.ArgumentTypeError(f"must be >= 100, got {text}")
    return v


def live_buffer_capacity(interval_s: float, dcgm_interval_ms: int) -> int:
    """Records for at least five minutes: device (6) and up to 8 processes (3 each) per NVML
    tick, plus five DCGM records per DCGM tick."""
    window_s = 300
    nvml_ticks = math.ceil(window_s / interval_s)
    dcgm_ticks = math.ceil(window_s * 1000 / dcgm_interval_ms)
    return nvml_ticks * (6 + 8 * 3) + dcgm_ticks * 5


def build_live(args):
    """Build the live app (collector not started). Separate from run so it is testable."""
    from gso.buffer import RingBuffer
    from gso.collector import Collector
    from gso.tui.app import GsoApp

    dcgmi_cmd = shlex.split(args.dcgmi_cmd)
    caps = detect(dcgmi_cmd=dcgmi_cmd, dcgm_gpu_index=args.dcgm_gpu_index)
    buffer = RingBuffer(live_buffer_capacity(args.interval, args.dcgm_interval_ms))
    collector = Collector(
        buffer,
        caps,
        args.gpu,
        interval_s=args.interval,
        dcgm_interval_ms=args.dcgm_interval_ms,
        dcgmi_cmd=dcgmi_cmd,
        dcgm_gpu_index=args.dcgm_gpu_index,
    )
    return GsoApp(buffer, caps, args.gpu, collector=collector, interval_s=args.interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gso", description="GPU Scheduling Observatory"
    )
    parser.add_argument("--version", action="version", version=f"gso {__version__}")
    sub = parser.add_subparsers(dest="command")
    doc = sub.add_parser("doctor", help="report what can be measured on this machine")
    doc.add_argument(
        "--dcgmi-cmd",
        default="dcgmi",
        help='dcgmi command, e.g. "docker exec gso-dcgm dcgmi"',
    )
    doc.add_argument(
        "--dcgm-gpu-index",
        type=int,
        default=None,
        help="GPU index passed to dcgmi -i (differs from the host index in a container)",
    )
    live = sub.add_parser("live", help="live metrics screen for one GPU")
    live.add_argument("--gpu", type=int, required=True, help="GPU index to watch")
    live.add_argument("--interval", type=_positive_float, default=1.0,
                      help="refresh interval in seconds (default 1.0)")
    live.add_argument("--dcgm-interval-ms", type=_dcgm_interval_ms, default=1000,
                      help="DCGM sampling interval in ms, >= 100 (default 1000)")
    live.add_argument("--dcgmi-cmd", default="dcgmi",
                      help='dcgmi command, e.g. "docker exec gso-dcgm dcgmi"')
    live.add_argument("--dcgm-gpu-index", type=int, default=None,
                      help="GPU index passed to dcgmi -i (differs from the host index in a container)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "live":
        build_live(args).run()
        return 0
    if args.command == "doctor":
        caps = detect(
            dcgmi_cmd=shlex.split(args.dcgmi_cmd), dcgm_gpu_index=args.dcgm_gpu_index
        )
        print(format_doctor(caps))
    return 0

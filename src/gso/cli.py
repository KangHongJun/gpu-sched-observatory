"""gso 명령행 진입점."""

import argparse
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


def main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)
    if args.command == "doctor":
        caps = detect(
            dcgmi_cmd=shlex.split(args.dcgmi_cmd), dcgm_gpu_index=args.dcgm_gpu_index
        )
        print(format_doctor(caps))
    return 0

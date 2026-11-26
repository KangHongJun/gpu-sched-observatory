"""gso 명령행 진입점."""

import argparse
import math
import os
import shlex
import sys
import time
from pathlib import Path

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


def _interval(text: str) -> float:
    v = _positive_float(text)
    if v < MIN_INTERVAL_S:
        raise argparse.ArgumentTypeError(f"must be >= {MIN_INTERVAL_S}, got {text}")
    return v


def _dcgm_interval_ms(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if v < 100:
        raise argparse.ArgumentTypeError(f"must be >= 100, got {text}")
    return v


MIN_INTERVAL_S = 0.1
DRAIN_EVERY_S = 1.0
EXIT_IO_ERROR = 3
EXIT_INTERRUPTED = 130


def live_buffer_capacity(interval_s: float, dcgm_interval_ms: int) -> int:
    """Records for at least five minutes: device (6) and up to 8 processes (3 each) per NVML
    tick, plus five DCGM records per DCGM tick."""
    window_s = 300
    nvml_ticks = math.ceil(window_s / interval_s)
    dcgm_ticks = math.ceil(window_s * 1000 / dcgm_interval_ms)
    return nvml_ticks * (6 + 8 * 3) + dcgm_ticks * 5


def build_collection(args):
    """Detect, create the buffer and the (not started) collector. No Textual import."""
    from gso.buffer import RingBuffer
    from gso.collector import Collector

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
    return buffer, caps, collector


def build_live(args):
    """Build the live app (collector not started). Separate from run so it is testable."""
    from gso.tui.app import GsoApp  # Textual only on the interactive path

    buffer, caps, collector = build_collection(args)
    return GsoApp(buffer, caps, args.gpu, collector=collector, interval_s=args.interval)


def _summary(count: int, path, elapsed_s: float, lost: int, what: str) -> str:
    line = f"gso: wrote {count} records to {path} in {elapsed_s:.1f}s"
    if lost:
        line += f"; {lost} records {what}"
    return line


def run_interactive(args, clock=time.monotonic) -> int:
    from gso.export import write_records

    t0 = clock()
    app = build_live(args)
    try:
        app.run()
    finally:
        app.collector.stop()
    if args.export:
        n = write_records(app.buffer.snapshot(), args.export)
        print(_summary(n, args.export, clock() - t0, app.buffer.dropped,
                       "dropped by the ring buffer (not exported)"), file=sys.stderr)
    return 0


def run_headless(args, sleep=time.sleep, clock=time.monotonic) -> int:
    from gso.export import Drainer, open_stream

    try:
        stream = open_stream(args.export)  # first: an unwritable location fails before detect()
    except OSError as e:
        print(f"gso: cannot write {args.export}: {e}", file=sys.stderr)
        return EXIT_IO_ERROR
    t0 = clock()  # reset below, once detect() is done: the duration counts collection only
    code = 0
    io_error: OSError | None = None
    collector = None
    committed = False
    try:
        try:
            buffer, _caps, collector = build_collection(args)
        except KeyboardInterrupt:
            return EXIT_INTERRUPTED  # nothing collected, nothing to flush
        drainer = Drainer(buffer, stream)
        t0 = clock()
        try:
            collector.start()
            while True:
                remaining = args.duration - (clock() - t0)
                if remaining <= 0:
                    break
                sleep(min(DRAIN_EVERY_S, remaining))
                drainer.drain()
        except KeyboardInterrupt:
            code = EXIT_INTERRUPTED
        except OSError as e:
            io_error = e
        finally:
            collector.stop()
        if io_error is None:
            try:
                drainer.drain()  # whatever the collector pushed up to its stop
            except OSError as e:
                io_error = e
        if io_error is None and stream.count > 0:
            try:
                stream.commit()
                committed = True
            except OSError as e:
                io_error = e
    finally:
        if not committed:
            stream.close()
    shown = args.export if committed or stream.count == 0 else stream.partial_path
    print(_summary(stream.count, shown, clock() - t0, drainer.lost,
                   "dropped by the ring buffer before they could be written"), file=sys.stderr)
    if io_error is not None:
        kept = f"; target not replaced, data kept in {stream.partial_path}" if stream.count else ""
        print(f"gso: I/O error while writing {args.export}: {io_error}{kept}", file=sys.stderr)
    status = collector.status
    errors = False
    for label, e in (("NVML", status.nvml_error), ("DCGM", status.dcgm_error)):
        if e:
            errors = True
            print(f"gso: {label}: {e}", file=sys.stderr)
    if io_error is not None:
        return EXIT_IO_ERROR
    if stream.count == 0 and code == 0:
        why = "see the errors above" if errors else "the collector produced nothing"
        print(f"gso: no records were collected ({why})", file=sys.stderr)
    if code:
        return code
    return 0 if stream.count > 0 else 1


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
    live = sub.add_parser(
        "live",
        help="live metrics screen for one GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit codes of --headless:\n"
        "  0    records written\n"
        "  1    nothing written\n"
        "  2    usage error\n"
        "  3    I/O error while writing (data kept in PATH.partial)\n"
        "  130  interrupted (Ctrl-C; collected records are still written)",
    )
    live.add_argument("--gpu", type=int, required=True, help="GPU index to watch")
    live.add_argument("--interval", type=_interval, default=1.0,
                      help=f"refresh interval in seconds, >= {MIN_INTERVAL_S} (default 1.0)")
    live.add_argument("--dcgm-interval-ms", type=_dcgm_interval_ms, default=1000,
                      help="DCGM sampling interval in ms, >= 100 (default 1000)")
    live.add_argument("--dcgmi-cmd", default="dcgmi",
                      help='dcgmi command, e.g. "docker exec gso-dcgm dcgmi"')
    live.add_argument("--dcgm-gpu-index", type=int, default=None,
                      help="GPU index passed to dcgmi -i (differs from the host index in a container)")
    live.add_argument("--export", metavar="PATH", default=None,
                      help="write metric records to PATH (.jsonl or .csv). Interactive: written when "
                           "the screen closes and limited to the buffer (about the last 5 minutes); "
                           "with --headless: streamed for the whole run via PATH.partial, which "
                           "replaces PATH only when the run ends in an orderly way")
    live.add_argument("--headless", action="store_true",
                      help="no terminal UI: collect for --duration seconds and write --export")
    live.add_argument("--duration", type=_positive_float, default=None, metavar="SECONDS",
                      help="headless run length in seconds (> 0)")
    from gso import cli_bench, cli_capture

    cli_capture.add_parser(sub)
    cli_bench.add_parser(sub)
    return parser


def _validate_live(parser: argparse.ArgumentParser, args) -> None:
    from gso.export import check_export_path

    if args.headless and args.duration is None:
        parser.error("--headless requires --duration")
    if args.headless and not args.export:
        parser.error("--headless requires --export")
    if args.duration is not None and not args.headless:
        parser.error("--duration requires --headless (the interactive screen has no duration)")
    if args.export:
        try:
            check_export_path(args.export)
        except ValueError as e:
            parser.error(str(e))
        target = Path(args.export)
        parent = target.resolve().parent
        if target.is_dir():
            parser.error(f"export path is a directory: {args.export}")
        if not parent.is_dir():
            parser.error(f"directory does not exist: {target.parent}")
        if not os.access(parent, os.W_OK | os.X_OK):
            parser.error(f"directory is not writable: {target.parent}")


def main(argv: list[str] | None = None, *, sleep=time.sleep, clock=time.monotonic) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    tail = None  # everything after "--" is a command to launch (gso capture only)
    if "--" in argv:
        i = argv.index("--")
        argv, tail = argv[:i], argv[i + 1 :]
    args = parser.parse_args(argv)
    if tail is not None and args.command != "capture":
        parser.error("unexpected arguments after '--'")
    if args.command == "capture":
        from gso import cli_capture

        cli_capture.validate(parser, args, tail)
        return cli_capture.run_capture(args)
    if args.command == "bench":
        from gso import cli_bench

        cli_bench.validate(parser, args)
        return cli_bench.run(args)
    if args.command == "live":
        _validate_live(parser, args)
        if args.headless:
            return run_headless(args, sleep=sleep, clock=clock)
        return run_interactive(args, clock=clock)
    if args.command == "doctor":
        caps = detect(
            dcgmi_cmd=shlex.split(args.dcgmi_cmd), dcgm_gpu_index=args.dcgm_gpu_index
        )
        print(format_doctor(caps))
    return 0

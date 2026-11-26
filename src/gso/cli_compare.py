"""``gso compare``: comparison report of an MPS A/B benchmark run directory."""

from __future__ import annotations

import sys
from pathlib import Path

from gso.bench import report

EXIT_USAGE = 2
EXIT_IO_ERROR = 3

_EPILOG = (
    "Reads DIR/results.jsonl (and DIR/run.json) written by `gso bench`. Only repetitions that\n"
    "are ok and fair enter the statistics; excluded ones are counted. Unknown values are N/A.\n"
    "exit codes: 0 report printed  2 usage error (missing DIR or results.jsonl, nothing parsable)\n"
    "            3 cannot write --md"
)


def add_parser(sub) -> None:
    import argparse

    p = sub.add_parser(
        "compare",
        help="comparison report of a benchmark run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    p.add_argument(
        "dir", metavar="DIR", help="directory with results.jsonl from `gso bench`"
    )
    p.add_argument(
        "--md",
        metavar="PATH",
        default=None,
        help="also write the report as markdown to PATH",
    )


def run(args) -> int:
    try:
        rows, skipped = report.load_results(args.dir)
    except report.ReportError as e:
        print(f"gso compare: {e}", file=sys.stderr)
        return EXIT_USAGE
    meta = report.load_meta(args.dir)
    print(report.to_text(rows, meta, skipped), end="")
    if args.md:
        try:
            Path(args.md).write_text(
                report.to_markdown(rows, meta, skipped), encoding="utf-8"
            )
        except OSError as e:
            print(f"gso compare: cannot write {args.md}: {e}", file=sys.stderr)
            return EXIT_IO_ERROR
        print(f"gso compare: wrote {args.md}", file=sys.stderr)
    return 0

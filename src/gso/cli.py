"""gso 명령행 진입점. 하위 명령은 이후 Task에서 추가한다."""

import argparse

from gso import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gso", description="GPU Scheduling Observatory"
    )
    parser.add_argument("--version", action="version", version=f"gso {__version__}")
    parser.parse_args(argv)
    return 0

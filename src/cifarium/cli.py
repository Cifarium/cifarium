"""Command line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cifarium",
        description="Find the smallest model that handles your examples right in the browser.",
    )
    parser.add_argument("--version", action="version", version=f"cifarium {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    smoke = commands.add_parser(
        "smoke",
        help="temporary check: load a small model in Chrome and compute vectors",
        description=(
            "Temporary end-to-end check. Downloads the engine and all-MiniLM-L6-v2 into "
            "the cache, starts the local server and the installed Chrome, computes the "
            "vectors of 20 English sentences on the processor and prints the result."
        ),
    )
    smoke.add_argument(
        "--headed", action="store_true", help="open a Chrome window instead of running without one"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "smoke":
        # Imported here so that `cifarium --help` does not load Playwright and numpy.
        from . import smoke

        return smoke.run(headed=args.headed)
    parser.print_help()
    return 0

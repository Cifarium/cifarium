"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__

STARTER_DIR = Path(__file__).resolve().parent / "starter"
STARTER_FILES = ("task.yaml", "examples.jsonl")


def run_init(folder: Path) -> int:
    """Write the starter settings and examples into ``folder``. Never overwrites."""
    existing = [name for name in STARTER_FILES if (folder / name).exists()]
    if existing:
        print(
            f"Nothing written: {' and '.join(existing)} already in {folder}. "
            "cifarium init never overwrites files.",
            file=sys.stderr,
        )
        return 1
    for name in STARTER_FILES:
        try:
            with open(folder / name, "xb") as file:  # "x" fails instead of overwriting
                file.write((STARTER_DIR / name).read_bytes())
        except FileExistsError:
            print(f"Stopped: {name} appeared in {folder} while writing.", file=sys.stderr)
            return 1
    print(f"Wrote {' and '.join(STARTER_FILES)} to {folder}")
    print("Next: put your own examples into examples.jsonl, one per line,")
    print("and list your labels in task.yaml.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cifarium",
        description="Find the smallest model that handles your examples right in the browser.",
    )
    parser.add_argument("--version", action="version", version=f"cifarium {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    commands.add_parser(
        "init",
        help="write a starter task.yaml and examples.jsonl into the current folder",
        description=(
            "Write a starter task.yaml and examples.jsonl for a classification task into "
            "the current folder. Writes nothing if either file is already there."
        ),
    )

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
    if args.command == "init":
        return run_init(Path.cwd())
    if args.command == "smoke":
        # Imported here so that `cifarium --help` does not load Playwright and numpy.
        from . import smoke

        return smoke.run(headed=args.headed)
    parser.print_help()
    return 0

"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

from . import __version__, exitcodes
from .config import DEFAULT_FILE
from .examples import DEFAULT_FILE as EXAMPLES_FILE

STARTER_DIR = Path(__file__).resolve().parent / "starter"
STARTER_FILES = (DEFAULT_FILE, EXAMPLES_FILE)

EXIT_CODES_HELP = (
    "Exit codes: 0 done, 1 no model fits, 2 the settings, the examples or the command line are "
    "wrong, 3 the environment failed (Chrome, the network, the disk)."
)


def run_init(folder: Path) -> int:
    """Write the starter settings and examples into ``folder``. Never overwrites.

    Either both files are written or none: a file written before a failure is removed. Exit code
    (:mod:`exitcodes`): 0 written; 2 a file is already there; 3 the folder cannot be written.
    """
    existing = [name for name in STARTER_FILES if (folder / name).exists()]
    if existing:
        print(
            f"Nothing written: {' and '.join(existing)} already in {folder}. "
            "cifarium init never overwrites files.",
            file=sys.stderr,
        )
        return exitcodes.BAD_INPUT
    written: list[Path] = []
    for name in STARTER_FILES:
        target = folder / name
        try:
            with open(target, "xb") as file:  # "x" fails instead of overwriting
                written.append(target)
                file.write((STARTER_DIR / name).read_bytes())
        except OSError as err:
            for path in written:
                path.unlink(missing_ok=True)
            if isinstance(err, FileExistsError):
                reason = f"{name} appeared while writing"
            else:
                reason = f"cannot write {name}: {err.strerror or err}"
            print(f"Nothing written to {folder}: {reason}.", file=sys.stderr)
            return exitcodes.ENVIRONMENT
    print(f"Wrote {' and '.join(STARTER_FILES)} to {folder}")
    print(f"Next: put your own examples into {EXAMPLES_FILE}, one per line,")
    print(f"at least two for every label, and list your labels in {DEFAULT_FILE}.")
    return exitcodes.DONE


def _guarded(command: Callable[[], int]) -> int:
    """Run a command; a failure nobody planned for is printed with its traceback and exits with
    the code of the environment, so that 1 keeps meaning only "no model fits"."""
    try:
        return command()
    except Exception:  # noqa: BLE001 - this is the last place that can keep the exit codes true
        traceback.print_exc()
        print(
            "\nFAILED: cifarium stopped on something it did not expect (the traceback is above). "
            "Please report it at https://github.com/Cifarium/cifarium/issues",
            file=sys.stderr,
        )
        return exitcodes.ENVIRONMENT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cifarium",
        description="Find the smallest model that handles your examples right in the browser.",
        epilog=EXIT_CODES_HELP,
    )
    parser.add_argument("--version", action="version", version=f"cifarium {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    commands.add_parser(
        "init",
        help=f"write a starter {DEFAULT_FILE} and {EXAMPLES_FILE} into the current folder",
        description=(
            f"Write a starter {DEFAULT_FILE} and {EXAMPLES_FILE} for a classification task "
            "into the current folder. Writes nothing if either file is already there. "
            "Exit code 0 when both files were written, 2 when a file is already there, 3 when the "
            "folder cannot be written."
        ),
    )

    build = commands.add_parser(
        "build",
        help="find the lightest model that gets your examples right in the browser",
        description=(
            "Read the settings and the examples, download the engine and the candidate models "
            "into the cache, run them in the installed Chrome from the lightest, and print why "
            "each one was or was not taken and which model was chosen. Then write the folder for "
            "the site to the `output` of the settings. " + EXIT_CODES_HELP
        ),
    )
    build.add_argument(
        "settings", nargs="?", default=DEFAULT_FILE, help=f"the settings file (default: {DEFAULT_FILE})"
    )
    build.add_argument(
        "--headed", action="store_true", help="open a Chrome window instead of running without one"
    )

    # A service command for people who maintain the catalog. No `help=` on purpose: with it
    # argparse lists the command in `cifarium --help`, and `help=SUPPRESS` prints "==SUPPRESS==".
    update = commands.add_parser(
        "catalog-update",
        description=(
            "Compare every model catalog entry with Hugging Face and show what changed: "
            "version, file sizes, dtype, the license in the cards, closed access. Reads the "
            "Hugging Face API only and downloads no model files. By default writes nothing. "
            "Exit code 0 when the catalog is up to date, 1 when something differs or failed."
        ),
    )
    update.add_argument(
        "--write",
        action="store_true",
        help="rewrite revision, common_files and variants of the changed entries; "
        "license, license_source, languages and recipes are never touched",
    )
    update.add_argument(
        "--only", action="append", default=[], metavar="ID", help="check only this entry; may be repeated"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        return _guarded(lambda: run_init(Path.cwd()))
    if args.command == "build":
        # Imported here so that `cifarium --help` does not load Playwright and numpy.
        def build_command() -> int:
            from . import build

            return build.run(args.settings, headed=args.headed)

        return _guarded(build_command)
    if args.command == "catalog-update":
        from . import catalog_update

        return catalog_update.run(write=args.write, only=args.only)
    parser.print_help()
    return 0

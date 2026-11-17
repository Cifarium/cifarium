"""The invariant of the build folder: everything in it may be published, so none of it holds the
text of an example (ARCHITECTURE.md, section 7).

:func:`example_leaks` looks for the texts the way a leak would show up: whole, in the escapes of
JSON (a manifest or a report would write them so), in any case, and as a cut piece of
:data:`WINDOW` characters, because a file that keeps "the first 40 characters" is a leak too.
``vendor/`` and ``models/`` are third-party files pinned by hash or size and copied as they are,
so they are not searched: a model's vocabulary holds words, and a minified library holds
sentences, that an example may contain by chance.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

WINDOW = 20
"""Characters of an example that, found in a file, count as a leak."""

THIRD_PARTY = ("vendor/", "models/")


def needles(text: str) -> set[str]:
    """The lower-case forms of ``text`` to look for in a file."""
    text = text.strip()
    pieces = {text}
    if len(text) > WINDOW:
        pieces.update(text[start:start + WINDOW] for start in range(len(text) - WINDOW + 1))
    found: set[str] = set()
    for piece in pieces:
        for form in (piece, json.dumps(piece)[1:-1], json.dumps(piece, ensure_ascii=False)[1:-1]):
            found.add(form.lower())
    return {form for form in found if form}


def example_leaks(folder: Path, texts: Iterable[str]) -> list[str]:
    """``"<file>: <excerpt>"`` for every file outside the third-party ones that holds a text."""
    wanted = {form: text for text in texts for form in needles(text)}
    leaks = []
    for file in sorted(Path(folder).rglob("*")):
        name = file.relative_to(folder).as_posix()
        if not file.is_file() or name.startswith(THIRD_PARTY):
            continue
        content = file.read_bytes().decode("utf-8", errors="ignore").lower()
        for form, text in wanted.items():
            if form in content:
                leaks.append(f"{name}: {form!r} of {text[:40]!r}")
                break
    return leaks

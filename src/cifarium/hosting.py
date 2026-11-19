"""The three hosting files of a build folder, read the way their servers read them.

``netlify``, ``vercel`` and ``nginx`` give the headers a server with that file sends for a path
(``/vendor/x.js``). The first two follow the documented rules for the patterns the files use, the
third picks a ``location`` the way nginx does (exact, then regular expression or the longest
prefix). :func:`headers_of` reads the file of a folder and returns such a function; ``check`` hands
it to :class:`server.SiteServer`, so the page of the check gets the headers of the file and nothing
else, and a test can see what a browser does with them (ARCHITECTURE.md, sections 1.6 and 9).

The readers know the patterns of the files cifarium writes (``bundle.HOSTING_FILES``), not every
feature of the three servers: a file the developer rewrote with other rules may be read wrongly,
and the hosts themselves were never run (section 7).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

HeadersOf = Callable[[str], dict[str, str]]
"""The headers a host sends for a path such as ``/vendor/x.js``."""

ISOLATION = {"Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp"}
"""The two headers that make a page ``crossOriginIsolated`` (section 1.6)."""

FOLDER = "hosting"
"""Folder of the build folder that holds the three files."""

FILES = {"netlify": "_headers", "vercel": "vercel.json", "nginx": "nginx.conf"}
"""Which hosting file belongs to which name. ``_headers`` is read by Netlify and Cloudflare Pages."""

DEFAULT = "netlify"

PROBES = ("/", "/index.html", "/cifarium-worker.js", "/vendor/x.js", "/models/x", "/cifarium.json")
"""Paths a file is read at once, to learn that it cannot be read before a page depends on it."""


class HostingError(ValueError):
    """A hosting file is missing or cannot be read as its server reads it."""


def netlify(text: str, path: str) -> dict[str, str]:
    """``_headers``: a path is a line without indent, its headers follow with indent. ``*`` at
    the end takes everything below; the headers of every matching rule add up."""
    rules: list[tuple[str, dict[str, str]]] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0] in " \t":
            name, _, value = line.strip().partition(":")
            rules[-1][1][name.strip()] = value.strip()
        else:
            rules.append((line.strip(), {}))
    headers: dict[str, str] = {}
    for pattern, values in rules:
        matches = path.startswith(pattern[:-1]) if pattern.endswith("*") else path == pattern
        if matches:
            headers.update(values)
    return headers


def vercel(text: str, path: str) -> dict[str, str]:
    """``vercel.json``: every ``source`` is a path or a pattern with a group, like ``/vendor/(.*)``."""
    headers: dict[str, str] = {}
    for rule in json.loads(text)["headers"]:
        source = rule["source"]
        matches = re.fullmatch(source, path) is not None if "(" in source else source == path
        if matches:
            headers.update({item["key"]: item["value"] for item in rule["headers"]})
    return headers


def nginx(text: str, path: str) -> dict[str, str]:
    """``nginx.conf``: the ``add_header`` lines of the one location that serves ``path``."""
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    locations: list[tuple[str, str, dict[str, str]]] = []
    for found in re.finditer(r"location\s+(?:(=|~)\s+)?(\S+)\s*\{", text):
        depth, end = 1, found.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            end += 1
        body = text[found.end():end]
        headers = {name: value for name, value in re.findall(r'add_header\s+(\S+)\s+"([^"]*)"\s+always;', body)}
        locations.append((found.group(1) or "", found.group(2), headers))
    for kind, pattern, headers in locations:
        if kind == "=" and pattern == path:
            return headers
    prefixes = [item for item in locations if item[0] == "" and path.startswith(item[1])]
    longest = max(prefixes, key=lambda item: len(item[1]), default=None)
    regular = next((item for item in locations if item[0] == "~" and re.search(item[1], path)), None)
    chosen = regular or longest
    return chosen[2] if chosen else {}


READERS: dict[str, Callable[[str, str], dict[str, str]]] = {"netlify": netlify, "vercel": vercel, "nginx": nginx}


def headers_of(folder: str | Path, hosting: str = DEFAULT) -> HeadersOf:
    """The headers a host would send for a path, by the file ``hosting`` of ``<folder>/hosting/``.

    ``hosting`` is a key of :data:`FILES`. Raises :class:`HostingError` when it is not, when the
    file is not in the folder, is not text, or cannot be read by its reader.
    """
    if hosting not in FILES:
        raise HostingError(f"unknown hosting {hosting!r}; the files are {', '.join(sorted(FILES))}")
    name = FILES[hosting]
    file = Path(folder) / FOLDER / name
    try:
        text = file.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HostingError(f"{FOLDER}/{name} is not in {folder}: nothing says what headers the host sends") from None
    except (OSError, UnicodeDecodeError) as error:
        raise HostingError(f"{FOLDER}/{name} cannot be read: {error}") from error
    reader = READERS[hosting]
    try:
        for probe in PROBES:
            reader(text, probe)
    except (ValueError, KeyError, IndexError, TypeError, AttributeError, re.error) as error:
        raise HostingError(f"{FOLDER}/{name} cannot be read the way {hosting} reads it: {error!r}") from error
    return lambda path: reader(text, path)

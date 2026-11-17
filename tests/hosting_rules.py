"""The three hosting files of a build folder, read the way their servers read them.

``netlify``, ``vercel`` and ``nginx`` give the headers a server with that file sends for a path
(``/vendor/x.js``); the first two follow the documented rules for the patterns the files use,
the third picks a ``location`` the way nginx does (exact, then regular expression or the
longest prefix). :class:`RuleServer` serves a folder with such a function, so a test can see
what a browser does with the headers of the file and nothing else.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from cifarium.server import mime_type

HeadersOf = Callable[[str], dict[str, str]]

ISOLATION = {"Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp"}


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


class RuleServer:
    """Serves a folder with the headers ``headers_of(path)`` names for each file, and no others."""

    def __init__(self, folder: Path, headers_of: HeadersOf) -> None:
        root = Path(folder).resolve()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                path = unquote(urlsplit(self.path).path)
                target = (root / (path.lstrip("/") or "index.html")).resolve()
                if not target.is_relative_to(root) or not target.is_file():
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mime_type(target.name))
                self.send_header("Content-Length", str(len(data)))
                for name, value in headers_of(path).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05},
                                        daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def __enter__(self) -> RuleServer:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._httpd.shutdown()
        self._thread.join()
        self._httpd.server_close()

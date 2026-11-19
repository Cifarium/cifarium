"""Local static servers: one for the bench page, one for a build folder.

:class:`BenchServer` gives every response the isolation headers and ``Cache-Control: no-store``,
and content types come from an explicit table instead of the system one (ARCHITECTURE.md, section 5).
On Windows the system table is the registry, where ``.js`` is sometimes ``text/plain``; a browser
refuses to run such a module.

:class:`SiteServer` is what ``check`` serves a build folder with: the folder at the root, as a site
has it, and the headers that a hosting file names for each path and no others (section 9).
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

#: Sent with every response of the bench server, errors included.
COMMON_HEADERS = {
    # Together these two make the page crossOriginIsolated, which lets
    # ONNX Runtime use several threads.
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    # The HTTP cache must not distort the measurement of a first load.
    "Cache-Control": "no-store",
}

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".json": "application/json",
    ".wasm": "application/wasm",
    ".onnx": "application/octet-stream",
    ".onnx_data": "application/octet-stream",
    ".bin": "application/octet-stream",
}
DEFAULT_MIME_TYPE = "application/octet-stream"

Locate = Callable[[str], "Path | None"]
"""A request target to the file that answers it, or None for 404."""

HeadersOf = Callable[[str], Mapping[str, str]]
"""The path of a request (``/vendor/x.js``) to the headers its response carries."""


def mime_type(filename: str) -> str:
    return MIME_TYPES.get(os.path.splitext(filename)[1].lower(), DEFAULT_MIME_TYPE)


def resolve(routes: Mapping[str, Path], target: str) -> Path | None:
    """Map a request target to a file inside one of the served folders.

    ``/vendor/4.3.2/x.js`` is looked up as ``routes["vendor"] / "4.3.2" / "x.js"``.
    A target that ends with ``/`` means ``index.html``. Returns ``None`` when
    there is no such file or the target tries to leave the folder.
    """
    segments = unquote(urlsplit(target).path).split("/")
    if len(segments) < 3 or segments[0] != "":
        return None
    prefix, *rest = segments[1:]
    root = routes.get(prefix)
    if root is None:
        return None
    return _file_in(root, rest)


def resolve_site(folder: Path, target: str) -> Path | None:
    """Map a request target to a file of ``folder`` served at the root of a site.

    ``/`` is ``index.html`` and ``/vendor/4.3.2/x.js`` is ``folder / "vendor" / "4.3.2" / "x.js"``.
    ``None`` when there is no such file or the target tries to leave the folder.
    """
    segments = unquote(urlsplit(target).path).split("/")
    if segments[0] != "":
        return None
    return _file_in(folder, segments[1:])


def _file_in(root: Path, rest: list[str]) -> Path | None:
    if rest[-1] == "":
        rest[-1] = "index.html"
    if not all(_is_plain_name(segment) for segment in rest):
        return None
    try:
        file = root.joinpath(*rest).resolve(strict=True)
        inside = file.is_relative_to(root.resolve(strict=True))
    except OSError:
        return None
    return file if inside and file.is_file() else None


def _is_plain_name(segment: str) -> bool:
    """True for an ordinary file or folder name: no ``..``, dotfiles or separators."""
    return bool(segment) and not segment.startswith(".") and not any(c in segment for c in "\\:\0")


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # With port 0 the system picks a free port; address reuse would only allow
    # a second process to bind the same one on Windows.
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], locate: Locate, headers_of: HeadersOf):
        super().__init__(address, _Handler)
        self.locate = locate
        self.headers_of = headers_of
        self.requests: list[tuple[str, str, int]] = []

    def handle_error(self, request, client_address) -> None:
        # A closed tab drops its connections; that is not worth a traceback.
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    protocol_version = "HTTP/1.1"
    server_version = "cifarium"
    sys_version = ""

    def do_GET(self) -> None:
        self._serve(with_body=True)

    def do_HEAD(self) -> None:
        self._serve(with_body=False)

    def _serve(self, *, with_body: bool) -> None:
        file = self.server.locate(self.path)
        try:
            stream = open(file, "rb") if file else None
        except OSError:
            stream = None
        if stream is None:
            body = b"Not found\n"
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if with_body:
                self.wfile.write(body)
            return
        with stream:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type(file.name))
            self.send_header("Content-Length", str(os.fstat(stream.fileno()).st_size))
            self.end_headers()
            if with_body:
                try:
                    shutil.copyfileobj(stream, self.wfile, 1 << 20)
                except ConnectionError:
                    self.close_connection = True

    def end_headers(self) -> None:
        # A request that did not parse has no path yet.
        path = unquote(urlsplit(getattr(self, "path", "")).path)
        for name, value in self.server.headers_of(path).items():
            self.send_header(name, value)
        super().end_headers()

    def log_request(self, code="-", size="-") -> None:
        status = code.value if isinstance(code, HTTPStatus) else code
        if isinstance(status, int):
            self.server.requests.append((self.command, self.path, status))

    def log_message(self, format: str, *args) -> None:
        pass  # keep the terminal for the command's own output


class _Hosted:
    """A server in a thread of its own on 127.0.0.1 and a random free port."""

    def __init__(self, locate: Locate, headers_of: HeadersOf):
        self._httpd = _Server(("127.0.0.1", 0), locate, headers_of)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def requests(self) -> list[tuple[str, str, int]]:
        """Every request answered so far: (method, target, status)."""
        return list(self._httpd.requests)

    def start(self):
        if self._thread is None:
            # serve_forever notices a shutdown request once per poll interval;
            # the default of 0.5 s would make every stop() wait that long.
            self._thread = threading.Thread(
                target=self._httpd.serve_forever,
                kwargs={"poll_interval": 0.05},
                name="cifarium-server",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join()
            self._thread = None
        self._httpd.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()


class BenchServer(_Hosted):
    """Serves folders under URL prefixes, every response with the isolation headers.

    ``routes`` maps the first path segment to a folder::

        with BenchServer({"bench": bench_dir, "models": models_dir}) as server:
            print(server.url)  # http://127.0.0.1:<port>
    """

    def __init__(self, routes: Mapping[str, Path]):
        roots = {prefix: Path(root) for prefix, root in routes.items()}
        super().__init__(lambda target: resolve(roots, target), lambda path: COMMON_HEADERS)


class SiteServer(_Hosted):
    """Serves a folder at the root of a site with the headers ``headers_of`` names for each path.

    What a host does: a file of the folder at its own path, ``/`` as ``index.html``, the headers of
    the hosting file for the path and no others, not even ``Cache-Control``. ``replace`` puts other
    files at some paths (``{"/": page, "/index.html": page}``), and the folder is not touched::

        with SiteServer(folder, hosting.headers_of(folder), replace={"/": page}) as server:
            print(server.url)  # http://127.0.0.1:<port>
    """

    def __init__(self, folder: str | os.PathLike[str], headers_of: HeadersOf, *,
                 replace: Mapping[str, str | os.PathLike[str]] | None = None):
        root = Path(folder)
        swaps = {path: Path(file) for path, file in (replace or {}).items()}

        def locate(target: str) -> Path | None:
            swapped = swaps.get(unquote(urlsplit(target).path))
            if swapped is not None:
                return swapped if swapped.is_file() else None
            return resolve_site(root, target)

        super().__init__(locate, headers_of)

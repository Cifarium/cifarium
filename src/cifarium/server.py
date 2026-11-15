"""Local static server for the bench page.

Every response carries the cross-origin isolation headers and
``Cache-Control: no-store``, and content types come from an explicit table
instead of the system one (ARCHITECTURE.md, section 5). On Windows the system
table is the registry, where ``.js`` is sometimes ``text/plain``; a browser
refuses to run such a module.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

#: Sent with every response, errors included.
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

    def __init__(self, address: tuple[str, int], routes: dict[str, Path]):
        super().__init__(address, _Handler)
        self.routes = routes
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
        file = resolve(self.server.routes, self.path)
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
        for name, value in COMMON_HEADERS.items():
            self.send_header(name, value)
        super().end_headers()

    def log_request(self, code="-", size="-") -> None:
        status = code.value if isinstance(code, HTTPStatus) else code
        if isinstance(status, int):
            self.server.requests.append((self.command, self.path, status))

    def log_message(self, format: str, *args) -> None:
        pass  # keep the terminal for the bench's own output


class BenchServer:
    """Serves folders under URL prefixes on 127.0.0.1 and a random free port.

    ``routes`` maps the first path segment to a folder::

        with BenchServer({"bench": bench_dir, "models": models_dir}) as server:
            print(server.url)  # http://127.0.0.1:<port>
    """

    def __init__(self, routes: Mapping[str, Path]):
        self._httpd = _Server(("127.0.0.1", 0), {prefix: Path(root) for prefix, root in routes.items()})
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

    def start(self) -> BenchServer:
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

    def __enter__(self) -> BenchServer:
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

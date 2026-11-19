"""The example page of a build in the real Chrome: it loads the model of its folder and names the
label of a text, behind each hosting file's headers and behind a server with none.

The folder is the one `cifarium build` wrote for the English set (tests/conftest.py). The
servers apply the headers of `hosting/_headers`, `vercel.json` or `nginx.conf` as those hosts
would, and nothing else (`cifarium.hosting`, `server.SiteServer`); the last one is `python -m http.server`, the
command the build prints for looking at the folder. Runs only with `pytest -m e2e`; it downloads
the same files as tests/test_build_e2e.py.
"""

from __future__ import annotations

import json
import math
import re
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from test_build_e2e import ENGLISH

from cifarium import bundle, hosting
from cifarium.bench import Bench, open_bench
from cifarium.server import SiteServer

pytestmark = pytest.mark.e2e

TEXT, LABEL = ENGLISH[0]
"""A text of the examples and its label: the nearest centre is its own."""


@pytest.fixture(scope="module")
def bench() -> Iterator[Bench]:
    with open_bench() as opened:
        yield opened


@pytest.fixture(scope="module")
def folder(english_build) -> Path:
    assert english_build.code == 0, english_build.err
    return english_build.folder


def use_page(bench: Bench, url: str, text: str = TEXT) -> dict[str, Any]:
    """Click the page's one button to load the model, type ``text``, click again; what it shows."""
    page = bench.context.new_page()
    events: list[str] = []
    page.on("console", lambda message: events.append(f"console {message.type}: {message.text}"))
    page.on("pageerror", lambda error: events.append(f"script error: {error}"))
    try:
        page.goto(url)
        page.wait_for_function("() => !document.getElementById('go').disabled", timeout=30_000)
        assert page.inner_text("#go") == "Load the model"
        page.click("#go")
        page.wait_for_function("() => ['ready', 'failed'].includes(document.body.dataset.state)",
                               timeout=120_000)
        if page.evaluate("document.body.dataset.state") == "failed":
            raise AssertionError(f"the page failed: {page.inner_text('#error')}\n" + "\n".join(events[-10:]))
        assert page.inner_text("#go") == "Name the label"
        assert page.is_disabled("#go"), "nothing typed yet: the button waits"
        page.fill("#text", text)
        page.click("#go")
        page.wait_for_function("() => document.getElementById('result').dataset.label", timeout=30_000)
        return {
            "label": page.get_attribute("#result", "data-label"),
            "shown": page.inner_text("#result"),
            "scores": page.inner_text("#scores"),
            "info": page.text_content("#info"),
            "isolated": page.evaluate("crossOriginIsolated"),
            "error_hidden": page.is_hidden("#error"),
        }
    finally:
        page.close()


def threads_of(info: str) -> int:
    match = re.search(r"device\s+\S+, (\d+) threads?", info)
    assert match, info
    return int(match.group(1))


def expected_threads(bench: Bench) -> int:
    # What ONNX Runtime picks on an isolated page: half the cores, at most four.
    return min(4, math.ceil((bench.device["hardwareConcurrency"] or 1) / 2))


def check_page(shown: dict[str, Any], folder: Path) -> None:
    assert shown["label"] == LABEL
    assert re.fullmatch(rf"{LABEL} \(\d\.\d\d\)", shown["shown"]), shown["shown"]
    # The page reads `scores` of the answer: every label, the best first.
    listed = re.findall(r"(\w+) (-?\d\.\d\d)", shown["scores"])
    values = [float(value) for _, value in listed]
    assert sorted(name for name, _ in listed) == ["returns", "shipping", "technical"]
    assert listed[0][0] == LABEL and values == sorted(values, reverse=True)
    assert listed[0][1] == shown["shown"].split("(")[1].rstrip(")")
    assert shown["error_hidden"]
    manifest = json.loads((folder / "cifarium.json").read_text(encoding="utf-8"))
    info = shown["info"]
    assert "Xenova/all-MiniLM-L6-v2 (apache-2.0)" in info
    assert "labels        shipping, returns, technical" in info
    weight = float(re.search(r"first visit\s+([\d.]+) MB", info).group(1))
    assert abs(weight - bundle.first_load(manifest, "cpu")[0] / 1e6) < 0.06


@pytest.mark.parametrize("name", sorted(hosting.FILES))
def test_with_the_headers_of_a_hosting_file_alone_the_page_is_isolated_and_uses_threads(
        bench, folder, name):
    with SiteServer(folder, hosting.headers_of(folder, name)) as server:
        shown = use_page(bench, server.url + "/")

    check_page(shown, folder)
    assert shown["isolated"] is True
    assert threads_of(shown["info"]) == expected_threads(bench)
    assert "not cross-origin isolated" not in shown["info"]


def test_without_headers_the_page_still_works_in_one_thread_and_says_why(bench, folder):
    with SiteServer(folder, lambda path: {}) as server:
        shown = use_page(bench, server.url + "/")

    check_page(shown, folder)
    assert shown["isolated"] is False
    assert threads_of(shown["info"]) == 1
    assert "The page is not cross-origin isolated, so the model computes in one thread." in shown["info"]


def test_the_page_and_its_thread_ask_only_for_files_of_the_folder(bench, folder):
    # Section 14: nothing is sent anywhere. Every request of the page and of the thread it starts
    # (the browser reports both) goes to the server the folder is on, is a GET, and asks for a
    # file of the folder; the one thing besides is the icon the browser asks every page for.
    seen: list[tuple[str, str]] = []

    def note(request) -> None:
        # The antivirus of the author's laptop (Kaspersky) writes a script of its own into every
        # page served over plain http, and the browser reports that script's requests. They are
        # not the page's and not the thread's, and there are none on GitHub Actions.
        if not urlsplit(request.url).hostname.endswith(".kaspersky-labs.com"):
            seen.append((request.method, request.url))

    bench.context.on("request", note)
    try:
        with SiteServer(folder, hosting.headers_of(folder, "netlify")) as server:
            use_page(bench, server.url + "/")
            origin = server.url
    finally:
        bench.context.remove_listener("request", note)

    asked = {url.removeprefix(origin) for _, url in seen}
    assert all(url.startswith(origin + "/") for _, url in seen), [url for _, url in seen if not url.startswith(origin)]
    assert {method for method, _ in seen} == {"GET"}
    strangers = sorted(path for path in asked if path not in ("/", "/favicon.ico")
                       and not (folder / path.split("?")[0].lstrip("/")).is_file())
    assert strangers == []
    # The thread's requests are among them: the checks above would pass for a page that never loaded.
    manifest = json.loads((folder / "cifarium.json").read_text(encoding="utf-8"))
    wanted = {"/" + name for name in bundle.first_load(manifest, "cpu")[1]}
    assert wanted <= asked, sorted(wanted - asked)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextmanager
def python_http_server(folder: Path) -> Iterator[str]:
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(folder)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("python -m http.server did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_the_folder_opens_with_the_python_server_the_build_names(bench, folder):
    # `python -m http.server 8000 --directory <folder>`, as the end of `cifarium build` says.
    with python_http_server(folder) as url:
        shown = use_page(bench, url + "/")

    check_page(shown, folder)
    assert shown["isolated"] is False and threads_of(shown["info"]) == 1

"""Bench: the local server plus the installed Chrome, driven by Playwright.

Chrome is the one already installed on the machine (``channel="chrome"``),
started with a profile of its own from the cifarium cache. It starts without a
window; if the job needs a graphics card and the windowless browser has no real
adapter, it is restarted with a window (ARCHITECTURE.md, section 5).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright
from playwright.sync_api import Error as PlaywrightError

from . import paths
from .server import BenchServer

READY_TIMEOUT_MS = 30_000


class BenchError(RuntimeError):
    """The browser could not be started or the bench page failed."""


@dataclass
class Bench:
    """A running bench."""

    server: BenchServer
    context: BrowserContext
    #: False when Chrome runs with a window.
    headless: bool
    #: What the bench page reported about the machine (``DeviceInfo`` in bench.js).
    device: dict[str, Any]


def has_real_gpu(device: dict[str, Any]) -> bool:
    """True when the page got a WebGPU adapter that is not a software one."""
    gpu = device.get("webgpu")
    return bool(gpu) and not gpu.get("isFallbackAdapter")


@contextmanager
def open_bench(
    cache: Path | None = None, *, headed: bool = False, needs_gpu: bool = False
) -> Iterator[Bench]:
    """Start the server and Chrome, and close both on exit.

    ``headed`` opens a window straight away. Otherwise Chrome starts without
    one and gets a window only if ``needs_gpu`` is set and no real adapter is
    visible without it.
    """
    routes = {
        "bench": paths.web_dir() / "bench",
        "vendor": paths.vendor_dir(cache),
        "models": paths.models_dir(cache),
    }
    profile = paths.profile_dir(cache)
    with BenchServer(routes) as server, sync_playwright() as playwright:
        context, headless, device = start_chrome(
            launch=lambda headless: _launch(playwright, profile, headless=headless),
            probe=lambda context: _probe(context, server.url),
            headed=headed,
            needs_gpu=needs_gpu,
        )
        try:
            yield Bench(server=server, context=context, headless=headless, device=device)
        finally:
            _close(context)


def start_chrome(
    launch: Callable[[bool], BrowserContext],
    probe: Callable[[BrowserContext], dict[str, Any]],
    *,
    headed: bool,
    needs_gpu: bool,
) -> tuple[BrowserContext, bool, dict[str, Any]]:
    """Launch Chrome, ask the page what it sees and decide whether a window is needed.

    ``launch(headless)`` starts the browser and ``probe(context)`` returns the
    page's device report. Returns the browser, whether it is windowless, and
    the report from the browser that stayed open.
    """
    headless = not headed
    context = launch(headless)
    try:
        device = probe(context)
        if headless and needs_gpu and not has_real_gpu(device):
            _close(context)
            headless = False
            context = launch(False)
            device = probe(context)
    except BaseException:
        _close(context)
        raise
    return context, headless, device


def open_tab(context: BrowserContext, base_url: str) -> Page:
    """Open the bench page in a new tab and wait until its script is ready."""
    page = context.new_page()
    try:
        page.goto(f"{base_url}/bench/")
        page.wait_for_function(
            "() => window.cifariumBench !== undefined", timeout=READY_TIMEOUT_MS
        )
    except PlaywrightError as error:
        page.close()
        raise BenchError(f"The bench page did not start: {_first_line(error)}") from None
    return page


def _launch(playwright: Playwright, profile: Path, *, headless: bool) -> BrowserContext:
    profile.mkdir(parents=True, exist_ok=True)
    try:
        return playwright.chromium.launch_persistent_context(
            str(profile), channel="chrome", headless=headless
        )
    except PlaywrightError as error:
        # Only the first line: the rest of Playwright's message suggests
        # `playwright install chrome`, which overwrites an installed Chrome.
        raise BenchError(
            f"Could not start Google Chrome: {_first_line(error)}\n"
            "cifarium drives the Chrome that is already installed on this machine. "
            "Install Google Chrome if it is missing, and close any Chrome window "
            "left over from an earlier cifarium run."
        ) from None


def _probe(context: BrowserContext, base_url: str) -> dict[str, Any]:
    page = open_tab(context, base_url)
    try:
        return page.evaluate("() => window.cifariumBench.deviceInfo()")
    finally:
        page.close()


def _close(context: BrowserContext) -> None:
    try:
        context.close()
    except PlaywrightError:
        pass  # already closed, or Chrome is gone


def _first_line(error: Exception) -> str:
    return str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__

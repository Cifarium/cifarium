"""Bench: the local server plus the installed Chrome, driven by Playwright.

Chrome is the one already installed on the machine (``channel="chrome"``),
started with a profile of its own from the cifarium cache; the profile is
emptied before and after every session. Chrome starts without a window; if the
job needs a graphics card and the windowless browser has no real adapter, it
is restarted with a window (ARCHITECTURE.md, section 5).

Python and the page talk through ``window.cifariumBench`` (see web/bench/bench.js).
"""

from __future__ import annotations

import base64
import re
import shutil
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from . import catalog, paths, vendor
from .config import Label
from .server import BenchServer

READY_TIMEOUT_MS = 30_000

_EXTERNAL_DATA = re.compile(r"\.onnx_data(_\d+)?$")

# What Transformers.js 4.3.2 adds to the graph name for each dtype of the catalog.
DTYPE_SUFFIXES = {
    "fp32": "", "fp16": "_fp16", "q8": "_quantized", "int8": "_int8", "uint8": "_uint8",
    "q4": "_q4", "q4f16": "_q4f16", "bnb4": "_bnb4",
}


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

    def run(
        self,
        job: dict[str, Any],
        *,
        timeout: float = 300.0,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Run one job in a new tab and return what the page computed.

        ``job`` is an ``EmbedJob`` or a ``ZeroShotJob`` of bench.js (``operation``, ``model``,
        ``dtype``, ``device``, ``inputs``, ``modelFileName``, ``externalData``, ``threads`` if set,
        and ``pooling`` and ``prefix`` or ``hypotheses``); :func:`job_of` makes one from the catalog.
        The addresses of the engine and of the models folder are filled in here.
        The page's console messages and errors come back under ``events``.
        A new tab per job means that closing it frees the model's memory.
        """
        payload = {"engine": engine_urls(), "modelsPath": "/models/", **job}
        page, events = open_tab(self.context, self.server.url, on_progress=on_progress)
        try:
            # start() returns at once. Waiting is done by wait_for_function,
            # because its time limit is enforced by Playwright and so still
            # works while the page is busy computing.
            page.evaluate("job => window.cifariumBench.start(job)", payload)
            page.wait_for_function(
                "() => window.cifariumBench.outcome !== undefined",
                polling=100,
                timeout=timeout * 1000,
            )
            outcome = page.evaluate("() => window.cifariumBench.outcome")
        except PlaywrightTimeoutError:
            raise BenchError(
                f"The job exceeded its time limit of {timeout:g} s." + _recent(events)
            ) from None
        except PlaywrightError as error:
            raise BenchError(
                f"The bench page failed: {_first_line(error)}" + _recent(events)
            ) from None
        finally:
            _close(page)

        if "error" in outcome:
            raise BenchError(f"The job failed in the browser: {outcome['error']}" + _recent(events))
        return {**outcome["result"], "events": events}


def embed_job(
    entry: catalog.Entry,
    recipe: catalog.Recipe,
    inputs: Sequence[str],
    variant: str = "cpu",
    *,
    threads: int | None = None,
) -> dict[str, Any]:
    """The bench job that turns ``inputs`` into vectors the way ``recipe`` says.

    Pooling and prefix come from the recipe and are applied by the page, so the
    numbers match what the site will compute. The graph the library loads and
    the number of its ``.onnx_data`` files come from the files the catalog
    lists for the variant, not from the library's defaults. ``threads`` fixes the
    number of processor threads; without it the engine picks them, as on a site.
    """
    if recipe.kind != "embed-centroid":
        raise ValueError(f"{entry.id}: recipe {recipe.kind} does not turn text into vectors")
    return {"operation": "embed", **_model_fields(entry, variant, inputs, threads),
            "pooling": recipe.params["pooling"], "prefix": recipe.params["prefix"]}


def zero_shot_job(
    entry: catalog.Entry,
    recipe: catalog.Recipe,
    inputs: Sequence[str],
    sentences: Sequence[str],
    variant: str = "cpu",
    *,
    threads: int | None = None,
) -> dict[str, Any]:
    """The bench job that scores every sentence of ``sentences`` against every text of ``inputs``
    (``ZeroShotJob`` in bench.js): one hypothesis per label, in the order of the labels.
    :func:`hypotheses` makes them from the labels; the page takes each one whole."""
    if recipe.kind != "zero-shot":
        raise ValueError(f"{entry.id}: recipe {recipe.kind} does not check hypotheses")
    if len(sentences) < 2 or not all(isinstance(text, str) and text for text in sentences):
        raise ValueError(f"expected at least two hypotheses that are not empty, got {list(sentences)}")
    return {"operation": "zero-shot", **_model_fields(entry, variant, inputs, threads),
            "hypotheses": list(sentences)}


def job_of(
    candidate: catalog.Candidate,
    inputs: Sequence[str],
    labels: Sequence[Label],
    *,
    threads: int | None = None,
) -> dict[str, Any]:
    """The job of the candidate's recipe on the processor's variant: vectors for ``embed-centroid``,
    the scores of ``labels`` for ``zero-shot``. Raises :class:`ValueError` for another recipe."""
    entry, recipe = candidate.entry, candidate.recipe
    if recipe.kind == "zero-shot":
        return zero_shot_job(entry, recipe, inputs, hypotheses(labels), threads=threads)
    return embed_job(entry, recipe, inputs, threads=threads)


def _model_fields(entry: catalog.Entry, variant: str, inputs: Sequence[str], threads: int | None) -> dict[str, Any]:
    """What every job says about the model: its files, the device, the inputs and the threads."""
    if threads is not None and threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")
    chosen = entry.variants[variant]
    fields: dict[str, Any] = {
        "model": entry.repo,
        "dtype": chosen.dtype,
        "device": chosen.device,
        "inputs": list(inputs),
        "modelFileName": graph_name(chosen.files, chosen.dtype),
        "externalData": external_data_files(chosen.files),
    }
    if threads is not None:
        fields["threads"] = threads
    return fields


TEMPLATES = {
    "en": "This example is about {}.",
    "ru": "Этот текст про {}.",
}
"""The hypothesis of the recipe zero-shot: the sentence the model checks against the text, with the
label in place of ``{}``. Russian labels get the Russian one, all others the English one
(:func:`hypothesis_language`); ARCHITECTURE.md, section 3, has the choice and its measurement."""

CYRILLIC_SHARE = 0.5
"""The labels are Russian when more than this share of their letters is Cyrillic."""


def hypotheses(labels: Sequence[Label]) -> list[str]:
    """One hypothesis per label, in their order: the template of the labels' language with the
    label's text in it (:func:`label_text`)."""
    texts = [label_text(label) for label in labels]
    template = TEMPLATES[hypothesis_language(texts)]
    return [template.replace("{}", text, 1) for text in texts]


def label_text(label: Label) -> str:
    """What stands for the label in its hypothesis: its name, with underscores as spaces (``top_up``
    is "top up"). Not the description: on the golden task a description in the hypothesis cost
    5 to 30 points of the share (ARCHITECTURE.md, section 3); it stays for people and the page."""
    return label.name.replace("_", " ")


def hypothesis_language(texts: Iterable[str]) -> str:
    """``ru`` when more than half of the letters in ``texts`` are Cyrillic, otherwise ``en``.
    Other scripts get the English template: the multilingual model of the catalog reads a
    hypothesis in another language than the text (ARCHITECTURE.md, section 3)."""
    letters = cyrillic = 0
    for text in texts:
        for char in text:
            if char.isalpha():
                letters += 1
                cyrillic += unicodedata.name(char, "").startswith("CYRILLIC")
    return "ru" if letters and cyrillic / letters > CYRILLIC_SHARE else "en"


def graph_name(files: Iterable[str], dtype: str) -> str:
    """The ``model_file_name`` that makes the library load the one graph among ``files``.

    The library looks in ``onnx/`` and adds the dtype's suffix itself, so
    ``onnx/model_no_gather_q4.onnx`` at q4 is ``model_no_gather``.
    """
    graphs = [name for name in files if name.endswith(".onnx")]
    if len(graphs) != 1:
        raise ValueError(f"expected one .onnx graph, got {graphs}")
    folder, _, base = graphs[0].rpartition("/")
    stem, suffix = base[: -len(".onnx")], DTYPE_SUFFIXES[dtype]
    if folder != "onnx" or not stem.endswith(suffix) or stem == suffix:
        raise ValueError(f"the library cannot load {graphs[0]} at {dtype}: it looks for "
                         f"onnx/<name>{suffix}.onnx")
    return stem[: len(stem) - len(suffix)]


def external_data_files(files: Iterable[str]) -> int:
    """How many of ``files`` hold a graph's weights outside it (``.onnx_data``, ``.onnx_data_1``...)."""
    return sum(1 for name in files if _EXTERNAL_DATA.search(name))


def engine_urls() -> dict[str, str]:
    """Addresses of the engine files on the bench server (``EngineFiles`` in bench.js)."""
    base = f"/vendor/{vendor.TRANSFORMERS_VERSION}/"
    return {
        "library": base + vendor.LIBRARY_FILE,
        "wasmLoader": base + vendor.WASM_LOADER_FILE,
        "wasmBinary": base + vendor.WASM_BINARY_FILE,
    }


def decode_vectors(result: dict[str, Any]) -> np.ndarray:
    """Turn the page's base64 string into a ``(count, dims)`` float32 array."""
    data = base64.b64decode(result["vectors"])
    return np.frombuffer(data, dtype="<f4").reshape(result["count"], result["dims"])


def decode_scores(result: dict[str, Any]) -> np.ndarray:
    """The page's scores of a zero-shot job as a ``(texts, labels)`` float64 array. How many rows
    and columns there should be is for the evaluation to check; rows of different lengths are
    a page that went wrong: :class:`BenchError`."""
    rows = result["scores"]
    if len({len(row) for row in rows}) > 1:
        raise BenchError(f"the page returned rows of {sorted({len(row) for row in rows})} scores")
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(rows[0]) if rows else 0)


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
            _wipe(profile)


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


def open_tab(
    context: BrowserContext,
    base_url: str,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[Page, list[str]]:
    """Open the bench page in a new tab and wait until its script is ready.

    Returns the tab and a list that keeps filling with its console messages,
    script errors and crashes.
    """
    page = context.new_page()
    events: list[str] = []
    page.on("console", lambda message: events.append(f"console {message.type}: {message.text}"))
    page.on("pageerror", lambda error: events.append(f"script error: {error}"))
    page.on("crash", lambda _: events.append("the tab crashed, probably out of memory"))
    try:
        page.expose_function("cifariumProgress", on_progress or _ignore)
        page.goto(f"{base_url}/bench/")
        page.wait_for_function(
            "() => window.cifariumBench !== undefined", timeout=READY_TIMEOUT_MS
        )
    except PlaywrightError as error:
        _close(page)
        raise BenchError(
            f"The bench page did not start: {_first_line(error)}" + _recent(events)
        ) from None
    return page, events


def _launch(playwright: Playwright, profile: Path, *, headless: bool) -> BrowserContext:
    # Every session starts from an empty profile. The server gets a new port on
    # each run, and browser storage is kept per address including the port, so
    # the model and the executor cached by earlier runs (about 50 MB each time
    # even for a small model) could never be reused and would pile up.
    _wipe(profile)
    profile.mkdir(parents=True, exist_ok=True)
    try:
        return playwright.chromium.launch_persistent_context(
            str(profile), channel="chrome", headless=headless
        )
    except PlaywrightError as error:
        # Only the first line: the rest of Playwright's message may suggest
        # `playwright install chrome`, which overwrites an installed Chrome.
        raise BenchError(
            f"Could not start Google Chrome: {_first_line(error)}\n"
            "cifarium drives the Chrome that is already installed on this machine. "
            "Install Google Chrome if it is missing, and close any Chrome window "
            "left over from an earlier cifarium run."
        ) from None


def _probe(context: BrowserContext, base_url: str) -> dict[str, Any]:
    page, _ = open_tab(context, base_url)
    try:
        return page.evaluate("() => window.cifariumBench.deviceInfo()")
    finally:
        _close(page)


def _close(target: BrowserContext | Page) -> None:
    try:
        target.close()
    except PlaywrightError:
        pass  # already closed, or Chrome is gone


def _wipe(profile: Path) -> None:
    """Delete the bench's Chrome profile; files still held by Chrome are left for next time."""
    shutil.rmtree(profile, ignore_errors=True)


def _ignore(message: str) -> None:
    pass


def _first_line(error: Exception) -> str:
    lines = str(error).strip().splitlines()
    return lines[0] if lines else type(error).__name__


def _recent(events: list[str], limit: int = 5) -> str:
    """The last few page events, formatted to follow an error message."""
    return "".join(f"\n  {event}" for event in events[-limit:])

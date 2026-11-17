"""Bench logic that needs no browser: jobs from the catalog, when Chrome gets a window,
decoding vectors."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from cifarium import catalog, vendor
from cifarium.bench import (
    DTYPE_SUFFIXES,
    decode_vectors,
    embed_job,
    engine_urls,
    external_data_files,
    graph_name,
    has_real_gpu,
    start_chrome,
)

INTEL = {"webgpu": {"vendor": "intel", "isFallbackAdapter": False}}
SOFTWARE = {"webgpu": {"vendor": "google", "isFallbackAdapter": True}}
NO_ADAPTER = {"webgpu": None}


class FakeContext:
    def __init__(self, headless: bool):
        self.headless = headless
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeChrome:
    """Records launches; the page's report depends on whether there is a window."""

    def __init__(self, *, windowless: dict, windowed: dict):
        self.reports = {True: windowless, False: windowed}
        self.launched: list[FakeContext] = []

    def launch(self, headless: bool) -> FakeContext:
        context = FakeContext(headless)
        self.launched.append(context)
        return context

    def probe(self, context: FakeContext) -> dict:
        return self.reports[context.headless]


def test_vectors_are_decoded_as_little_endian_float32_rows():
    rows = np.array([[0.5, -1.0, 2.0], [0.25, 0.0, 1e-3]], dtype="<f4")
    result = {"count": 2, "dims": 3, "vectors": base64.b64encode(rows.tobytes()).decode("ascii")}

    vectors = decode_vectors(result)

    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    assert np.array_equal(vectors, rows)


def _entry(entry_id: str) -> catalog.Entry:
    return catalog.read_entry(catalog.CATALOG_DIR / f"{entry_id}.yaml")


def _centroid_recipe(entry: catalog.Entry) -> catalog.Recipe:
    return next(recipe for recipe in entry.recipes if recipe.kind == "embed-centroid")


def test_a_job_takes_pooling_and_prefix_from_the_recipe():
    entry = _entry("multilingual-e5-small")
    job = embed_job(entry, _centroid_recipe(entry), ("Где моя посылка?",))
    assert job == {
        "model": "Xenova/multilingual-e5-small",
        "dtype": "q8",
        "device": "wasm",
        "inputs": ["Где моя посылка?"],
        "pooling": "mean",
        "prefix": "query: ",
        "modelFileName": "model",
        "externalData": 0,
    }


def test_a_job_fixes_the_threads_only_when_asked():
    entry = _entry("all-minilm-l6-v2")
    recipe = _centroid_recipe(entry)
    assert "threads" not in embed_job(entry, recipe, ["a"])
    assert embed_job(entry, recipe, ["a"], threads=1)["threads"] == 1
    with pytest.raises(ValueError, match="at least 1"):
        embed_job(entry, recipe, ["a"], threads=0)


def test_a_job_counts_the_external_data_files_of_the_variant():
    entry = _entry("embeddinggemma-300m")
    job = embed_job(entry, _centroid_recipe(entry), ["a"])
    assert (job["pooling"], job["prefix"], job["externalData"]) == (
        "none", "task: classification | query: ", 1)
    assert job["dtype"] == "q4"


def test_every_vector_recipe_in_the_catalog_makes_a_job():
    for candidate in catalog.candidates("classify"):
        if candidate.recipe.kind == "embed-centroid":
            job = embed_job(candidate.entry, candidate.recipe, ["a"])
            assert job["pooling"] in ("mean", "cls", "none")


def test_a_zero_shot_recipe_makes_no_vector_job():
    entry = _entry("mobilebert-uncased-mnli")
    with pytest.raises(ValueError, match="does not turn text into vectors"):
        embed_job(entry, entry.recipes[0], ["a"])


@pytest.mark.parametrize(
    ("graph", "dtype", "name"),
    [
        ("onnx/model_quantized.onnx", "q8", "model"),
        ("onnx/model_q4.onnx", "q4", "model"),
        ("onnx/model_no_gather_q4.onnx", "q4", "model_no_gather"),
        ("onnx/model.onnx", "fp32", "model"),
    ],
)
def test_the_graph_name_is_the_file_without_the_dtype_suffix(graph, dtype, name):
    assert graph_name(["tokenizer.json", graph, graph + "_data"], dtype) == name


@pytest.mark.parametrize(
    ("files", "dtype"),
    [
        (["onnx/model_q4.onnx"], "q8"),
        (["model_quantized.onnx"], "q8"),
        (["onnx/_quantized.onnx"], "q8"),
        (["onnx/model_q4.onnx", "onnx/model_quantized.onnx"], "q8"),
        (["onnx/model_q4.onnx_data"], "q4"),
    ],
)
def test_a_graph_the_library_would_not_load_is_refused(files, dtype):
    with pytest.raises(ValueError):
        graph_name(files, dtype)


def test_every_catalog_dtype_has_a_suffix():
    assert set(DTYPE_SUFFIXES) == set(catalog.DTYPES)


def test_external_data_files_are_the_onnx_data_parts():
    files = ["config.json", "onnx/model.onnx", "onnx/model.onnx_data", "onnx/model.onnx_data_1",
             "onnx/model.onnx_database"]
    assert external_data_files(files) == 2


def test_engine_urls_point_into_the_vendor_folder():
    urls = engine_urls()
    assert set(urls) == {"library", "wasmLoader", "wasmBinary"}
    assert all(url.startswith(f"/vendor/{vendor.TRANSFORMERS_VERSION}/") for url in urls.values())


@pytest.mark.parametrize(
    ("device", "expected"),
    [(INTEL, True), (SOFTWARE, False), (NO_ADAPTER, False), ({}, False)],
)
def test_has_real_gpu(device, expected):
    assert has_real_gpu(device) is expected


def test_stays_windowless_when_the_gpu_is_visible():
    chrome = FakeChrome(windowless=INTEL, windowed=INTEL)

    context, headless, device = start_chrome(
        chrome.launch, chrome.probe, headed=False, needs_gpu=True
    )

    assert headless is True
    assert device == INTEL
    assert chrome.launched == [context]
    assert not context.closed


@pytest.mark.parametrize("windowless", [NO_ADAPTER, SOFTWARE])
def test_restarts_with_a_window_when_the_job_needs_a_gpu(windowless):
    chrome = FakeChrome(windowless=windowless, windowed=INTEL)

    context, headless, device = start_chrome(
        chrome.launch, chrome.probe, headed=False, needs_gpu=True
    )

    first, second = chrome.launched
    assert first.headless and first.closed
    assert context is second and not second.headless and not second.closed
    assert headless is False
    assert device == INTEL


def test_no_restart_for_a_cpu_job():
    chrome = FakeChrome(windowless=NO_ADAPTER, windowed=INTEL)

    context, headless, device = start_chrome(
        chrome.launch, chrome.probe, headed=False, needs_gpu=False
    )

    assert headless is True
    assert device == NO_ADAPTER
    assert chrome.launched == [context]


def test_headed_opens_a_window_straight_away():
    chrome = FakeChrome(windowless=NO_ADAPTER, windowed=INTEL)

    context, headless, _ = start_chrome(chrome.launch, chrome.probe, headed=True, needs_gpu=False)

    assert headless is False
    assert [c.headless for c in chrome.launched] == [False]
    assert context is chrome.launched[0]


def test_closes_chrome_when_the_page_fails():
    chrome = FakeChrome(windowless=INTEL, windowed=INTEL)

    def broken_probe(context):
        raise RuntimeError("the bench page did not start")

    with pytest.raises(RuntimeError):
        start_chrome(chrome.launch, broken_probe, headed=False, needs_gpu=False)

    assert chrome.launched[0].closed

"""Bench logic that needs no browser: when Chrome gets a window, decoding vectors."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from cifarium import vendor
from cifarium.bench import decode_vectors, engine_urls, has_real_gpu, start_chrome

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

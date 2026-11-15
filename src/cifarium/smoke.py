"""Temporary end-to-end check: one small model, twenty sentences, on the processor.

Downloads the engine and the model into the cache, starts the bench, computes
the vectors in Chrome and prints what came back. It exists until the real
commands do: remove it, and its entry in cli.py, once `cifarium build` covers
the same path.
"""

from __future__ import annotations

import functools
import re
import statistics
import sys

import numpy as np

from . import hub, vendor
from .bench import Bench, BenchError, decode_vectors, open_bench

MODEL = "Xenova/all-MiniLM-L6-v2"
REVISION = "751bff37182d3f1213fa05d7196b954e230abad9"
DTYPE = "q8"
FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model_quantized.onnx")
DIMS = 384
TIME_LIMIT_S = 300.0

# Ten pairs: both sentences of a pair say roughly the same thing.
SENTENCES = (
    "The cat is sleeping on the sofa.",
    "A kitten naps on the couch.",
    "How do I reset my password?",
    "I forgot my password and need to change it.",
    "The stock market fell sharply today.",
    "Shares dropped steeply in today's trading.",
    "She plays the violin in an orchestra.",
    "He performs as a violinist with a symphony.",
    "It is raining heavily outside.",
    "A downpour is soaking the streets.",
    "Where can I track my order?",
    "I want to know when my package will arrive.",
    "The recipe needs two cups of flour.",
    "You have to add two cups of flour to the dough.",
    "Our flight was delayed by three hours.",
    "The plane took off three hours later than scheduled.",
    "The new phone has a great camera.",
    "This smartphone takes excellent photos.",
    "He scored the winning goal in the final.",
    "The striker won the championship match with a late goal.",
)

_say = functools.partial(print, flush=True)


def run(*, headed: bool = False) -> int:
    """Run the check and return the process exit code."""
    _say("cifarium smoke test")
    job = {"model": MODEL, "dtype": DTYPE, "device": "wasm", "inputs": list(SENTENCES)}
    try:
        vendor.ensure_engine(report=lambda message: _say(f"  {message}"))
        _say(
            f"  engine    Transformers.js {vendor.TRANSFORMERS_VERSION}, "
            f"onnxruntime-web {vendor.ORT_WEB_VERSION}"
        )
        folder = hub.download_files(MODEL, REVISION, FILES)
        size = sum((folder / name).stat().st_size for name in FILES)
        _say(f"  model     {MODEL} ({DTYPE}), {size / 1e6:.1f} MB")
        with open_bench(headed=headed) as bench:
            _say(f"  browser   {_describe_browser(bench)}")
            result = bench.run(
                job, timeout=TIME_LIMIT_S, on_progress=lambda message: _say(f"  page      {message}")
            )
    except (vendor.VendorError, hub.HubError, BenchError) as error:
        print(f"\nFAILED: {error}", file=sys.stderr)
        return 1

    vectors = decode_vectors(result)
    times = result["timesMs"]
    within, across = _pair_similarity(vectors)
    device = "processor (wasm)" if result["device"] == "wasm" else "graphics card (webgpu)"
    rows = [
        ("vectors", f"{vectors.shape[0]} x {vectors.shape[1]}"),
        ("device", f"{device}, {result['threads']} threads"),
        ("crossOriginIsolated", str(result["crossOriginIsolated"]).lower()),
        ("model load", f"{result['loadMs'] / 1000:.2f} s"),
        (
            "per sentence",
            f"{statistics.median(times[1:]):.1f} ms median, first call {times[0]:.1f} ms",
        ),
        ("similarity", f"{within:.2f} within pairs, {across:.2f} across pairs"),
    ]
    _say()
    for name, value in rows:
        _say(f"  {name:<21} {value}")

    problems = _problems(vectors, within, across)
    if problems:
        print("\nFAILED: " + "; ".join(problems), file=sys.stderr)
        return 1
    _say("\nOK")
    return 0


def _describe_browser(bench: Bench) -> str:
    version = re.search(r"Chrome/(\d+)", bench.device["userAgent"])
    name = f"Chrome {version.group(1)}" if version else "Chrome"
    window = "without a window" if bench.headless else "with a window"
    gpu = bench.device["webgpu"]
    if not gpu:
        adapter = "none"
    else:
        adapter = " ".join(part for part in (gpu["vendor"], gpu["architecture"]) if part) or "unnamed"
        if gpu["isFallbackAdapter"]:
            adapter += " (software)"
    return f"{name} {window}, WebGPU adapter: {adapter}"


def _pair_similarity(vectors: np.ndarray) -> tuple[float, float]:
    """Mean cosine similarity inside the sentence pairs and between different pairs."""
    similarity = vectors @ vectors.T
    index = np.arange(len(vectors))
    same_pair = (index[:, None] // 2) == (index[None, :] // 2)
    upper = index[:, None] < index[None, :]
    return float(similarity[same_pair & upper].mean()), float(similarity[~same_pair & upper].mean())


def _problems(vectors: np.ndarray, within: float, across: float) -> list[str]:
    problems = []
    if vectors.shape != (len(SENTENCES), DIMS):
        problems.append(f"expected {len(SENTENCES)} x {DIMS} vectors, got {vectors.shape}")
    if not np.isfinite(vectors).all():
        problems.append("the vectors contain NaN or infinity")
    elif not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
        problems.append("the vectors are not of unit length")
    if not within > across:
        problems.append("paraphrases are not closer to each other than unrelated sentences")
    return problems

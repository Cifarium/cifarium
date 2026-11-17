"""The runtime (web/runtime/) in the real Chrome, on a build folder this file puts together by hand.

The folder follows ARCHITECTURE.md, section 7: engine, all-MiniLM-L6-v2 from the cache, manifest,
and the centres of the 20 examples of examples/english. A server of its own serves it with the
isolation headers. The labels the runtime gives the 100 held-out examples must match
evaluate.predict on the same centres. The build command does not write this folder yet (row 7 of
the queue); :func:`write_build` is the layout it has to produce.

Runs only with `pytest -m e2e`. It downloads the same files as tests/test_build_e2e.py.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from playwright.sync_api import Page

import cifarium
from cifarium import catalog, config, evaluate, examples, hub, paths, vendor
from cifarium.bench import (
    Bench,
    decode_vectors,
    embed_job,
    external_data_files,
    graph_name,
    has_real_gpu,
    open_bench,
)
from cifarium.server import BenchServer

pytestmark = pytest.mark.e2e

GOLDEN = Path(__file__).resolve().parents[1] / "examples" / "english"
MODEL_ID = "all-minilm-l6-v2"
ENGINE_FILES = {
    "library": vendor.LIBRARY_FILE,
    "loader": vendor.WASM_LOADER_FILE,
    "executor": vendor.WASM_BINARY_FILE,
}
TIMEOUT_S = 180.0

# A page that only imports the runtime. Python passes it an async function as text and waits,
# with a time limit, for its outcome.
TEST_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>cifarium runtime test</title>
<script type="module">
import * as runtime from './cifarium.js';
window.runtime = runtime;
window.start = (source, arg) => {
  window.outcome = undefined;
  const scenario = (0, eval)(`(${source})`);
  Promise.resolve().then(() => scenario(arg)).then(
    (result) => { window.outcome = { result }; },
    (error) => { window.outcome = { error: `${error?.code ?? ''} ${error?.message ?? error}` }; },
  );
};
</script>
"""

# Load one manifest and name the label of every text, all calls at once.
LOAD_AND_RUN = """async ({ manifest, device, texts, clear }) => {
  if (clear) await caches.delete('transformers-cache');
  const progress = [];
  const model = await runtime.createCifarium(manifest);
  const info = await model.load({ device, onProgress: (p) => progress.push(p) });
  const results = await Promise.all(texts.map((text) => model.run(text)));
  window.model = model;
  return { info, progress, results };
}"""


@dataclass
class Golden:
    bench: Bench
    server: BenchServer
    folder: Path
    names: list[str]
    texts: list[str]
    """The 100 held-out examples."""
    labels: list[str]
    """What evaluate.predict names for them on the centres the folder holds."""
    scores: list[float]
    bytes: int
    """What the manifest says a first visit downloads."""
    files: set[str]
    """Paths of every file of the build's first load, relative to the manifest."""

    def url(self, name: str) -> str:
        return f"{self.server.url}/site/{name}"


def write_build(out: Path, entry: catalog.Entry, recipe: catalog.Recipe, names: list[str],
                centres: np.ndarray) -> dict[str, Any]:
    """Lay out a build folder for a classify task by examples, and return its manifest.

    Section 7: the runtime, the engine under vendor/transformers-<version>/, the model under
    models/<revision>/<organisation>/<model>/ (the library refuses a model id with "@<hash>",
    so the version is a folder of its own), labels.json and labels.bin under data/.
    """
    runtime = paths.web_dir() / "runtime"
    for name in ("cifarium.js", "cifarium-worker.js"):
        shutil.copyfile(runtime / name, out / name)

    engine_path = f"vendor/transformers-{vendor.TRANSFORMERS_VERSION}/"
    (out / engine_path).mkdir(parents=True)
    source = vendor.ensure_engine()
    engine: dict[str, Any] = {
        "transformers": vendor.TRANSFORMERS_VERSION,
        "onnxruntime_web": vendor.ORT_WEB_VERSION,
    }
    for role, name in ENGINE_FILES.items():
        shutil.copyfile(source / name, out / engine_path / name)
        engine[role] = {"path": engine_path + name, "bytes": vendor.ENGINE_BYTES[name]}

    model_path = f"models/{entry.revision}/"
    target = out / model_path / entry.repo
    for name in entry.files():
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hub.model_dir(entry.repo) / name, target / name)

    (out / "data").mkdir()
    labels = json.dumps(names, ensure_ascii=False).encode("utf-8")
    (out / "data" / "labels.json").write_bytes(labels)
    rows = np.asarray(centres, dtype="<f4").tobytes()
    (out / "data" / "labels.bin").write_bytes(rows)

    manifest = {
        "format": 1,
        "cifarium": cifarium.__version__,
        "task": "classify",
        "recipe": recipe.kind,
        "model": {"repo": entry.repo, "revision": entry.revision, "license": entry.license,
                  "path": model_path},
        "variants": {"cpu": variant_record(entry, "cpu")},
        "params": {"pooling": recipe.params["pooling"], "prefix": recipe.params["prefix"]},
        "engine": engine,
        "data": {
            "labels": {"path": "data/labels.json", "bytes": len(labels)},
            "centres": {"path": "data/labels.bin", "bytes": len(rows)},
        },
    }
    write_manifest(out / "cifarium.json", manifest)
    return manifest


def variant_record(entry: catalog.Entry, name: str) -> dict[str, Any]:
    variant = entry.variants[name]
    return {
        "device": variant.device,
        "dtype": variant.dtype,
        "graph": graph_name(variant.files, variant.dtype),
        "externalData": external_data_files(variant.files),
        "files": entry.files(name),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def first_load(manifest: dict[str, Any], variant: str) -> tuple[int, set[str]]:
    """Bytes and paths of everything a first visit downloads for ``variant``."""
    refs = [manifest["engine"][role] for role in ENGINE_FILES] + list(manifest["data"].values())
    folder = manifest["model"]["path"] + manifest["model"]["repo"] + "/"
    files = {ref["path"]: ref["bytes"] for ref in refs}
    files.update({folder + name: size for name, size in manifest["variants"][variant]["files"].items()})
    return sum(files.values()), set(files)


@pytest.fixture(scope="module")
def golden(tmp_path_factory) -> Golden:
    entry = next(item for item in catalog.load() if item.id == MODEL_ID)
    recipe = next(item for item in entry.recipes if item.task == "classify")
    settings = config.load(GOLDEN / "cifarium.yaml")
    shipped = examples.load(settings)
    held = examples.read(GOLDEN / "heldout.jsonl", settings.labels)
    names = [label.name for label in shipped.labels]
    vendor.ensure_engine()
    hub.download_files(entry.repo, entry.revision, entry.files())

    out = tmp_path_factory.mktemp("build")
    with open_bench() as bench:
        texts = [item.input for item in shipped.items + held.items]
        vectors = decode_vectors(bench.run(embed_job(entry, recipe, texts)))
        centres = evaluate.centroids(vectors[: len(shipped.items)],
                                     [item.expected for item in shipped.items], names)
        # The runtime reads the centres as float32, so Python predicts from the same numbers.
        centres = centres.astype("<f4").astype(np.float64)
        best, scores = evaluate.predict(vectors[len(shipped.items):], centres)

        manifest = write_build(out, entry, recipe, names, centres)
        write_variants(out, manifest)
        (out / "test.html").write_text(TEST_PAGE, encoding="utf-8")
        total, files = first_load(manifest, "cpu")
        with BenchServer({"site": out}) as server:
            yield Golden(
                bench=bench, server=server, folder=out, names=names,
                texts=[item.input for item in held.items],
                labels=[names[k] for k in best], scores=[float(s) for s in scores],
                bytes=total, files=files,
            )


def write_variants(out: Path, manifest: dict[str, Any]) -> None:
    """Manifests next to cifarium.json that the catalog of stage 1 cannot produce, or must refuse."""
    cpu = manifest["variants"]["cpu"]
    # The same q8 graph on the graphics card: there are no gpu variants in the catalog yet.
    write_manifest(out / "gpu.json", {**manifest, "variants": {
        "cpu": cpu, "gpu": {**cpu, "device": "webgpu"}}})
    # A card's variant whose graph is not a graph at all.
    model = out / manifest["model"]["path"] / manifest["model"]["repo"]
    (model / "onnx" / "broken.onnx").write_bytes(b"not an ONNX graph\n" * 512)
    broken = {**cpu, "device": "webgpu", "dtype": "fp32", "graph": "broken", "files": {
        **{name: size for name, size in cpu["files"].items() if not name.startswith("onnx/")},
        "onnx/broken.onnx": 512 * 18}}
    write_manifest(out / "broken.json", {**manifest, "variants": {"cpu": cpu, "gpu": broken}})

    write_manifest(out / "format2.json", {**manifest, "format": 2})
    write_manifest(out / "recipe.json", {**manifest, "recipe": "nli-labels"})
    write_manifest(out / "task.json", {**manifest, "task": "search"})
    write_manifest(out / "nodata.json", {key: value for key, value in manifest.items() if key != "data"})


def open_page(golden: Golden) -> tuple[Page, list[str]]:
    page = golden.bench.context.new_page()
    events: list[str] = []
    page.on("console", lambda message: events.append(f"console {message.type}: {message.text}"))
    page.on("pageerror", lambda error: events.append(f"script error: {error}"))
    page.goto(golden.url("test.html"))
    page.wait_for_function("() => window.start !== undefined", timeout=30_000)
    return page, events


def call(golden: Golden, page: Page, events: list[str], source: str, arg: Any = None,
         timeout: float = TIMEOUT_S) -> Any:
    """Run ``source``, an async function, on the page with ``arg``; return what it returns."""
    page.evaluate("([source, arg]) => window.start(source, arg)", [source, arg])
    page.wait_for_function("() => window.outcome !== undefined", polling=100, timeout=timeout * 1000)
    outcome = page.evaluate("() => window.outcome")
    if "error" in outcome:
        # Requests of the worker do not reach the page's events; the server saw them all.
        failed = [f"HTTP {status} {target}" for _, target, status in golden.server.requests if status >= 400]
        raise AssertionError(f"the page failed: {outcome['error']}\n" + "\n".join(events[-10:] + failed))
    return outcome["result"]


def requests_for(golden: Golden, suffix: str, since: int) -> int:
    return sum(1 for method, target, _ in golden.server.requests[since:]
               if method == "GET" and target.endswith(suffix))


def expected_threads(golden: Golden) -> int:
    # What ONNX Runtime picks on an isolated page: half the cores, at most four.
    cores = golden.bench.device["hardwareConcurrency"] or 1
    return min(4, math.ceil(cores / 2))


def assert_labels_match(golden: Golden, results: list[dict[str, Any]]) -> None:
    assert [result["label"] for result in results] == golden.labels
    differences = [abs(result["score"] - score) for result, score in zip(results, golden.scores)]
    assert max(differences) < 1e-4


def assert_progress_ends_at_100(progress: list[dict[str, Any]], total: int, files: set[str]) -> None:
    assert progress, "onProgress was never called"
    last = progress[-1]
    assert last["percent"] == 100
    assert last["allLoaded"] == last["allTotal"] == total
    assert {event["file"] for event in progress} == files
    for event in progress:
        assert 0 <= event["loaded"] <= event["total"]
        assert 0 <= event["allLoaded"] <= event["allTotal"]


def test_labels_on_held_out_match_python_and_the_first_load_is_counted(golden):
    page, events = open_page(golden)
    mark = len(golden.server.requests)
    out = call(golden, page, events, LOAD_AND_RUN, {
        "manifest": "cifarium.json", "device": "auto", "texts": golden.texts, "clear": True})

    assert_labels_match(golden, out["results"])
    assert_progress_ends_at_100(out["progress"], golden.bytes, golden.files)
    info = out["info"]
    assert info["state"] == "ready"
    assert (info["variant"], info["device"], info["dtype"]) == ("cpu", "wasm", "q8")
    assert info["fallback"] is None
    assert info["crossOriginIsolated"] is True
    assert info["threads"] == expected_threads(golden)
    assert info["bytes"] == golden.bytes
    assert info["labels"] == golden.names
    assert info["runtime"] == info["cifarium"] == cifarium.__version__
    assert (info["model"], info["transformers"]) == ("Xenova/all-MiniLM-L6-v2", vendor.TRANSFORMERS_VERSION)
    # The runtime downloads the executor itself and the library does not fetch it again.
    assert requests_for(golden, "/" + vendor.WASM_BINARY_FILE, mark) == 1
    page.close()


def test_dispose_stops_the_thread_and_a_second_load_comes_from_the_store(golden):
    page, events = open_page(golden)
    call(golden, page, events, LOAD_AND_RUN, {
        "manifest": "cifarium.json", "device": "wasm", "texts": golden.texts[:3], "clear": True})
    assert page.workers, "the runtime started no worker"

    mark = len(golden.server.requests)
    out = call(golden, page, events, """async () => {
      await window.model.dispose();
      let error = null;
      try { await window.model.run('hello'); } catch (e) { error = e.code; }
      return { error, state: window.model.info().state };
    }""")
    assert out == {"error": "disposed", "state": "disposed"}
    deadline = time.monotonic() + 10
    while page.workers and time.monotonic() < deadline:
        page.wait_for_timeout(100)
    assert page.workers == []

    # A new instance on the same page: the model and the executor come from the Cache API.
    again = call(golden, page, events, LOAD_AND_RUN, {
        "manifest": "cifarium.json", "device": "wasm", "texts": golden.texts[:3], "clear": False})
    assert [result["label"] for result in again["results"]] == golden.labels[:3]
    assert again["progress"][-1]["percent"] == 100
    assert requests_for(golden, ".wasm", mark) == 0
    assert requests_for(golden, ".onnx", mark) == 0
    page.close()


def test_the_gpu_variant_is_chosen_when_the_browser_offers_an_adapter(golden):
    page, events = open_page(golden)
    out = call(golden, page, events, LOAD_AND_RUN, {
        "manifest": "gpu.json", "device": "auto", "texts": golden.texts, "clear": False})
    info = out["info"]
    if has_real_gpu(golden.bench.device):
        assert (info["variant"], info["device"], info["fallback"]) == ("gpu", "webgpu", None)
        # The card computes in another order; only near ties may flip.
        same = sum(result["label"] == label for result, label in zip(out["results"], golden.labels))
        assert same >= len(golden.labels) - 2, Counter(
            (label, result["label"]) for result, label in zip(out["results"], golden.labels))
    else:
        assert (info["variant"], info["device"]) == ("cpu", "wasm")
        assert "adapter" in info["fallback"] or "WebGPU" in info["fallback"]
        assert_labels_match(golden, out["results"])
    page.close()


def test_a_failing_gpu_variant_hands_over_to_the_processor(golden):
    page, events = open_page(golden)
    out = call(golden, page, events, LOAD_AND_RUN, {
        "manifest": "broken.json", "device": "webgpu", "texts": golden.texts, "clear": True})
    info = out["info"]
    assert (info["variant"], info["device"]) == ("cpu", "wasm")
    assert info["fallback"].startswith("the graphics card failed: ")
    assert info["bytes"] == golden.bytes
    assert_labels_match(golden, out["results"])
    # What the card's attempt downloaded stays counted; without an adapter it downloads nothing.
    assert out["progress"][-1]["percent"] == 100
    assert out["progress"][-1]["allTotal"] >= golden.bytes
    page.close()


def test_errors_name_what_went_wrong(golden):
    page, events = open_page(golden)
    out = call(golden, page, events, """async () => {
      const code = async (action) => {
        try { await action(); return 'no error'; } catch (e) {
          return e instanceof runtime.CifariumError ? e.code : `not a CifariumError: ${e}`;
        }
      };
      const model = await runtime.createCifarium('cifarium.json');
      const result = {
        missing: await code(() => runtime.createCifarium('missing.json')),
        format: await code(() => runtime.createCifarium('format2.json')),
        recipe: await code(() => runtime.createCifarium('recipe.json')),
        task: await code(() => runtime.createCifarium('task.json')),
        nodata: await code(() => runtime.createCifarium('nodata.json')),
        beforeLoad: await code(() => model.run('hello')),
        notText: await code(() => model.run(42)),
        device: await code(() => model.load({ device: 'cuda' })),
        state: model.info().state,
      };
      // A run made while load() is still going waits for it.
      const loading = model.load();
      const early = await model.run('I lost my card');
      await loading;
      result.early = early.label;
      result.again = (await model.load()).state;
      await model.dispose();
      result.afterDispose = await code(() => model.load());
      return result;
    }""")
    assert out == {
        "missing": "manifest", "format": "unsupported", "recipe": "unsupported",
        "task": "unsupported", "nodata": "manifest", "beforeLoad": "not-loaded",
        "notText": "input", "device": "input", "state": "created",
        "early": out["early"], "again": "ready", "afterDispose": "disposed",
    }
    assert out["early"] in golden.names
    page.close()

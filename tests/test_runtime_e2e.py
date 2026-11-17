"""The runtime (web/runtime/) in the real Chrome, on the folder `bundle.write` lays out.

The folder is the one of ARCHITECTURE.md, section 7: engine, all-MiniLM-L6-v2 from the cache,
manifest, and the centres of the 20 examples of examples/english. A server of its own serves it
with the isolation headers. The labels and scores the runtime gives the 100 held-out examples must
match evaluate.predict on the same centres. Next to the manifest the fixture writes the ones the
catalog of stage 1 cannot produce, or the runtime must refuse. The contracts of section 14 are
checked here as the browser shows them: the fields of `info()`, the codes of the errors, the
reasons of `fallback`, the phases of the progress.

Runs only with `pytest -m e2e`. It downloads the same files as tests/test_build_e2e.py.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from playwright.sync_api import Page

import cifarium
from cifarium import bundle, catalog, config, evaluate, examples, hub, vendor
from cifarium.bench import Bench, decode_vectors, embed_job, has_real_gpu, open_bench
from cifarium.server import BenchServer

pytestmark = pytest.mark.e2e

GOLDEN = Path(__file__).resolve().parents[1] / "examples" / "english"
MODEL_ID = "all-minilm-l6-v2"
TIMEOUT_S = 180.0

# The contracts of ARCHITECTURE.md, section 14, as the browser shows them. A change to one of these
# lists is a change of the contract: it goes with a line in CHANGELOG.md.
INFO_KEYS = sorted([
    "state", "runtime", "cifarium", "format", "task", "recipe", "model", "revision", "license",
    "transformers", "onnxruntimeWeb", "labels", "variant", "device", "dtype", "threads",
    "crossOriginIsolated", "bytes", "fallback",
])
EXPORTS = ["CifariumError", "ERROR_CODES", "FALLBACK_REASONS", "MANIFEST_FORMAT", "PROGRESS_PHASES",
           "VERSION", "createCifarium"]
ERROR_CODES = ["manifest", "unsupported", "input", "not-loaded", "network", "load", "run", "disposed"]
FALLBACK_REASONS = ["no-webgpu", "no-adapter", "software-adapter", "no-shader-f16", "requested", "failed"]
PROGRESS_PHASES = ["download", "prepare"]

# A page that only imports the runtime. Python passes it an async function as text and waits,
# with a time limit, for its outcome. `fakeGpu` makes the next background threads of the page see
# the WebGPU the spec describes (or none at all), so every reason of a fallback can be provoked
# on a computer that has a card and on one that has not: the thread starts from FAKE_GPU_WORKER,
# a file next to the page, which sets `navigator.gpu` and then imports the real thread file,
# replaying the messages that came while it was loading. (Not from a blob: its base address is
# `blob:`, and the library cannot resolve the model's path against it.)
FAKE_GPU_WORKER = """const spec = JSON.parse(new URL(self.location.href).searchParams.get('spec'));
const early = [];
self.onmessage = (event) => early.push(event);
const gpu = spec === null ? undefined : {
  requestAdapter: async () => {
    if (spec.throws) throw new Error('requestAdapter is broken');
    if (spec.adapter === null) return null;
    return { info: { isFallbackAdapter: spec.adapter.software }, features: new Set(spec.adapter.features) };
  },
};
Object.defineProperty(navigator, 'gpu', { configurable: true, get: () => gpu });
await import('./cifarium-worker.js');
for (const event of early) self.onmessage(event);
"""

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
window.fakeGpu = (spec) => {
  const RealWorker = window.Worker;
  window.Worker = class extends RealWorker {
    constructor(url, options) {
      super(`./fake-gpu-worker.js?spec=${encodeURIComponent(JSON.stringify(spec))}`, options);
    }
  };
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

# What a failed call looks like to a program: the code, the cause and the state.
FAILED_LOAD = """async ({ manifest, clear }) => {
  if (clear) await caches.delete('transformers-cache');
  const model = await runtime.createCifarium(manifest);
  window.model = model;
  const error = await model.load().then(() => null, (e) => e);
  return {
    code: error?.code ?? null,
    isCifariumError: error instanceof runtime.CifariumError,
    cause: error?.cause ? { isError: error.cause instanceof Error, name: error.cause.name } : null,
    message: error?.message ?? null,
    state: model.info().state,
  };
}"""

LOAD_AGAIN = """async () => {
  const info = await window.model.load();
  const result = await window.model.run('I lost my card');
  return { state: info.state, label: result.label, scores: result.scores.length };
}"""

@dataclass
class Golden:
    bench: Bench
    server: BenchServer
    folder: Path
    names: list[str]
    records: list[dict[str, Any]]
    """The labels as `info()` gives them: name and description, as the settings have them."""
    texts: list[str]
    """The 100 held-out examples."""
    labels: list[str]
    """What evaluate.predict names for them on the centres the folder holds."""
    scores: list[float]
    matrix: np.ndarray
    """The score of every label for every held-out example, rows in the order of ``texts``."""
    bytes: int
    """What the manifest says a first visit downloads."""
    files: set[str]
    """Paths of every file of the build's first load, relative to the manifest."""

    def url(self, name: str) -> str:
        return f"{self.server.url}/site/{name}"


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def golden(tmp_path_factory) -> Iterator[Golden]:
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
        matrix = vectors[len(shipped.items):] @ centres.T

        manifest = bundle.write(out, catalog.Candidate(entry, recipe), shipped.labels, centres).manifest
        write_variants(out, manifest)
        (out / "test.html").write_text(TEST_PAGE, encoding="utf-8")
        (out / "fake-gpu-worker.js").write_text(FAKE_GPU_WORKER, encoding="utf-8")
        total, files = bundle.first_load(manifest, "cpu")
        with BenchServer({"site": out}) as server:
            yield Golden(
                bench=bench, server=server, folder=out, names=names,
                records=[{"name": label.name, "description": label.description} for label in shipped.labels],
                texts=[item.input for item in held.items],
                labels=[names[k] for k in best], scores=[float(s) for s in scores], matrix=matrix,
                bytes=total, files=files,
            )


def write_variants(out: Path, manifest: dict[str, Any]) -> None:
    """Manifests next to cifarium.json that the catalog of stage 1 cannot produce, or must refuse."""
    cpu = manifest["variants"]["cpu"]
    # The same q8 graph on the graphics card: there are no gpu variants in the catalog yet.
    write_manifest(out / "gpu.json", {**manifest, "variants": {
        "cpu": cpu, "gpu": {**cpu, "device": "webgpu"}}})
    # Half-precision weights need the adapter's shader-f16; the graph is never read, because the
    # adapter is refused first.
    write_manifest(out / "gpu-fp16.json", {**manifest, "variants": {
        "cpu": cpu, "gpu": {**cpu, "device": "webgpu", "dtype": "fp16"}}})
    # A card's variant whose graph is not a graph at all.
    model = out / manifest["model"]["path"] / manifest["model"]["repo"]
    (model / "onnx" / "broken.onnx").write_bytes(b"not an ONNX graph\n" * 512)
    broken = {**cpu, "device": "webgpu", "dtype": "fp32", "graph": "broken", "files": {
        **{name: size for name, size in cpu["files"].items() if not name.startswith("onnx/")},
        "onnx/broken.onnx": 512 * 18}}
    write_manifest(out / "broken.json", {**manifest, "variants": {"cpu": cpu, "gpu": broken}})
    write_manifest(out / "broken-only.json", {**manifest, "variants": {"gpu": broken}})

    write_manifest(out / "format2.json", {**manifest, "format": 2})
    write_manifest(out / "format0.json", {**manifest, "format": 0})
    write_manifest(out / "recipe.json", {**manifest, "recipe": "nli-labels"})
    write_manifest(out / "task.json", {**manifest, "task": "search"})
    write_manifest(out / "nodata.json", {key: value for key, value in manifest.items() if key != "data"})
    write_manifest(out / "nolabels.json", {key: value for key, value in manifest.items() if key != "labels"})
    # The file of the centres is not on the server: an HTTP 404, which a new try does not mend.
    write_manifest(out / "missing-centres.json", {**manifest, "data": {
        "centres": {**manifest["data"]["centres"], "path": "data/nothing.bin"}}})


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


@contextmanager
def answering(golden: Golden, pattern: str, handler) -> Iterator[None]:
    """Let ``handler`` answer the requests that match ``pattern`` from the page and its threads."""
    golden.bench.context.route(pattern, handler)
    try:
        yield
    finally:
        golden.bench.context.unroute(pattern, handler)


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


def assert_scores_match(golden: Golden, results: list[dict[str, Any]]) -> None:
    """Every answer lists every label, the best first, with the scores Python computes."""
    for result, row in zip(results, golden.matrix):
        scores = result["scores"]
        assert sorted(item["label"] for item in scores) == sorted(golden.names)
        assert max(abs(item["score"] - row[golden.names.index(item["label"])]) for item in scores) < 1e-4
        values = [item["score"] for item in scores]
        assert values == sorted(values, reverse=True)
        assert (result["label"], result["score"]) == (scores[0]["label"], scores[0]["score"])
        assert set(result) == {"label", "score", "scores"}


def phases_of(progress: list[dict[str, Any]]) -> list[str]:
    """The phases in the order they came, a repeated one counted once."""
    phases: list[str] = []
    for event in progress:
        if not phases or phases[-1] != event["phase"]:
            phases.append(event["phase"])
    return phases


def assert_progress_ends_at_100(progress: list[dict[str, Any]], total: int, files: set[str]) -> None:
    assert progress, "onProgress was never called"
    last = progress[-1]
    assert last["percent"] == 100
    assert last["allLoaded"] == last["allTotal"] == total
    assert {event["file"] for event in progress} == files
    for event in progress:
        assert 0 <= event["loaded"] <= event["total"]
        assert 0 <= event["allLoaded"] <= event["allTotal"]
        assert event["phase"] in PROGRESS_PHASES
        assert set(event) == {"phase", "file", "loaded", "total", "allLoaded", "allTotal", "percent"}


def assert_the_phases_follow_the_bytes(progress: list[dict[str, Any]], *, retried: bool = False) -> None:
    """"download" while bytes come, then "prepare", which ends the load; after a failed card
    the processor's files may bring "download" back once."""
    assert phases_of(progress) in (["download", "prepare"],
                                   *([["download", "prepare", "download", "prepare"]] if retried else []))
    for event in progress:
        if event["phase"] == "prepare":
            assert event["percent"] == 100 and event["allLoaded"] == event["allTotal"]
    first_prepare = next(event for event in progress if event["phase"] == "prepare")
    before = progress[: progress.index(first_prepare)]
    assert before and any(event["percent"] < 100 for event in before), "the bytes came in one step"
    assert progress[-1]["phase"] == "prepare"


# --- the answers ----------------------------------------------------------------------------------


def test_labels_and_scores_on_held_out_match_python_and_the_first_load_is_counted(golden):
    page, events = open_page(golden)
    mark = len(golden.server.requests)
    out = call(golden, page, events, LOAD_AND_RUN, {
        "manifest": "cifarium.json", "device": "auto", "texts": golden.texts, "clear": True})

    assert_labels_match(golden, out["results"])
    assert_scores_match(golden, out["results"])
    assert_progress_ends_at_100(out["progress"], golden.bytes, golden.files)
    assert_the_phases_follow_the_bytes(out["progress"])
    info = out["info"]
    assert info["state"] == "ready"
    assert (info["variant"], info["device"], info["dtype"]) == ("cpu", "wasm", "q8")
    assert info["fallback"] is None
    assert info["crossOriginIsolated"] is True
    assert info["threads"] == expected_threads(golden)
    assert info["bytes"] == golden.bytes
    assert info["labels"] == golden.records
    assert all(label["description"] for label in info["labels"]), "the golden task describes its labels"
    assert info["runtime"] == info["cifarium"] == cifarium.__version__
    assert (info["model"], info["transformers"]) == ("Xenova/all-MiniLM-L6-v2", vendor.TRANSFORMERS_VERSION)
    assert info["onnxruntimeWeb"] == vendor.ORT_WEB_VERSION
    # The runtime downloads the executor itself and the library does not fetch it again.
    assert requests_for(golden, "/" + vendor.WASM_BINARY_FILE, mark) == 1
    page.close()


def test_info_has_the_labels_before_load_and_the_same_fields_after_and_the_module_its_exports(golden):
    page, events = open_page(golden)
    out = call(golden, page, events, """async () => {
      const model = await runtime.createCifarium('cifarium.json');
      const before = model.info();
      await model.load({ device: 'wasm' });
      const after = model.info();
      await model.dispose();
      return {
        before, after, exports: Object.keys(runtime).sort(), codes: runtime.ERROR_CODES,
        reasons: runtime.FALLBACK_REASONS, phases: runtime.PROGRESS_PHASES,
        frozen: Object.isFrozen(runtime.ERROR_CODES) && Object.isFrozen(runtime.FALLBACK_REASONS),
      };
    }""")

    assert sorted(out["before"]) == sorted(out["after"]) == INFO_KEYS
    assert out["before"]["labels"] == out["after"]["labels"] == golden.records
    assert out["before"]["state"] == "created" and out["after"]["state"] == "ready"
    for key in ("variant", "device", "dtype", "threads", "crossOriginIsolated", "bytes", "fallback"):
        assert out["before"][key] is None, key
    assert out["exports"] == sorted(EXPORTS)
    assert (out["codes"], out["reasons"], out["phases"]) == (ERROR_CODES, FALLBACK_REASONS, PROGRESS_PHASES)
    assert out["frozen"] is True
    page.close()


def test_a_long_text_is_cut_by_the_tokenizer_and_not_refused(golden):
    page, events = open_page(golden)
    out = call(golden, page, events, """async () => {
      const model = await runtime.createCifarium('cifarium.json');
      await model.load({ device: 'wasm' });
      // Far more than 512 tokens; only the end differs.
      const head = Array.from({ length: 1500 }, (_, i) => `parcel${i % 7}`).join(' ');
      const a = await model.run(`${head} the refund never came back`);
      const b = await model.run(`${head} my app crashes at login`);
      const tail = await model.run('the refund never came back');
      await model.dispose();
      return { a, b, tail };
    }""")

    assert out["a"] == out["b"], "what lies beyond the tokenizer's limit must not count"
    assert out["a"] != out["tail"]
    assert len(out["a"]["scores"]) == len(golden.names)
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
    assert_the_phases_follow_the_bytes(again["progress"])
    assert requests_for(golden, ".wasm", mark) == 0
    assert requests_for(golden, ".onnx", mark) == 0
    page.close()


# --- the graphics card and why it is not used -----------------------------------------------------


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
        assert info["fallback"]["reason"] in ("no-webgpu", "no-adapter", "software-adapter")
        assert info["fallback"]["message"]
        assert_labels_match(golden, out["results"])
        assert_scores_match(golden, out["results"])
    page.close()


def test_a_failing_gpu_variant_hands_over_to_the_processor(golden):
    page, events = open_page(golden)
    out = call(golden, page, events, LOAD_AND_RUN, {
        "manifest": "broken.json", "device": "webgpu", "texts": golden.texts, "clear": True})
    info = out["info"]
    assert (info["variant"], info["device"]) == ("cpu", "wasm")
    assert info["fallback"]["reason"] == "failed"
    assert info["fallback"]["message"].startswith("the graphics card failed: ")
    assert set(info["fallback"]) == {"reason", "message"}
    assert info["bytes"] == golden.bytes
    assert_labels_match(golden, out["results"])
    # What the card's attempt downloaded stays counted; without an adapter it downloads nothing.
    assert out["progress"][-1]["percent"] == 100
    assert out["progress"][-1]["allTotal"] >= golden.bytes
    assert_the_phases_follow_the_bytes(out["progress"], retried=True)
    page.close()


def test_the_page_asking_for_the_processor_is_a_fallback_with_the_reason_requested(golden):
    page, events = open_page(golden)
    out = call(golden, page, events, LOAD_AND_RUN, {
        "manifest": "gpu.json", "device": "wasm", "texts": golden.texts[:3], "clear": False})

    assert out["info"]["variant"] == "cpu"
    assert out["info"]["fallback"] == {"reason": "requested", "message": "the page asked for the processor"}
    page.close()


@pytest.mark.parametrize("manifest, spec, reason", [
    ("gpu.json", None, "no-webgpu"),
    ("gpu.json", {"adapter": None}, "no-adapter"),
    ("gpu.json", {"adapter": None, "throws": True}, "no-adapter"),
    ("gpu.json", {"adapter": {"software": True, "features": []}}, "software-adapter"),
    ("gpu-fp16.json", {"adapter": {"software": False, "features": []}}, "no-shader-f16"),
], ids=["no-webgpu", "no-adapter", "adapter-throws", "software-adapter", "no-shader-f16"])
def test_every_reason_the_card_is_not_used_is_named_for_a_program(golden, manifest, spec, reason):
    # The thread sees the WebGPU of `spec` instead of the browser's own (see TEST_PAGE).
    page, events = open_page(golden)
    out = call(golden, page, events, """async ({ manifest, spec }) => {
      window.fakeGpu(spec);
      const model = await runtime.createCifarium(manifest);
      const info = await model.load({ device: 'auto' });
      const result = await model.run('I lost my card');
      await model.dispose();
      return { info, label: result.label };
    }""", {"manifest": manifest, "spec": spec})

    assert out["info"]["variant"] == "cpu"
    assert out["info"]["fallback"]["reason"] == reason
    assert out["info"]["fallback"]["reason"] in FALLBACK_REASONS
    assert out["info"]["fallback"]["message"]
    if spec and spec.get("throws"):
        assert "requestAdapter() failed" in out["info"]["fallback"]["message"]
    assert out["label"] in golden.names
    page.close()


# --- errors ----------------------------------------------------------------------------------------


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
        format0: await code(() => runtime.createCifarium('format0.json')),
        recipe: await code(() => runtime.createCifarium('recipe.json')),
        task: await code(() => runtime.createCifarium('task.json')),
        nodata: await code(() => runtime.createCifarium('nodata.json')),
        nolabels: await code(() => runtime.createCifarium('nolabels.json')),
        beforeLoad: await code(() => model.run('hello')),
        notText: await code(() => model.run(42)),
        emptyBeforeLoad: await code(() => model.run('')),
        device: await code(() => model.load({ device: 'cuda' })),
        state: model.info().state,
      };
      // A run made while load() is still going waits for it.
      const loading = model.load();
      const early = await model.run('I lost my card');
      await loading;
      result.early = early.label;
      result.again = (await model.load()).state;
      result.empty = await code(() => model.run(''));
      result.spaces = await code(() => model.run('  \\n\\t '));
      await model.dispose();
      result.afterDispose = await code(() => model.load());
      return result;
    }""")
    assert out == {
        "missing": "manifest", "format": "unsupported", "format0": "manifest", "recipe": "unsupported",
        "task": "unsupported", "nodata": "manifest", "nolabels": "manifest", "beforeLoad": "not-loaded",
        "notText": "input", "emptyBeforeLoad": "input", "device": "input", "state": "created",
        "early": out["early"], "again": "ready", "empty": "input", "spaces": "input",
        "afterDispose": "disposed",
    }
    assert out["early"] in golden.names
    page.close()


def test_a_manifest_that_does_not_download_is_network_and_one_the_server_lacks_is_manifest(golden):
    def abort(route):
        route.abort("connectionreset")

    def busy(route):
        route.fulfill(status=503, body="Service Unavailable")

    page, events = open_page(golden)
    with answering(golden, "**/offline.json", abort), answering(golden, "**/busy.json", busy):
        out = call(golden, page, events, """async () => {
          const failure = async (url) => {
            try { await runtime.createCifarium(url); return null; } catch (e) {
              return { code: e.code, cause: e.cause ? e.cause.name : null, message: e.message };
            }
          };
          return {
            offline: await failure('offline.json'),
            busy: await failure('busy.json'),
            absent: await failure('missing.json'),
          };
        }""")

    assert (out["offline"]["code"], out["busy"]["code"], out["absent"]["code"]) == ("network", "network", "manifest")
    assert out["offline"]["cause"] == "TypeError", "the failed fetch is the cause"
    assert out["busy"]["cause"] is None and "HTTP 503" in out["busy"]["message"]
    page.close()


def test_a_file_that_does_not_download_is_network_and_a_new_load_can_succeed(golden):
    broken = {"on": True}

    def centres(route):
        route.abort("connectionreset") if broken["on"] else route.continue_()

    page, events = open_page(golden)
    with answering(golden, "**/data/labels.bin", centres):
        failed = call(golden, page, events, FAILED_LOAD, {"manifest": "cifarium.json", "clear": False})
        assert failed["code"] == "network" and failed["isCifariumError"]
        assert failed["cause"] == {"isError": True, "name": "TypeError"}
        assert "labels.bin" in failed["message"]
        assert failed["state"] == "failed"

        broken["on"] = False
        again = call(golden, page, events, LOAD_AGAIN)
    assert again == {"state": "ready", "label": again["label"], "scores": len(golden.names)}
    page.close()


def test_a_file_the_library_downloads_that_the_server_cannot_give_now_is_network(golden):
    # The model's graph goes through the library, which hides a failed download behind its own
    # error: the runtime sees only that the server said "try later".
    busy = {"on": True}

    def graph(route):
        if busy["on"]:
            route.fulfill(status=503, body="Service Unavailable")
        else:
            route.continue_()

    page, events = open_page(golden)
    with answering(golden, "**/onnx/model_quantized.onnx", graph):
        failed = call(golden, page, events, FAILED_LOAD, {"manifest": "cifarium.json", "clear": True})
        assert (failed["code"], failed["state"]) == ("network", "failed"), failed
        assert failed["cause"] and failed["cause"]["isError"]

        busy["on"] = False
        again = call(golden, page, events, LOAD_AGAIN)
    assert again["state"] == "ready"
    page.close()


def test_a_file_the_server_does_not_have_is_load_because_asking_again_changes_nothing(golden):
    page, events = open_page(golden)

    failed = call(golden, page, events, FAILED_LOAD, {"manifest": "missing-centres.json", "clear": False})

    assert (failed["code"], failed["state"]) == ("load", "failed")
    assert "HTTP 404" in failed["message"] and "nothing.bin" in failed["message"]
    page.close()


def test_a_model_that_does_not_start_anywhere_is_load_and_keeps_the_error_of_the_library(golden):
    # Every code of ERROR_CODES is provoked in this file but "run": it is for a model that fails
    # on a text after it has started, and no text does that to all-MiniLM. That the runtime
    # raises it is read in the source (tests/test_contracts.py).
    page, events = open_page(golden)
    # The only variant is a card's variant whose graph is not a graph: nothing is left to try.
    failed = call(golden, page, events, """async () => {
      const model = await runtime.createCifarium('broken-only.json');
      const error = await model.load({ device: 'webgpu' }).then(() => null, (e) => e);
      return { code: error.code, message: error.message, cause: error.cause?.name ?? null, state: model.info().state };
    }""")

    assert (failed["code"], failed["state"]) == ("load", "failed")
    assert "The model did not start: gpu (webgpu, fp32)" in failed["message"]
    assert failed["cause"] is not None, "the error of the library stays as the cause"
    page.close()

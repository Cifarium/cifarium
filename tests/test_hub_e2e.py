"""`host: hub` in the real Chrome: a folder without `models/`, whose page takes the model's files from
Hugging Face at the revision of the manifest, and the browser's store after rebuilds.

The end-to-end run is not part of the default run: `pytest -m e2e`. It needs the network to Hugging
Face, from the page as well as from the build. Into the cache of cifarium it downloads nothing the
files of the cache key of .github/workflows/tests.yml do not already name: all-MiniLM-L6-v2 is the
model of the English build (tests/test_build_e2e.py), and the model of zero-shot is the one
tests/test_zero_shot_e2e.py names (`MODEL`, imported from there, so that file stays the one that
names it). What the page downloads from Hugging Face goes into the Chrome profile, which is never
kept: every run downloads it again (about 120 MB).
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pytest
from publishable import example_leaks
from test_build_e2e import ENGLISH, write_task
from test_page_e2e import check_page, expected_threads, threads_of, use_page
from test_zero_shot_e2e import MODEL as ZERO_SHOT_MODEL

from cifarium import build, bundle, catalog, check, config, exitcodes, hosting, paths, vendor
from cifarium.bench import open_bench
from cifarium.server import SiteServer

pytestmark = pytest.mark.e2e

NAMES = ["shipping", "returns", "technical"]
CHECK_PAGE = {"/": check.PAGE, "/index.html": check.PAGE}
STORE = "transformers-cache"
"""The library's store in Cache API (`env.cacheKey`), where the runtime keeps the executor too."""


@dataclass(frozen=True)
class Built:
    code: int
    out: str
    err: str
    project: Path
    folder: Path

    @property
    def manifest(self) -> dict:
        return json.loads((self.folder / bundle.MANIFEST).read_text(encoding="utf-8"))


def candidate(model_id: str) -> catalog.Candidate:
    return next(c for c in catalog.candidates("classify") if c.entry.id == model_id)


def build_on_hub(tmp_path_factory, name: str, only: catalog.Candidate | None = None) -> Built:
    """`cifarium build` with `host: hub` on the English set; ``only`` makes the catalog that one model."""
    project = tmp_path_factory.mktemp(name)
    results = tmp_path_factory.mktemp(name + "-results")
    settings = write_task(project, ENGLISH, "host: hub\n")
    out, err = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(paths, "results_dir", lambda cache=None: results)
        if only is not None:
            patch.setattr(catalog, "candidates", lambda task: (only,))
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = build.run(settings)
    return Built(code, out.getvalue(), err.getvalue(), project, project / "cifarium-out")


@pytest.fixture(scope="module")
def vectors(tmp_path_factory) -> Built:
    """The English build, which chooses all-MiniLM (embed-centroid), with `host: hub`."""
    return build_on_hub(tmp_path_factory, "hub-vectors")


@pytest.fixture(scope="module")
def entailment(tmp_path_factory) -> Built:
    """The English build on a catalog of the zero-shot model alone, with `host: hub`."""
    return build_on_hub(tmp_path_factory, "hub-zero-shot", candidate(ZERO_SHOT_MODEL))


@pytest.fixture(scope="module")
def bench(vectors, entailment):
    """One Chrome for the module, started after both builds have closed theirs: the cache holds one
    Chrome profile, and two browsers at once would wipe each other's."""
    vendor.ensure_engine()
    with open_bench() as running:
        yield running


@contextlib.contextmanager
def watching(bench) -> Iterator[list[tuple[str, str]]]:
    """Every request of the browser's pages and their threads while the block runs: (method, url)."""
    seen: list[tuple[str, str]] = []

    def note(request) -> None:
        # The antivirus of the author's laptop writes a script of its own into every page served over
        # plain http (tests/test_page_e2e.py); its requests are not the page's.
        if not (urlsplit(request.url).hostname or "").endswith(".kaspersky-labs.com"):
            seen.append((request.method, request.url))

    bench.context.on("request", note)
    try:
        yield seen
    finally:
        bench.context.remove_listener("request", note)


@contextlib.contextmanager
def this_bench(bench):
    """`check.run` opens a browser of its own; the module has one, and Playwright's sync API starts one
    per thread."""

    @contextlib.contextmanager
    def open_this_one(*, headed: bool = False):
        yield bench

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(check, "open_bench", open_this_one)
        yield


def is_hugging_face(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    return urlsplit(url).scheme == "https" and (
        host in ("huggingface.co", "hf.co") or host.endswith((".huggingface.co", ".hf.co")))


# --- the folder ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("which, model, recipe", [("vectors", "Xenova/all-MiniLM-L6-v2", "embed-centroid"),
                                                  ("entailment", None, "zero-shot")])
def test_a_hub_build_writes_a_folder_without_the_model_and_ends_with_code_0(request, which, model, recipe):
    built: Built = request.getfixturevalue(which)
    assert built.code == exitcodes.DONE, built.out + built.err

    manifest = built.manifest
    assert (manifest["recipe"], manifest["model"]["host"], manifest["model"]["path"]) == (recipe, "hub", None)
    assert model is None or manifest["model"]["repo"] == model
    assert not (built.folder / "models").exists()
    assert "of it from Hugging Face" in built.out and "host: hub: the model's files stay on Hugging Face" in built.out
    document = json.loads((built.folder / bundle.REPORT_JSON).read_text(encoding="utf-8"))
    assert document["settings"]["host"] == "hub"
    # The weight of a first visit is the same as with the model in the folder: the bytes come from elsewhere.
    assert document["chosen"]["firstLoad"]["bytes"] == bundle.first_load(manifest, "cpu")[0]
    # On the disk of the site: the engine and less than a megabyte besides it.
    on_disk = sum(file.stat().st_size for file in built.folder.rglob("*") if file.is_file())
    assert on_disk < sum(vendor.ENGINE_BYTES.values()) + 1_000_000
    assert example_leaks(built.folder, [text for text, _ in ENGLISH]) == []


# --- the page --------------------------------------------------------------------------------------------


def test_the_example_page_names_the_label_with_the_model_from_hugging_face_at_the_revision(bench, vectors):
    folder, manifest = vectors.folder, vectors.manifest
    repo, revision = manifest["model"]["repo"], manifest["model"]["revision"]
    base = bundle.model_base(manifest)

    with SiteServer(folder, hosting.headers_of(folder, "netlify")) as server:
        with watching(bench) as first:
            shown = use_page(bench, server.url + "/")
        with watching(bench) as second:
            again = use_page(bench, server.url + "/")
        origin = server.url

    check_page(shown, folder)
    check_page(again, folder)
    # The headers of the hosting file isolate the page with the model on another site: Hugging Face
    # answers with CORS, which `require-corp` lets through (ARCHITECTURE.md, section 1.6).
    assert shown["isolated"] is True and threads_of(shown["info"]) == expected_threads(bench)
    assert {method for method, _ in first} == {"GET"}
    away = [url for _, url in first if not url.startswith(origin + "/")]
    # Every file of the variant from Hugging Face at the revision of the manifest, and nothing at `main`:
    # the library would ask `main` for the list and the sizes if the runtime gave it only `revision`.
    wanted = {base + name for name in manifest["variants"]["cpu"]["files"]}
    assert wanted <= set(away), sorted(wanted - set(away))
    assert all(is_hugging_face(url) for url in away), [url for url in away if not is_hugging_face(url)]
    on_hub = [url for url in away if urlsplit(url).hostname == "huggingface.co"]
    assert all(url.startswith(base) or urlsplit(url).path.startswith(f"/api/resolve-cache/models/{repo}/{revision}/")
               for url in on_hub), on_hub
    assert not any("/resolve/main/" in url for url in away)
    # The site is asked only for files of its folder, which has no models/.
    asked = {url.removeprefix(origin) for _, url in first if url.startswith(origin + "/")}
    strangers = sorted(path for path in asked if path not in ("/", "/favicon.ico")
                       and not (folder / path.split("?")[0].lstrip("/")).is_file())
    assert strangers == []
    # A second visit takes every file from the browser's store: not one request to Hugging Face.
    assert [url for _, url in second if is_hugging_face(url)] == []


def test_a_manifest_with_a_host_this_runtime_does_not_know_is_refused_before_any_download(bench, vectors, tmp_path):
    folder = tmp_path / "unknown-host"
    folder.mkdir()
    for name in bundle.RUNTIME_FILES:
        shutil.copyfile(vectors.folder / name, folder / name)
    manifest = vectors.manifest
    manifest["model"]["host"] = "cdn"
    (folder / bundle.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    with SiteServer(folder, lambda path: {}, replace=CHECK_PAGE) as server:
        outcome = check.drive(bench, server.url, ["a text"])

    failure = outcome["failure"]
    assert (failure["stage"], failure["code"]) == ("create", "unsupported")
    assert 'host "cdn"' in failure["message"] and "bundle, hub" in failure["message"]


# --- check ---------------------------------------------------------------------------------------------


def test_check_passes_a_zero_shot_folder_on_hugging_face_and_says_it_needs_the_network(
        bench, entailment, monkeypatch, capsys):
    monkeypatch.chdir(entailment.project)  # the folder and the examples come from cifarium.yaml

    with this_bench(bench):
        code = check.run()

    out, err = capsys.readouterr()
    assert code == exitcodes.DONE, out + err
    assert "host: hub, the page downloads the model's files from Hugging Face" in out
    for line in ("ok    isolation", "ok    first load", "PASSED: the folder works as a site will see it."):
        assert line in out, line


def test_check_on_a_hub_folder_that_cannot_reach_hugging_face_is_the_environment(bench, vectors, capsys):
    folder = vectors.folder
    examples = vectors.project / "examples.jsonl"
    blocked: list[str] = []

    def refuse(route) -> None:
        if is_hugging_face(route.request.url):
            blocked.append(route.request.url)
            route.abort("internetdisconnected")
        else:
            route.continue_()

    bench.context.route("**/*", refuse)
    try:
        with this_bench(bench):
            code = check.run(folder, examples=examples)
    finally:
        bench.context.unroute("**/*", refuse)

    out, err = capsys.readouterr()
    assert blocked, "the page asked Hugging Face for nothing"
    assert code == exitcodes.ENVIRONMENT, out + err
    assert "FAILED: the page could not download the model from Hugging Face [network]" in err


# --- the browser's store after rebuilds --------------------------------------------------------------


def stored(bench, origin: str) -> set[str]:
    """The keys of the library's store on the site at ``origin``."""
    page = bench.context.new_page()
    try:
        page.goto(origin + "/")
        return set(page.evaluate(f"async () => (await (await caches.open('{STORE}')).keys()).map((r) => r.url)"))
    finally:
        page.close()


def put_strangers(bench, origin: str) -> set[str]:
    """Files of the store that are not this folder's: another folder of the site, another model."""
    strangers = {origin + "/elsewhere/models/x/onnx/model_quantized.onnx",
                 "https://huggingface.co/someone/else/resolve/" + "c" * 40 + "/config.json"}
    page = bench.context.new_page()
    try:
        page.goto(origin + "/")
        page.evaluate("""async (keys) => {
            const store = await caches.open('%s');
            for (const key of keys) await store.put(key, new Response('not this folder'));
        }""" % STORE, sorted(strangers))
    finally:
        page.close()
    return strangers


def visit(bench, origin: str) -> set[str]:
    """Load the model of the folder on the page of `check`, as a visitor of the site would, and say
    what the store holds afterwards."""
    outcome = check.drive(bench, origin, ["Where is my parcel?"])
    assert "result" in outcome, outcome
    return stored(bench, origin)


def test_a_rebuild_leaves_no_file_of_the_earlier_build_in_the_visitors_store(bench, vectors, tmp_path):
    # Four builds into one folder on one site, each followed by a visit: another revision, then the
    # model on Hugging Face, then back in the folder. After each visit the store holds the files of this
    # build and of nothing earlier, and what is not this folder's stays.
    real = candidate("all-minilm-l6-v2")
    other = catalog.Candidate(dataclasses.replace(real.entry, revision="0" * 40), real.recipe)
    labels = [config.Label(name) for name in NAMES]
    centres = np.frombuffer((vectors.folder / "data" / "labels.bin").read_bytes(), dtype="<f4").reshape(3, -1)
    folder = tmp_path / "site"
    names = list(real.entry.files("cpu"))
    repo = real.entry.repo

    def in_folder(origin: str, revision: str) -> set[str]:
        return {f"{origin}/models/{revision}/{repo}/{name}" for name in names}

    on_hub = {bundle.hub_address(repo, real.entry.revision, name) for name in names}
    bundle.write(folder, real, labels, centres)
    with SiteServer(folder, hosting.headers_of(folder, "netlify"), replace=CHECK_PAGE) as server:
        origin = server.url
        manifest = json.loads((folder / bundle.MANIFEST).read_text(encoding="utf-8"))
        executor = f"{origin}/{manifest['engine']['executor']['path']}"
        listing = f"{origin}/{bundle.MANIFEST}?cifarium-cached-files"

        first = visit(bench, origin)
        assert in_folder(origin, real.entry.revision) <= first and {executor, listing} <= first
        strangers = put_strangers(bench, origin)

        bundle.write(folder, other, labels, centres)
        second = visit(bench, origin)
        assert not in_folder(origin, real.entry.revision) & second, "the files of the earlier revision stayed"
        assert in_folder(origin, other.entry.revision) <= second
        assert {executor, listing} <= second and strangers <= second

        bundle.write(folder, real, labels, centres, host="hub")
        third = visit(bench, origin)
        assert on_hub <= third
        assert not in_folder(origin, other.entry.revision) & third, "the files of the folder stayed after hub"
        assert {executor, listing} <= third and strangers <= third

        bundle.write(folder, real, labels, centres)
        fourth = visit(bench, origin)
        assert not on_hub & fourth, "the files from Hugging Face stayed after the model came back into the folder"
        assert in_folder(origin, real.entry.revision) <= fourth
        assert {executor, listing} <= fourth and strangers <= fourth

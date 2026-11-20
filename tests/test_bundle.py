"""The folder of a build without a browser: layout, manifest, label data, what a rebuild keeps
and deletes, and the three hosting files. The engine and the model are small made-up files, so
the sizes of the catalog entry here are the ones of those files.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from publishable import example_leaks

import cifarium
from cifarium import bundle, catalog, hosting, paths, vendor
from cifarium.config import Label

ENGINE_SIZES = {vendor.LIBRARY_FILE: 11, vendor.WASM_LOADER_FILE: 7, vendor.WASM_BINARY_FILE: 13}
MODEL_FILES = {"config.json": 3, "tokenizer.json": 4, "tokenizer_config.json": 2}
GRAPH = {"onnx/model_quantized.onnx": 5}
REVISION = "a" * 40
LABELS = ["shipping", "returns", "technical"]
CENTRES = np.eye(3)


REPORTS = {bundle.REPORT_JSON: b'{"format": 1}\n', bundle.REPORT_MARKDOWN: b"# cifarium build report\n"}
"""Stand-ins for the files of the report (tests/test_report.py has the real ones): the folder of
a build always carries both."""

APACHE = "Apache License\nVersion 2.0, January 2004\nTEST-APACHE-BODY $notaplaceholder\n"
MIT = "MIT License\n\nCopyright (c) Test Corporation\nTEST-MIT-BODY\n"


def fill(folder: Path, sizes: dict[str, int]) -> None:
    for number, (name, size) in enumerate(sizes.items()):
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes([number + 1]) * size)


def fill_engine(folder: Path) -> None:
    """The engine folder of the cache: the three files and the two license texts :mod:`vendor` stores."""
    fill(folder, ENGINE_SIZES)
    (folder / vendor.TRANSFORMERS_LICENSE_FILE).write_text(APACHE, encoding="utf-8")
    (folder / vendor.ORT_LICENSE_FILE).write_text(MIT.replace("\n", "\r\n"), encoding="utf-8", newline="")


def entry(revision: str = REVISION, license: str = "apache-2.0") -> catalog.Entry:
    variants = {"cpu": catalog.Variant("cpu", "wasm", "q8", dict(GRAPH))}
    return catalog.Entry(Path("tiny.yaml"), "tiny", "org/tiny", revision, license, "org/source",
                         ("en",), dict(MODEL_FILES), variants, ())


def candidate(revision: str = REVISION, license: str = "apache-2.0") -> catalog.Candidate:
    recipe = catalog.Recipe("classify", "embed-centroid", {"pooling": "mean", "prefix": "", "dimensions": 3})
    return catalog.Candidate(entry(revision, license), recipe)


@dataclass
class Sources:
    engine: Path
    model: Path

    def write(self, folder: Path, who: catalog.Candidate | None = None, names=LABELS, centres=CENTRES, **more):
        """``names`` are plain names or Label objects, in the order of the settings."""
        labels = [name if isinstance(name, Label) else Label(name) for name in names]
        more.setdefault("reports", REPORTS)
        return bundle.write(folder, who or candidate(), labels, centres, engine_dir=self.engine,
                            model_dir=self.model, engine_sizes=ENGINE_SIZES, **more)


@pytest.fixture
def sources(tmp_path: Path) -> Sources:
    engine, model = tmp_path / "cache" / "engine", tmp_path / "cache" / "model"
    fill_engine(engine)
    fill(model, {**MODEL_FILES, **GRAPH})
    return Sources(engine, model)


@pytest.fixture
def out(tmp_path: Path) -> Path:
    return tmp_path / "site" / "cifarium-out"


def tree(folder: Path) -> set[str]:
    return {file.relative_to(folder).as_posix() for file in folder.rglob("*") if file.is_file()}


# --- layout and manifest -------------------------------------------------------------------------


def test_the_folder_has_the_files_of_section_7(sources, out):
    built = sources.write(out)

    engine = f"vendor/transformers-{vendor.TRANSFORMERS_VERSION}/"
    model = f"models/{REVISION}/org/tiny/"
    assert tree(out) == {
        "index.html", "cifarium.js", "cifarium-worker.js", "cifarium.json", "LICENSES.txt",
        "report.json", "report.md",
        engine + vendor.LIBRARY_FILE, engine + vendor.WASM_LOADER_FILE, engine + vendor.WASM_BINARY_FILE,
        model + "config.json", model + "tokenizer.json", model + "tokenizer_config.json",
        model + "onnx/model_quantized.onnx",
        "data/labels.bin",
        "hosting/_headers", "hosting/vercel.json", "hosting/nginx.conf",
    }
    assert set(built.files) == tree(out)
    assert built.size_bytes == sum((out / name).stat().st_size for name in tree(out))
    assert not list(out.rglob("*.part"))


def test_the_manifest_is_the_one_the_runtime_reads(sources, out):
    built = sources.write(out)

    manifest = json.loads((out / "cifarium.json").read_text(encoding="utf-8"))
    assert manifest == built.manifest
    engine = f"vendor/transformers-{vendor.TRANSFORMERS_VERSION}/"
    assert manifest == {
        "format": 1,
        "cifarium": cifarium.__version__,
        "task": "classify",
        "recipe": "embed-centroid",
        "labels": [{"name": name, "description": None} for name in LABELS],
        "model": {"repo": "org/tiny", "revision": REVISION, "license": "apache-2.0", "host": "bundle",
                  "path": f"models/{REVISION}/"},
        "variants": {"cpu": {"device": "wasm", "dtype": "q8", "graph": "model", "externalData": 0,
                             "files": {**MODEL_FILES, **GRAPH}}},
        "params": {"pooling": "mean", "prefix": ""},
        "engine": {
            "transformers": vendor.TRANSFORMERS_VERSION,
            "onnxruntimeWeb": vendor.ORT_WEB_VERSION,
            "library": {"path": engine + vendor.LIBRARY_FILE, "bytes": 11},
            "loader": {"path": engine + vendor.WASM_LOADER_FILE, "bytes": 7},
            "executor": {"path": engine + vendor.WASM_BINARY_FILE, "bytes": 13},
        },
        "data": {"centres": {"path": "data/labels.bin", "bytes": 36}},
    }
    total, files = bundle.first_load(manifest, "cpu")
    assert total == built.first_load_bytes == 11 + 7 + 13 + 9 + 5 + 36
    assert files <= tree(out)
    for path in files:
        assert (out / path).stat().st_size > 0


def test_the_weight_of_a_first_visit_is_one_number_for_the_selection_the_build_and_the_manifest(sources, out):
    from cifarium import selection

    labels = [Label(name) for name in LABELS]
    counted_before_the_build = selection.first_load_bytes(candidate(), labels, ENGINE_SIZES)

    built = sources.write(out)

    # engine 11 + 7 + 13, the files of the model 9 + 5, the centres of 3 labels of 3 numbers 36.
    assert counted_before_the_build == built.first_load_bytes == 11 + 7 + 13 + 9 + 5 + 36
    assert bundle.first_load(built.manifest, "cpu")[0] == built.first_load_bytes
    # Not the page, not the runtime, not the manifest, not the licenses: a visitor's first visit
    # downloads the page and the runtime too, but their weight is not what the limit is about.
    assert built.size_bytes > built.first_load_bytes


def test_a_model_with_two_variants_counts_only_the_variant_asked_for(sources, out):
    gpu = catalog.Variant("gpu", "webgpu", "q4f16", {"onnx/model_q4f16.onnx": 4})
    two = catalog.Entry(Path("tiny.yaml"), "tiny", "org/tiny", REVISION, "apache-2.0", "org/source", ("en",),
                        dict(MODEL_FILES), {**entry().variants, "gpu": gpu}, ())
    who = catalog.Candidate(two, candidate().recipe)
    manifest = bundle.manifest_of(who, [Label(name) for name in LABELS], 36, variants=("cpu", "gpu"),
                                  engine_sizes=ENGINE_SIZES)

    cpu, gpu_total = bundle.first_load(manifest, "cpu")[0], bundle.first_load(manifest, "gpu")[0]

    assert cpu == 11 + 7 + 13 + 9 + 5 + 36
    assert gpu_total == 11 + 7 + 13 + 9 + 4 + 36


def test_the_manifest_format_is_the_one_the_runtime_reads():
    source = (paths.web_dir() / "runtime" / "cifarium.js").read_text(encoding="utf-8")
    match = re.search(r"^export const MANIFEST_FORMAT = (\d+);$", source, re.MULTILINE)
    assert match and int(match.group(1)) == bundle.FORMAT


def test_the_labels_are_in_the_manifest_with_their_descriptions_and_the_centres_are_float32_rows(sources, out):
    labels = [Label("доставка", "Где посылка и когда её привезут"), Label("возврат"), Label("сбой", "")]
    centres = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, -1.0]])

    sources.write(out, names=labels, centres=centres)

    raw = (out / "cifarium.json").read_bytes()
    assert "доставка".encode() in raw, "text stays text, not \\u escapes"
    assert json.loads(raw)["labels"] == [
        {"name": "доставка", "description": "Где посылка и когда её привезут"},
        {"name": "возврат", "description": None},
        {"name": "сбой", "description": ""},
    ]
    assert not (out / "data" / "labels.json").exists()
    rows = np.frombuffer((out / "data/labels.bin").read_bytes(), dtype="<f4").reshape(3, 2)
    assert np.array_equal(rows, centres.astype("<f4"))


def test_the_data_of_the_manifest_is_the_files_of_the_recipe(sources, out):
    manifest = sources.write(out).manifest

    # embed-centroid: the centres of the labels, nothing else. A recipe that needs other files
    # names them under `data` in the same way.
    assert list(manifest["data"]) == ["centres"]
    assert bundle.owned(manifest) >= {manifest["data"]["centres"]["path"]}


def test_runtime_and_page_are_the_package_files_byte_for_byte(sources, out):
    sources.write(out)

    for name in ("cifarium.js", "cifarium-worker.js"):
        assert (out / name).read_bytes() == (paths.web_dir() / "runtime" / name).read_bytes()
    assert (out / "index.html").read_bytes() == (paths.web_dir() / "templates" / "index.html").read_bytes()
    for name in ("_headers", "vercel.json", "nginx.conf"):
        assert (out / "hosting" / name).read_bytes() == (paths.web_dir() / "templates" / "hosting" / name).read_bytes()


def test_the_license_file_names_every_part_and_has_no_placeholder_left(sources, out):
    sources.write(out, candidate(license="gemma"))

    text = (out / "LICENSES.txt").read_bytes().decode("utf-8")
    assert "\r" not in text
    assert not re.search(r"\$(?:cifarium|transformers|onnxruntime_web|engine_path|model_path|repo|revision|"
                         r"license|license_source|model_notice|transformers_license|onnxruntime_license)\b", text)
    assert f"Transformers.js {vendor.TRANSFORMERS_VERSION}" in text and "Apache-2.0" in text
    assert f"ONNX Runtime Web {vendor.ORT_WEB_VERSION}" in text and "MIT" in text
    assert "License: gemma" in text
    assert "https://huggingface.co/org/source" in text
    assert f"https://huggingface.co/org/tiny at revision {REVISION}" in text
    assert f"models/{REVISION}/org/tiny/" in text


def test_the_license_file_holds_the_full_texts_of_the_licenses_of_the_engine(sources, out):
    sources.write(out)

    text = (out / "LICENSES.txt").read_text(encoding="utf-8")
    # Both texts whole and as the engine folder stores them (one with CRLF there, LF here), and a
    # dollar sign in a license text is not taken for a placeholder.
    assert APACHE.strip("\n") in text
    assert MIT.strip("\n") in text
    assert "TEST-APACHE-BODY $notaplaceholder" in text
    assert text.index("Apache License\nVersion 2.0") < text.index("MIT License\n\nCopyright (c) Test Corporation")
    assert text.endswith(MIT.strip("\n") + "\n")


def test_the_notice_of_the_model_goes_into_the_license_file_as_the_catalog_has_it(sources, out):
    gemma = catalog.Entry(Path("g.yaml"), "tiny", "org/tiny", REVISION, "gemma", "org/source", ("en",),
                          dict(MODEL_FILES), entry().variants, (),
                          "Gemma is provided under and subject to the Gemma Terms of Use found at x\nSecond line")
    who = catalog.Candidate(gemma, candidate().recipe)

    sources.write(out, who)

    text = (out / "LICENSES.txt").read_text(encoding="utf-8")
    assert ("  Taken from https://huggingface.co/org/tiny at revision " + REVISION + ".\n\n"
            "  The notice that goes with the model, from its terms or its card:\n"
            "    Gemma is provided under and subject to the Gemma Terms of Use found at x\n"
            "    Second line\n\n"
            "When you pass the files on") in text


def test_the_models_of_the_catalog_that_need_a_notice_get_it_in_their_license_file(sources):
    labels = [Label(name) for name in LABELS]
    with_notice = {c.entry.id: c for c in catalog.candidates("classify") if c.entry.notice}

    assert set(with_notice) == {"embeddinggemma-300m", "mobilebert-uncased-mnli"}
    for who in with_notice.values():
        manifest = bundle.manifest_of(who, labels, bundle.centres_bytes(who, 3))
        text = bundle.licenses_text(manifest, who.entry, sources.engine).decode("utf-8")
        for line in who.entry.notice.splitlines():
            assert f"    {line}".rstrip() in text, (who.entry.id, line)
    gemma = bundle.licenses_text(
        bundle.manifest_of(with_notice["embeddinggemma-300m"], labels, 0),
        with_notice["embeddinggemma-300m"].entry, sources.engine).decode("utf-8")
    # The sentence the terms of Gemma ask to ship with every distribution of the model.
    assert "Gemma is provided under and subject to the Gemma Terms of Use found at gemma.ai/resources/terms" in gemma
    assert "License: gemma" in gemma


def test_a_model_without_a_notice_has_no_notice_paragraph(sources, out):
    sources.write(out)

    assert "The notice that goes with the model" not in (out / "LICENSES.txt").read_text(encoding="utf-8")


def test_the_file_of_licenses_is_the_same_bytes_every_time(sources, tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    sources.write(first)
    sources.write(second)

    assert (first / "LICENSES.txt").read_bytes() == (second / "LICENSES.txt").read_bytes()


def test_an_engine_folder_without_the_license_texts_stops_the_build_and_says_what_to_do(sources, out):
    (sources.engine / vendor.ORT_LICENSE_FILE).unlink()

    with pytest.raises(bundle.BundleError, match=r"LICENSE-onnxruntime-web\.txt is missing: delete the folder"):
        sources.write(out)

    assert not (out / "cifarium.json").exists()


# --- what the sources must be -------------------------------------------------------------------


def test_a_file_of_another_size_than_the_catalog_names_stops_the_build(sources, out):
    (sources.model / "onnx" / "model_quantized.onnx").write_bytes(b"short")  # 5 bytes: right
    (sources.model / "tokenizer.json").write_bytes(b"too long for the catalog")

    with pytest.raises(bundle.BundleError, match=r"tokenizer\.json is 24 bytes, but 4 are expected"):
        sources.write(out)

    assert not (out / "cifarium.json").exists()


def test_a_missing_file_stops_the_build_and_names_it(sources, out):
    (sources.engine / vendor.WASM_BINARY_FILE).unlink()

    with pytest.raises(bundle.BundleError, match=r"ort-wasm-simd-threaded\.asyncify\.wasm is missing"):
        sources.write(out)

    assert not (out / "cifarium.json").exists()


@pytest.mark.parametrize(
    "names, centres, problem",
    [(["one"], np.eye(1), "at least two"),
     (["a", "a"], np.eye(2), "all different"),
     (["a", "b", "c"], np.eye(2), "one centre each"),
     (["a", "b"], np.array([1.0, 0.0]), "one centre each"),
     (["a", "b"], np.array([[1.0, np.nan], [0.0, 1.0]]), "not a finite number")],
)
def test_labels_and_centres_that_do_not_fit_are_refused_before_anything_is_written(sources, out, names, centres,
                                                                                    problem):
    with pytest.raises(bundle.BundleError, match=problem):
        sources.write(out, names=names, centres=centres)

    assert not out.exists()


def test_a_recipe_without_a_folder_yet_and_a_missing_variant_are_refused(sources, out):
    # Both recipes of the catalog have a folder; a recipe that comes later has none until it is written.
    other = catalog.Candidate(entry(), catalog.Recipe("classify", "chat", {}))
    with pytest.raises(bundle.BundleError, match="the recipe chat is not written yet"):
        sources.write(out, other)
    with pytest.raises(bundle.BundleError, match="tiny has no gpu variant"):
        sources.write(out, variants=("cpu", "gpu"))
    assert not out.exists()
    assert bundle.RECIPES == set(catalog.RECIPES)


# --- the recipe zero-shot --------------------------------------------------------------------------


def zero_shot(revision: str = REVISION) -> catalog.Candidate:
    return catalog.Candidate(entry(revision), catalog.Recipe("classify", "zero-shot", {}))


def test_the_folder_of_zero_shot_has_no_centres_and_the_hypotheses_are_in_the_manifest(sources, out):
    built = sources.write(out, zero_shot(), names=[Label("shipping", "where it is"), Label("top_up"), Label("tech")],
                          centres=None)

    engine = f"vendor/transformers-{vendor.TRANSFORMERS_VERSION}/"
    assert "data/labels.bin" not in tree(out) and not (out / "data").exists()
    assert tree(out) >= {"cifarium.json", "cifarium.js", "cifarium-worker.js", "index.html", "LICENSES.txt",
                         engine + vendor.WASM_BINARY_FILE, f"models/{REVISION}/org/tiny/onnx/model_quantized.onnx"}
    manifest = json.loads((out / "cifarium.json").read_text(encoding="utf-8"))
    assert manifest == built.manifest
    assert manifest["recipe"] == "zero-shot" and manifest["data"] == {}
    # One finished sentence for each label in the order of the labels, from the bench's rule: the
    # name with spaces for underscores, no description.
    assert manifest["params"] == {"hypotheses": ["This example is about shipping.", "This example is about top up.",
                                                 "This example is about tech."]}
    assert [label["name"] for label in manifest["labels"]] == ["shipping", "top_up", "tech"]
    assert set(built.files) == tree(out)


def test_the_hypotheses_of_the_manifest_are_those_the_bench_asks_in_the_language_of_the_labels(sources, out):
    from cifarium.bench import hypotheses

    labels = [Label("доставка"), Label("возврат"), Label("сбой")]

    built = sources.write(out, zero_shot(), names=labels, centres=None)

    assert built.manifest["params"]["hypotheses"] == hypotheses(labels) == [
        "Этот текст про доставка.", "Этот текст про возврат.", "Этот текст про сбой."]


def test_the_first_load_of_zero_shot_is_the_engine_and_the_files_of_the_model_and_nothing_else(sources, out):
    built = sources.write(out, zero_shot(), centres=None)

    engine = sum(ENGINE_SIZES.values())
    model = sum({**MODEL_FILES, **GRAPH}.values())
    assert bundle.first_load(built.manifest, "cpu")[0] == built.first_load_bytes == engine + model
    # What the selection weighed before there was a folder is the same number.
    assert bundle.manifest_of(zero_shot(), [Label(name) for name in LABELS], 0,
                              engine_sizes=ENGINE_SIZES)["data"] == {}
    assert bundle.centres_bytes(zero_shot(), 3) == 0


def test_zero_shot_is_written_again_to_the_same_bytes_and_nothing_of_a_text_is_in_it(sources, tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"

    sources.write(first, zero_shot(), centres=None)
    sources.write(second, zero_shot(), centres=None)

    assert tree(first) == tree(second)
    for name in tree(first):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_a_zero_shot_folder_over_a_folder_of_centres_removes_the_file_of_the_centres_and_keeps_the_developers(
        sources, out):
    sources.write(out)  # the earlier build: embed-centroid, with data/labels.bin
    assert (out / "data" / "labels.bin").is_file()
    (out / "data" / "mine.txt").write_text("keep", encoding="utf-8")

    built = sources.write(out, zero_shot(), centres=None)

    assert built.removed == ("data/labels.bin",)
    assert not (out / "data" / "labels.bin").exists() and (out / "data" / "mine.txt").is_file()


def test_the_centres_of_a_zero_shot_folder_are_refused_and_so_is_their_absence_for_vectors(sources, out):
    with pytest.raises(bundle.BundleError, match="zero-shot has no centres"):
        sources.write(out, zero_shot(), centres=CENTRES)
    with pytest.raises(bundle.BundleError, match="needs the centres"):
        sources.write(out, centres=None)
    with pytest.raises(bundle.BundleError, match="at least two"):
        sources.write(out, zero_shot(), names=["one"], centres=None)
    assert not out.exists()


def test_an_output_that_is_a_file_is_refused(sources, tmp_path):
    target = tmp_path / "file"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(bundle.BundleError, match="is a file, not a folder"):
        sources.write(target)


# --- a build into a folder that has one ---------------------------------------------------------


def test_the_same_build_again_changes_nothing_and_does_not_copy_the_big_files_again(sources, out):
    sources.write(out)
    model = out / f"models/{REVISION}/org/tiny/onnx/model_quantized.onnx"
    before = {name: (out / name).read_bytes() for name in tree(out)}
    stamp = model.stat().st_mtime_ns

    built = sources.write(out)

    assert built.removed == ()
    assert {name: (out / name).read_bytes() for name in tree(out)} == before
    assert model.stat().st_mtime_ns == stamp


def test_another_revision_and_other_labels_replace_the_old_build_and_leave_other_files(sources, out):
    sources.write(out)
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    (out / "models").joinpath("readme.txt").write_text("mine", encoding="utf-8")
    new = "b" * 40

    built = sources.write(out, candidate(new), names=["x", "y"], centres=np.eye(2))

    old = f"models/{REVISION}/org/tiny/"
    assert set(built.removed) == {old + name for name in (*MODEL_FILES, *GRAPH)}
    assert not (out / f"models/{REVISION}").exists()
    assert (out / f"models/{new}/org/tiny/config.json").is_file()
    manifest = json.loads((out / "cifarium.json").read_text(encoding="utf-8"))
    assert [label["name"] for label in manifest["labels"]] == ["x", "y"]
    assert (out / "data/labels.bin").stat().st_size == 16
    assert manifest["model"]["revision"] == new
    assert (out / "notes.txt").read_text(encoding="utf-8") == "mine"
    assert (out / "models" / "readme.txt").is_file()


def test_a_file_the_developer_put_into_an_old_model_folder_stays_with_its_folder(sources, out):
    sources.write(out)
    mine = out / f"models/{REVISION}/org/tiny/extra.txt"
    mine.write_text("mine", encoding="utf-8")

    sources.write(out, candidate("b" * 40))

    assert mine.read_text(encoding="utf-8") == "mine"
    assert not (out / f"models/{REVISION}/org/tiny/config.json").exists()


@pytest.mark.parametrize("old", ["not json at all", '{"format": 7}', '{"format": 1, "engine": 3}', "[]"])
def test_a_manifest_that_cannot_be_read_means_nothing_is_deleted(sources, out, old):
    sources.write(out)
    (out / "cifarium.json").write_text(old, encoding="utf-8")
    other = f"models/{REVISION}/org/tiny/config.json"

    built = sources.write(out, candidate("b" * 40))

    assert built.removed == ()
    assert (out / other).is_file()
    assert json.loads((out / "cifarium.json").read_text(encoding="utf-8"))["format"] == 1


def test_a_path_that_leaves_the_folder_in_the_old_manifest_is_never_deleted(sources, out, tmp_path):
    sources.write(out)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    manifest = json.loads((out / "cifarium.json").read_text(encoding="utf-8"))
    manifest["engine"]["loader"]["path"] = "../../outside.txt"
    manifest["data"]["centres"]["path"] = outside.as_posix()
    (out / "cifarium.json").write_text(json.dumps(manifest), encoding="utf-8")

    built = sources.write(out)

    assert outside.read_text(encoding="utf-8") == "keep"
    assert "../../outside.txt" not in built.removed and outside.as_posix() not in built.removed


def test_a_failed_rebuild_leaves_the_old_manifest_and_files_in_place(sources, out):
    sources.write(out)
    manifest = (out / "cifarium.json").read_bytes()
    (sources.model / "config.json").write_bytes(b"x" * 99)

    with pytest.raises(bundle.BundleError):
        sources.write(out, candidate("b" * 40))

    assert (out / "cifarium.json").read_bytes() == manifest
    assert (out / f"models/{REVISION}/org/tiny/config.json").is_file()


# --- host: hub -------------------------------------------------------------------------------------------


def test_a_folder_with_host_hub_has_no_models_and_its_manifest_names_the_model_on_hugging_face(sources, out):
    # The model's files are not read at all: the cache of the model may be empty.
    shutil.rmtree(sources.model)
    built = sources.write(out, host="hub")

    assert not (out / "models").exists()
    assert not any(name.startswith("models/") for name in tree(out))
    assert set(built.files) == tree(out)
    manifest = json.loads((out / "cifarium.json").read_text(encoding="utf-8"))
    assert manifest == built.manifest
    assert manifest["model"] == {"repo": "org/tiny", "revision": REVISION, "license": "apache-2.0",
                                 "host": "hub", "path": None}
    # Everything but `model.path` is the manifest of the same build with the model in the folder.
    bundled = bundle.manifest_of(candidate(), [Label(name) for name in LABELS], 36, engine_sizes=ENGINE_SIZES)
    assert {**manifest, "model": None} == {**bundled, "model": None}
    assert bundled["model"] == {**manifest["model"], "host": "bundle", "path": f"models/{REVISION}/"}


def test_the_first_visit_of_a_hub_folder_weighs_the_same_and_names_the_files_on_hugging_face(sources, out):
    built = sources.write(out, host="hub")

    total, files = bundle.first_load(built.manifest, "cpu")
    assert total == built.first_load_bytes == 11 + 7 + 13 + 9 + 5 + 36
    base = f"https://huggingface.co/org/tiny/resolve/{REVISION}/"
    assert bundle.model_base(built.manifest) == base == bundle.hub_address("org/tiny", REVISION)
    assert {name for name in files if name.startswith("https://")} == {base + name for name in (*MODEL_FILES, *GRAPH)}
    # The folder owns none of them, so no rebuild ever tries to delete an address.
    assert not any(name.startswith(("models/", "https:")) for name in bundle.owned(built.manifest))
    # The size on disk is the folder without the model: what an upload sends.
    assert built.size_bytes == sum((out / name).stat().st_size for name in tree(out))


def test_a_hub_build_over_a_bundle_build_removes_the_model_and_keeps_the_developers_files(sources, out):
    sources.write(out)
    mine = out / f"models/{REVISION}/org/tiny/extra.txt"
    mine.write_text("mine", encoding="utf-8")

    built = sources.write(out, host="hub")

    assert set(built.removed) == {f"models/{REVISION}/org/tiny/{name}" for name in (*MODEL_FILES, *GRAPH)}
    assert mine.read_text(encoding="utf-8") == "mine"
    assert not (out / f"models/{REVISION}/org/tiny/config.json").exists()

    # And back: the model comes into the folder again, and nothing of the hub build is left to remove.
    again = sources.write(out)
    assert again.removed == ()
    assert (out / f"models/{REVISION}/org/tiny/onnx/model_quantized.onnx").is_file()


def test_a_manifest_without_host_is_a_bundle_manifest_for_a_rebuild(sources, out):
    # Folders built before `model.host` existed: their model is still deleted by a hub build.
    sources.write(out)
    manifest = json.loads((out / "cifarium.json").read_text(encoding="utf-8"))
    del manifest["model"]["host"]
    (out / "cifarium.json").write_text(json.dumps(manifest), encoding="utf-8")

    built = sources.write(out, host="hub")

    assert len(built.removed) == len(MODEL_FILES) + len(GRAPH)


def test_the_license_file_of_a_hub_folder_says_the_model_is_not_in_it(sources, out):
    sources.write(out, host="hub")

    text = (out / "LICENSES.txt").read_text(encoding="utf-8")
    assert "The model's files are not in this folder: the page downloads them from Hugging Face." in text
    assert "The model: not in this folder: the page downloads it from Hugging Face\n" in text
    assert f"https://huggingface.co/org/tiny at revision {REVISION}" in text
    assert "models/" not in text and "served from your own site" not in text


def test_a_hub_folder_is_the_same_bytes_in_another_folder(sources, tmp_path):
    one = sources.write(tmp_path / "one", host="hub")
    two = sources.write(tmp_path / "two", host="hub")

    assert one.files == two.files
    for name in one.files:
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes(), name


def test_an_unknown_host_is_a_mistake_of_the_caller(sources, out):
    with pytest.raises(ValueError, match="host must be one of"):
        sources.write(out, host="cdn")
    with pytest.raises(ValueError, match="host must be one of"):
        bundle.manifest_of(candidate(), [Label("a"), Label("b")], 0, host="cdn")
    assert not out.exists()
    assert bundle.HOSTS == ("bundle", "hub") and bundle.HUB_ADDRESS == "https://huggingface.co/"


# --- the report files ----------------------------------------------------------------------------------


def test_the_folder_owns_the_two_report_names_and_a_rebuild_writes_them_again(sources, out):
    first = sources.write(out)
    assert {"report.json", "report.md"} <= bundle.owned(first.manifest)
    (out / "report.md").write_text("an older report", encoding="utf-8")
    (out / "report.json").write_text("{}", encoding="utf-8")

    # Another model: its files replace the old ones, and the reports are the new build's.
    second = sources.write(out, candidate("b" * 40), reports={"report.json": b"[]\n", "report.md": b"new\n"})

    assert (out / "report.json").read_bytes() == b"[]\n" and (out / "report.md").read_bytes() == b"new\n"
    assert "report.json" in second.files and "report.md" in second.files
    assert not any(name.startswith("report") for name in second.removed)


def test_a_build_given_no_report_leaves_the_one_of_an_earlier_build_and_does_not_list_it(sources, out):
    sources.write(out)

    built = sources.write(out, reports={})

    assert (out / "report.json").read_bytes() == REPORTS["report.json"]
    assert "report.json" not in built.files and "report.md" not in built.files
    assert built.size_bytes == sum((out / name).stat().st_size for name in built.files)
    assert built.removed == ()


def test_the_report_is_written_before_the_manifest_which_stays_the_last_file(sources, out, monkeypatch):
    order: list[str] = []
    real = bundle._write_bytes

    def spy(path, data):
        order.append(path.name)
        real(path, data)

    monkeypatch.setattr(bundle, "_write_bytes", spy)

    sources.write(out)

    assert order[-1] == "cifarium.json"
    assert {"report.json", "report.md"} <= set(order[:-1])


def test_a_report_that_cannot_be_written_leaves_the_old_manifest_and_is_a_bundle_error(sources, out, monkeypatch):
    sources.write(out)
    manifest = (out / "cifarium.json").read_bytes()
    real = bundle._write_bytes

    def fail_on_report(path, data):
        if path.name == "report.md":
            raise PermissionError(13, "Permission denied")
        real(path, data)

    monkeypatch.setattr(bundle, "_write_bytes", fail_on_report)

    with pytest.raises(bundle.BundleError, match="Permission denied"):
        sources.write(out, candidate("b" * 40))

    assert (out / "cifarium.json").read_bytes() == manifest


def test_only_the_files_of_the_report_may_be_given_to_write(sources, out):
    with pytest.raises(ValueError, match="not files of the report"):
        sources.write(out, reports={"report.json": b"{}", "index.html": b"x"})
    with pytest.raises(ValueError, match="not files of the report"):
        bundle.write_reports(out, {"cifarium.json": b"{}"})
    assert not out.exists()


def test_write_reports_makes_the_folder_and_touches_nothing_else(sources, out):
    names = bundle.write_reports(out, REPORTS)

    assert sorted(names) == ["report.json", "report.md"] and tree(out) == {"report.json", "report.md"}
    assert not list(out.rglob("*.part"))
    # An earlier build next to it stays as it was, report and all.
    sources.write(out)
    before = snapshot(out)
    bundle.write_reports(out, {"report.md": b"# no model passed\n"})
    after = snapshot(out)
    assert after.pop("report.md") == b"# no model passed\n" and before.pop("report.md") == REPORTS["report.md"]
    assert after == before


def test_write_reports_refuses_a_file_where_the_folder_should_be_and_says_how_it_failed(tmp_path):
    taken = tmp_path / "cifarium-out"
    taken.write_text("a file", encoding="utf-8")

    with pytest.raises(bundle.BundleError, match="is a file, not a folder"):
        bundle.write_reports(taken, REPORTS)
    assert taken.read_text(encoding="utf-8") == "a file"

    blocked = tmp_path / "blocked.txt"
    blocked.write_text("x", encoding="utf-8")
    with pytest.raises(bundle.BundleError, match="cannot write"):
        bundle.write_reports(blocked / "inside", REPORTS)


def test_stale_is_what_the_old_build_had_and_the_new_one_has_not():
    assert bundle.stale({"a", "b", "c"}, {"b", "d"}) == ["a", "c"]
    assert bundle.stale([], {"b"}) == []


def test_a_folder_of_the_first_layout_is_brought_up_to_date_and_its_label_file_goes(sources, out):
    # Before the labels moved into the manifest, a build also wrote data/labels.json.
    sources.write(out)
    (out / "data").joinpath("labels.json").write_text('["shipping", "returns", "technical"]', encoding="utf-8")
    manifest = json.loads((out / "cifarium.json").read_text(encoding="utf-8"))
    manifest["data"]["labels"] = {"path": "data/labels.json", "bytes": 38}
    (out / "cifarium.json").write_text(json.dumps(manifest), encoding="utf-8")

    built = sources.write(out)

    assert built.removed == ("data/labels.json",)
    assert not (out / "data" / "labels.json").exists()


# --- the same input, the same bytes; and nothing in the folder is private --------------------------


def snapshot(folder: Path) -> dict[str, bytes]:
    return {name: (folder / name).read_bytes() for name in sorted(tree(folder))}


def test_the_same_input_gives_the_same_bytes_in_another_folder(sources, tmp_path):
    labels = [Label("shipping", "Where is it"), Label("returns"), Label("technical", "Something broke")]
    first, second = tmp_path / "one" / "cifarium-out", tmp_path / "two" / "elsewhere"

    sources.write(first, names=labels)
    sources.write(second, names=labels)

    assert snapshot(first) == snapshot(second)
    text = (second / "cifarium.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in text and str(tmp_path).replace("\\", "/") not in text
    assert not re.search(r"\b(?:19|20)\d\d-\d\d-\d\d", text), "a date in the manifest"
    assert not re.search(r"\b1[5-9]\d{8}\b", text), "a Unix time in the manifest"


def test_the_checker_of_the_invariant_finds_a_text_whole_escaped_cut_and_in_any_case(tmp_path):
    text = "Where is my package? The tracking page hasn't updated in four days."
    folder = tmp_path / "folder"
    for name, content in {
        "plain.txt": f"before {text} after",
        "escaped.json": json.dumps({"x": "Где моя посылка? Трек-номер не обновляется уже неделю."}),
        "cut.txt": "the tracking page hasn't updated in four days. no more",
        "upper.txt": "WHERE IS MY PACKAGE? THE TRACKING PAGE HASN'T UPDATED",
        "vendor/library.js": text,
        "clean.txt": "shipping, returns, technical",
    }.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(content, encoding="utf-8")

    leaks = example_leaks(folder, [text, "Где моя посылка? Трек-номер не обновляется уже неделю."])

    assert {leak.split(":")[0] for leak in leaks} == {"plain.txt", "escaped.json", "cut.txt", "upper.txt"}


def test_a_build_folder_holds_no_text_of_examples_and_every_file_may_be_published(sources, out):
    examples = [
        "Where is my package? The tracking page hasn't updated in four days.",
        "I'd like to return these shoes, they are too small.",
        "Где моя посылка? Трек-номер не обновляется уже неделю.",
    ]
    sources.write(out, names=[Label("shipping", "Where the parcel is"), Label("returns"), Label("technical")])

    assert example_leaks(out, examples) == []
    # The files are the fixed ones of section 7: no report, no cache, no log of the build.
    assert not [name for name in tree(out) if name.endswith((".log", ".jsonl", ".yaml", ".yml", ".npz", ".part"))]


# --- the hosting files ----------------------------------------------------------------------------

READERS = {"_headers": hosting.netlify, "vercel.json": hosting.vercel, "nginx.conf": hosting.nginx}
LONG = "public, max-age=31536000, immutable"


def isolated(path: str) -> bool:
    return path in ("/", "/index.html", "/cifarium-worker.js") or path.startswith("/vendor/")


@pytest.mark.parametrize("name", sorted(READERS))
def test_every_hosting_file_isolates_the_page_the_worker_and_the_engine_and_nothing_else(sources, out, name):
    sources.write(out)
    text = (out / "hosting" / name).read_text(encoding="utf-8")

    for path in ["/", *("/" + file for file in sorted(tree(out)))]:
        headers = READERS[name](text, path)
        has = {key: value for key, value in headers.items() if key in hosting.ISOLATION}
        assert has == (hosting.ISOLATION if isolated(path) else {}), f"{name}: {path}"


@pytest.mark.parametrize("name", sorted(READERS))
def test_every_hosting_file_keeps_only_vendor_and_models_for_a_year(sources, out, name):
    sources.write(out)
    text = (out / "hosting" / name).read_text(encoding="utf-8")

    for path in ["/", *("/" + file for file in sorted(tree(out)))]:
        cache = READERS[name](text, path).get("Cache-Control", "")
        if path.startswith(("/vendor/", "/models/")):
            assert cache == LONG, f"{name}: {path}"
        else:
            assert "immutable" not in cache and "max-age=31536000" not in cache, f"{name}: {path}"


def rule_paths(name: str) -> list:
    """The paths a hosting file puts a rule on, in the file's order."""
    text = (paths.web_dir() / "templates" / "hosting" / name).read_text(encoding="utf-8")
    if name == "vercel.json":
        return [rule["source"] for rule in json.loads(text)["headers"]]
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    if name == "_headers":
        return [line.strip() for line in lines if line.strip() and line[0] not in " \t"]
    return re.findall(r"location\s+((?:=|~)\s+)?(\S+)\s*\{", "\n".join(lines))


def test_the_paths_that_carry_headers_are_the_contract_and_a_change_is_a_decision():
    # ARCHITECTURE.md, sections 1.6 and 14: a path added or dropped here breaks the sites that
    # copied these files. Change a list only together with CHANGELOG.md.
    assert rule_paths("_headers") == ["/", "/index.html", "/cifarium-worker.js", "/vendor/*", "/models/*"]
    assert rule_paths("vercel.json") == [
        "/", "/index.html", "/cifarium-worker.js", "/vendor/(.*)", "/models/(.*)"]
    assert rule_paths("nginx.conf") == [
        ("~ ", r"^/(index\.html)?$"), ("= ", "/cifarium-worker.js"), ("", "/vendor/"), ("", "/models/"),
        ("~ ", r"^/(cifarium\.json|cifarium\.js|data/.*)$")]


def test_the_netlify_file_says_that_cloudflare_pages_reads_it_too():
    text = (paths.web_dir() / "templates" / "hosting" / "_headers").read_text(encoding="utf-8")
    assert "Netlify and Cloudflare Pages" in text


def test_vercel_json_is_json_and_has_nothing_but_headers():
    data = json.loads((paths.web_dir() / "templates" / "hosting" / "vercel.json").read_text(encoding="utf-8"))
    assert list(data) == ["headers"]


def test_nginx_knows_the_module_file_type_of_the_loader():
    text = (paths.web_dir() / "templates" / "hosting" / "nginx.conf").read_text(encoding="utf-8")
    assert re.search(r"text/javascript\s+js\s+mjs;", text) and re.search(r"application/wasm\s+wasm;", text)

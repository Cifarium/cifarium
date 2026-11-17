"""The folder of a build without a browser: layout, manifest, label data, what a rebuild keeps
and deletes, and the three hosting files. The engine and the model are small made-up files, so
the sizes of the catalog entry here are the ones of those files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import hosting_rules
import numpy as np
import pytest

import cifarium
from cifarium import bundle, catalog, paths, vendor

ENGINE_SIZES = {vendor.LIBRARY_FILE: 11, vendor.WASM_LOADER_FILE: 7, vendor.WASM_BINARY_FILE: 13}
MODEL_FILES = {"config.json": 3, "tokenizer.json": 4, "tokenizer_config.json": 2}
GRAPH = {"onnx/model_quantized.onnx": 5}
REVISION = "a" * 40
LABELS = ["shipping", "returns", "technical"]
CENTRES = np.eye(3)


def fill(folder: Path, sizes: dict[str, int]) -> None:
    for number, (name, size) in enumerate(sizes.items()):
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes([number + 1]) * size)


def entry(revision: str = REVISION, license: str = "apache-2.0") -> catalog.Entry:
    variants = {"cpu": catalog.Variant("cpu", "wasm", "q8", dict(GRAPH))}
    return catalog.Entry(Path("tiny.yaml"), "tiny", "org/tiny", revision, license, "org/source",
                         ("en",), dict(MODEL_FILES), variants, ())


def candidate(revision: str = REVISION, license: str = "apache-2.0") -> catalog.Candidate:
    recipe = catalog.Recipe("classify", "embed-centroid", {"pooling": "mean", "prefix": ""})
    return catalog.Candidate(entry(revision, license), recipe)


@dataclass
class Sources:
    engine: Path
    model: Path

    def write(self, folder: Path, who: catalog.Candidate | None = None, names=LABELS, centres=CENTRES, **more):
        return bundle.write(folder, who or candidate(), names, centres, engine_dir=self.engine,
                            model_dir=self.model, engine_sizes=ENGINE_SIZES, **more)


@pytest.fixture
def sources(tmp_path: Path) -> Sources:
    engine, model = tmp_path / "cache" / "engine", tmp_path / "cache" / "model"
    fill(engine, ENGINE_SIZES)
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
        engine + vendor.LIBRARY_FILE, engine + vendor.WASM_LOADER_FILE, engine + vendor.WASM_BINARY_FILE,
        model + "config.json", model + "tokenizer.json", model + "tokenizer_config.json",
        model + "onnx/model_quantized.onnx",
        "data/labels.json", "data/labels.bin",
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
        "model": {"repo": "org/tiny", "revision": REVISION, "license": "apache-2.0",
                  "path": f"models/{REVISION}/"},
        "variants": {"cpu": {"device": "wasm", "dtype": "q8", "graph": "model", "externalData": 0,
                             "files": {**MODEL_FILES, **GRAPH}}},
        "params": {"pooling": "mean", "prefix": ""},
        "engine": {
            "transformers": vendor.TRANSFORMERS_VERSION,
            "onnxruntime_web": vendor.ORT_WEB_VERSION,
            "library": {"path": engine + vendor.LIBRARY_FILE, "bytes": 11},
            "loader": {"path": engine + vendor.WASM_LOADER_FILE, "bytes": 7},
            "executor": {"path": engine + vendor.WASM_BINARY_FILE, "bytes": 13},
        },
        "data": {"labels": {"path": "data/labels.json", "bytes": (out / "data/labels.json").stat().st_size},
                 "centres": {"path": "data/labels.bin", "bytes": 36}},
    }
    total, files = bundle.first_load(manifest, "cpu")
    assert total == built.first_load_bytes == 11 + 7 + 13 + 9 + 5 + (out / "data/labels.json").stat().st_size + 36
    assert files <= tree(out)
    for path in files:
        assert (out / path).stat().st_size > 0


def test_the_manifest_format_is_the_one_the_runtime_reads():
    source = (paths.web_dir() / "runtime" / "cifarium.js").read_text(encoding="utf-8")
    match = re.search(r"^export const MANIFEST_FORMAT = (\d+);$", source, re.MULTILINE)
    assert match and int(match.group(1)) == bundle.FORMAT


def test_the_label_data_is_utf8_names_and_little_endian_float32_rows(sources, out):
    names = ["доставка", "возврат", "сбой"]
    centres = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, -1.0]])

    sources.write(out, names=names, centres=centres)

    raw = (out / "data/labels.json").read_bytes()
    assert json.loads(raw) == names
    assert "доставка".encode() in raw
    rows = np.frombuffer((out / "data/labels.bin").read_bytes(), dtype="<f4").reshape(3, 2)
    assert np.array_equal(rows, centres.astype("<f4"))


def test_runtime_and_page_are_the_package_files_byte_for_byte(sources, out):
    sources.write(out)

    for name in ("cifarium.js", "cifarium-worker.js"):
        assert (out / name).read_bytes() == (paths.web_dir() / "runtime" / name).read_bytes()
    assert (out / "index.html").read_bytes() == (paths.web_dir() / "templates" / "index.html").read_bytes()
    for name in ("_headers", "vercel.json", "nginx.conf"):
        assert (out / "hosting" / name).read_bytes() == (paths.web_dir() / "templates" / "hosting" / name).read_bytes()


def test_the_license_file_names_every_part_and_has_no_placeholder_left(sources, out):
    sources.write(out, candidate(license="gemma"))

    text = (out / "LICENSES.txt").read_text(encoding="utf-8")
    assert "$" not in text and "\r" not in text
    assert f"Transformers.js {vendor.TRANSFORMERS_VERSION}" in text and "Apache-2.0" in text
    assert f"ONNX Runtime Web {vendor.ORT_WEB_VERSION}" in text and "MIT" in text
    assert "License: gemma" in text
    assert "https://huggingface.co/org/source" in text
    assert f"https://huggingface.co/org/tiny at revision {REVISION}" in text
    assert f"models/{REVISION}/org/tiny/" in text


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
    other = catalog.Candidate(entry(), catalog.Recipe("classify", "zero-shot", {}))
    with pytest.raises(bundle.BundleError, match="zero-shot is not written yet"):
        sources.write(out, other)
    with pytest.raises(bundle.BundleError, match="tiny has no gpu variant"):
        sources.write(out, variants=("cpu", "gpu"))
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
    assert json.loads((out / "data/labels.json").read_text(encoding="utf-8")) == ["x", "y"]
    assert (out / "data/labels.bin").stat().st_size == 16
    assert json.loads((out / "cifarium.json").read_text(encoding="utf-8"))["model"]["revision"] == new
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
    manifest["data"]["labels"]["path"] = "../../outside.txt"
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


def test_stale_is_what_the_old_build_had_and_the_new_one_has_not():
    assert bundle.stale({"a", "b", "c"}, {"b", "d"}) == ["a", "c"]
    assert bundle.stale([], {"b"}) == []


# --- the hosting files ----------------------------------------------------------------------------

READERS = {"_headers": hosting_rules.netlify, "vercel.json": hosting_rules.vercel, "nginx.conf": hosting_rules.nginx}
LONG = "public, max-age=31536000, immutable"


def isolated(path: str) -> bool:
    return path in ("/", "/index.html", "/cifarium-worker.js") or path.startswith("/vendor/")


@pytest.mark.parametrize("name", sorted(READERS))
def test_every_hosting_file_isolates_the_page_the_worker_and_the_engine_and_nothing_else(sources, out, name):
    sources.write(out)
    text = (out / "hosting" / name).read_text(encoding="utf-8")

    for path in ["/", *("/" + file for file in sorted(tree(out)))]:
        headers = READERS[name](text, path)
        has = {key: value for key, value in headers.items() if key in hosting_rules.ISOLATION}
        assert has == (hosting_rules.ISOLATION if isolated(path) else {}), f"{name}: {path}"


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


def test_vercel_json_is_json_and_has_nothing_but_headers():
    data = json.loads((paths.web_dir() / "templates" / "hosting" / "vercel.json").read_text(encoding="utf-8"))
    assert list(data) == ["headers"]


def test_nginx_knows_the_module_file_type_of_the_loader():
    text = (paths.web_dir() / "templates" / "hosting" / "nginx.conf").read_text(encoding="utf-8")
    assert re.search(r"text/javascript\s+js\s+mjs;", text) and re.search(r"application/wasm\s+wasm;", text)

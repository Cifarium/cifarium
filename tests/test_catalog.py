"""Model catalog: the entries shipped in the package and the checks, no network."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cifarium import catalog
from cifarium.catalog import CatalogError

GOOD = """\
id: tiny-model
repo: Org/tiny-model
revision: 0123456789abcdef0123456789abcdef01234567
license: apache-2.0
license_source: org/tiny-source
languages: [en]
common_files:
  config.json: 10
  tokenizer.json: 20
variants:
  cpu:
    device: wasm
    dtype: q8
    files:
      onnx/model_quantized.onnx: 100
recipes:
  - task: classify
    kind: embed-centroid
    pooling: mean
    dimensions: 384
"""


def write(folder: Path, text: str, name: str = "tiny-model") -> Path:
    path = folder / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def problems_of(path: Path) -> list[str]:
    with pytest.raises(CatalogError) as caught:
        catalog.read_entry(path)
    return caught.value.problems


def test_the_catalog_in_the_package_reads_without_errors():
    entries = catalog.load()

    assert [entry.id for entry in entries] == [
        "all-minilm-l6-v2",
        "bge-small-en-v1.5",
        "embeddinggemma-300m",
        "mdeberta-v3-base-xnli",
        "mobilebert-uncased-mnli",
        "multilingual-e5-small",
        "nli-deberta-v3-xsmall",
        "paraphrase-multilingual-minilm-l12-v2",
    ]
    for entry in entries:
        assert re.fullmatch(r"[0-9a-f]{40}", entry.revision), entry.id
        assert set(entry.variants) == {"cpu"}, entry.id
        assert entry.variants["cpu"].dtype in ("q8", "q4"), entry.id
        assert {"config.json", "tokenizer.json", "tokenizer_config.json"} <= set(entry.common_files), entry.id
        assert all(size > 0 for size in entry.files().values()), entry.id


def test_classify_has_eight_candidates_five_by_examples_and_three_label_fit():
    found = catalog.candidates("classify")

    assert len(found) == 8
    kinds = sorted(candidate.recipe.kind for candidate in found)
    assert kinds == ["embed-centroid"] * 5 + ["zero-shot"] * 3


def test_other_tasks_have_no_candidates_yet():
    assert catalog.candidates("search") == ()
    assert catalog.candidates("summarize") == ()


def test_only_gemma_and_the_unknown_license_need_license_any():
    others = {entry.id: entry.license for entry in catalog.load() if not catalog.is_permissive(entry.license)}

    assert others == {"embeddinggemma-300m": "gemma", "mobilebert-uncased-mnli": "unknown"}


def test_e5_asks_for_its_query_prefix_and_bge_for_the_first_token():
    recipes = {c.entry.id: c.recipe.params for c in catalog.candidates("classify")}

    assert recipes["multilingual-e5-small"] == {"pooling": "mean", "prefix": "query: ", "dimensions": 384}
    assert recipes["bge-small-en-v1.5"] == {"pooling": "cls", "prefix": "", "dimensions": 384}
    assert recipes["mdeberta-v3-base-xnli"] == {}


def test_the_vector_lengths_of_the_catalog_are_those_of_the_models():
    # Read from hidden_size in the config.json of each model (the cache of a build holds them); the
    # first load of a build counts the centres of the labels from these numbers.
    lengths = {c.entry.id: c.recipe.params["dimensions"] for c in catalog.candidates("classify")
               if c.recipe.kind == "embed-centroid"}

    assert lengths == {"all-minilm-l6-v2": 384, "bge-small-en-v1.5": 384, "embeddinggemma-300m": 768,
                       "multilingual-e5-small": 384, "paraphrase-multilingual-minilm-l12-v2": 384}


def test_a_license_that_is_not_permissive_comes_with_its_notice_and_the_others_may_have_one():
    entries = {entry.id: entry for entry in catalog.load()}

    for entry in entries.values():
        assert (entry.notice is not None) or catalog.is_permissive(entry.license), entry.id
    assert "gemma.ai/resources/terms" in entries["embeddinggemma-300m"].notice
    assert "Gemma is provided under and subject to the Gemma Terms of Use" in entries["embeddinggemma-300m"].notice
    assert entries["mobilebert-uncased-mnli"].notice.startswith("The card of this model names no license")


def test_a_notice_is_required_when_the_license_is_not_permissive(tmp_path):
    problems = problems_of(write(tmp_path, GOOD.replace("license: apache-2.0", "license: gemma")))

    assert any(problem.startswith("notice: missing; the license gemma is not permissive") for problem in problems)


def test_a_notice_is_kept_as_written_and_is_optional_for_a_permissive_license(tmp_path):
    plain = catalog.read_entry(write(tmp_path, GOOD))
    noted = catalog.read_entry(write(
        tmp_path, GOOD.replace("languages:", "notice: |\n  Copyright (c) Org\n  Second line\nlanguages:"), "tiny-model"))

    assert plain.notice is None
    assert noted.notice == "Copyright (c) Org\nSecond line"


@pytest.mark.parametrize("value", ["5", "''", "[a]"])
def test_a_notice_must_be_text(tmp_path, value):
    problems = problems_of(write(tmp_path, GOOD.replace("languages:", f"notice: {value}\nlanguages:")))

    assert any(problem.startswith("notice: must be the text of the notice") for problem in problems), problems


def test_the_files_of_an_entry_are_the_common_files_and_the_variants(tmp_path):
    entry = catalog.read_entry(write(tmp_path, GOOD))

    assert entry.files() == {"config.json": 10, "tokenizer.json": 20, "onnx/model_quantized.onnx": 100}
    assert entry.recipes == (
        catalog.Recipe("classify", "embed-centroid", {"pooling": "mean", "prefix": "", "dimensions": 384}),)


@pytest.mark.parametrize(
    ("license", "permissive"),
    [("apache-2.0", True), ("mit", True), ("bsd-3-clause", True), ("gemma", False),
     ("unknown", False), ("cc-by-nc-4.0", False)],
)
def test_permissive_licenses(license, permissive):
    assert catalog.is_permissive(license) is permissive


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("revision: 0123456789abcdef0123456789abcdef01234567\n", "", "revision: missing"),
        ("0123456789abcdef0123456789abcdef01234567", "0123456", "revision: must be the full commit hash"),
        ("license: apache-2.0", "licence: apache-2.0", "licence: unknown key; did you mean license?"),
        ("id: tiny-model", "id: other-model", 'id: must match the file name tiny-model; got "other-model"'),
        ("id: tiny-model", "id: Tiny_Model", "id: must be lowercase letters, digits, dots and dashes"),
        ("repo: Org/tiny-model", "repo: tiny-model", 'repo: must be organisation/model; got "tiny-model"'),
        ("languages: [en]", "languages: [ru]", "languages: must be a list of en or multi"),
        ("config.json: 10", "config.json: -10", "common_files.config.json: must be a positive whole number of bytes"),
        ("config.json: 10", "config.json: '10'", 'common_files.config.json: must be a positive whole number of bytes; got "10"'),
        ("config.json: 10", "../config.json: 10", 'common_files: "../config.json" is not a path inside the repository'),
        ("  cpu:", "  tpu:", "variants.tpu: unknown variant; the variants are cpu, gpu"),
        ("device: wasm", "device: webgpu", 'variants.cpu.device: must be wasm; got "webgpu"'),
        ("dtype: q8", "dtype: q3", 'variants.cpu.dtype: unknown dtype "q3"'),
        ("    dtype: q8\n", "", "variants.cpu.dtype: missing"),
        ("kind: embed-centroid", "kind: embed-centroids", "did you mean embed-centroid?"),
        ("pooling: mean", "pooling: max", 'recipes, item 1: pooling: must be mean, cls or none; got "max"'),
        ("    pooling: mean\n", "", "recipes, item 1: pooling: must be mean, cls or none; got nothing"),
        ("pooling: mean", "pooling: mean\n    prefix: 5", "recipes, item 1: prefix: must be text; got 5"),
        ("    dimensions: 384\n", "", "recipes, item 1: dimensions: must be a positive whole number; got nothing"),
        ("dimensions: 384", "dimensions: 0", "dimensions: must be a positive whole number; got 0"),
        ("dimensions: 384", "dimensions: '384'", 'dimensions: must be a positive whole number; got "384"'),
        ("dimensions: 384", "dimensions: yes", "dimensions: must be a positive whole number; got true"),
        ("kind: embed-centroid\n    pooling: mean\n    dimensions: 384", "kind: zero-shot\n    hypothesis: x",
         "recipes, item 1: hypothesis: unknown parameter of zero-shot; its parameters: none"),
        ("task: classify", "task: translate", 'recipes, item 1: task: must be one of classify, search, summarize'),
        ("recipes:\n  - task: classify\n    kind: embed-centroid\n    pooling: mean\n    dimensions: 384\n",
         "recipes: []\n", "recipes: must list at least one recipe"),
        ("recipes:\n  - task: classify\n    kind: embed-centroid\n    pooling: mean\n    dimensions: 384\n",
         "", "recipes: missing"),
    ],
)
def test_each_problem_names_the_key(tmp_path, old, new, expected):
    assert old in GOOD
    found = problems_of(write(tmp_path, GOOD.replace(old, new, 1)))

    assert any(expected in problem for problem in found), found


def test_a_broken_entry_names_its_file(tmp_path):
    path = write(tmp_path, GOOD.replace("dtype: q8", "dtype: q3"))

    with pytest.raises(CatalogError) as caught:
        catalog.load(tmp_path)

    assert str(caught.value).startswith(f"{path}: variants.cpu.dtype")


def test_an_empty_catalog_folder_is_an_error(tmp_path):
    with pytest.raises(CatalogError, match="no catalog entries"):
        catalog.load(tmp_path)

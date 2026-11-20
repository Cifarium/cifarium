"""The runtime files without a browser; tests/test_runtime_e2e.py runs them in Chrome."""

from __future__ import annotations

import re

import cifarium
from cifarium import paths

RUNTIME = paths.web_dir() / "runtime"


def constant(name: str) -> str:
    source = (RUNTIME / "cifarium.js").read_text(encoding="utf-8")
    match = re.search(rf"^export const {name} = (.+);$", source, re.MULTILINE)
    assert match, f"cifarium.js has no {name}"
    return match.group(1)


def test_the_runtime_has_the_version_of_the_package():
    # The manifest's "cifarium" field and the runtime's VERSION are both this version.
    assert constant("VERSION") == f"'{cifarium.__version__}'"


def test_the_runtime_reads_manifest_format_1():
    assert constant("MANIFEST_FORMAT") == "1"


def test_the_runtime_reads_both_recipes_of_the_catalog_and_the_worker_has_a_pipeline_for_each():
    from cifarium import catalog

    source = (RUNTIME / "cifarium.js").read_text(encoding="utf-8")
    tasks = re.search(r"^const TASKS = \{ classify: \[(.*?)\] \};$", source, re.MULTILINE)
    assert tasks and re.findall(r"'([^']+)'", tasks.group(1)) == ["embed-centroid", "zero-shot"]
    assert set(re.findall(r"'([^']+)'", tasks.group(1))) == set(catalog.RECIPES)
    worker = (RUNTIME / "cifarium-worker.js").read_text(encoding="utf-8")
    pipelines = re.search(r"^const PIPELINES = \{ (.*?) \};$", worker, re.MULTILINE)
    assert pipelines and dict(re.findall(r"'([^']+)': '([^']+)'", pipelines.group(1))) == {
        "embed-centroid": "feature-extraction", "zero-shot": "zero-shot-classification"}


def test_the_runtime_starts_the_worker_that_lies_next_to_it():
    source = (RUNTIME / "cifarium.js").read_text(encoding="utf-8")
    assert "new URL('./cifarium-worker.js', import.meta.url)" in source
    assert (RUNTIME / "cifarium-worker.js").is_file()

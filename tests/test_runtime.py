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


def test_the_runtime_starts_the_worker_that_lies_next_to_it():
    source = (RUNTIME / "cifarium.js").read_text(encoding="utf-8")
    assert "new URL('./cifarium-worker.js', import.meta.url)" in source
    assert (RUNTIME / "cifarium-worker.js").is_file()

"""The cache of results in the real Chrome: two builds in a row on the English set give one table and
one folder, and an example that is added is the only vector the second build computes.

Not in ``tests/test_build_e2e.py`` on purpose: that file names what the end-to-end run downloads, so
the cache key of the job in .github/workflows/tests.yml is made of it, and a change to it makes the
runner download the engine and its license texts again. These tests download nothing the English
build does not (ARCHITECTURE.md, section 11).
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from test_build_e2e import ENGLISH, write_task

from cifarium import build, paths

REPORTS = {"report.json", "report.md"}
"""The two files that hold the times of the run: they differ between two builds, nothing else does."""


@dataclass(frozen=True)
class Again:
    code: int
    out: str
    err: str
    folder: Path
    seconds: float


def build_again(project: Path, items, results: Path, monkeypatch) -> Again:
    """A build of ``items`` in an empty project, with the cache of results of ``english_build``."""
    settings = write_task(project, items)
    monkeypatch.setattr(paths, "results_dir", lambda cache=None: results)
    out, err = io.StringIO(), io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = build.run(settings)
    return Again(code, out.getvalue(), err.getvalue(), project / "cifarium-out", time.perf_counter() - started)


def table_of(out: str) -> list[str]:
    lines = out.splitlines()
    start = lines.index("Every candidate:") + 1
    return lines[start:lines.index("", start)]


def files_of(folder: Path) -> dict[str, bytes]:
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in sorted(folder.rglob("*"))
            if path.is_file()}


def without_times(document: dict) -> dict:
    """The report with what the bench measured taken out: the one thing a second build may change."""
    document = json.loads(json.dumps(document))
    for row in document["candidates"]:
        row["speed"] = None
    document["chosen"]["oneThread"] = document["chosen"]["oneThreadError"] = None
    return document


@pytest.mark.e2e
def test_a_second_build_on_the_same_examples_takes_every_vector_from_the_cache_and_changes_nothing(
        english_build, tmp_path, monkeypatch):
    assert english_build.code == 0, english_build.err
    assert f"Vectors of the examples: {len(ENGLISH)} computed on the bench." in english_build.out
    assert list(english_build.results.glob("*.npz")), "the first build left nothing in the cache"

    second = build_again(tmp_path, ENGLISH, english_build.results, monkeypatch)

    assert second.code == 0, second.err
    assert f"Vectors of the examples: all {len(ENGLISH)} taken from the cache." in second.out
    assert f"all-minilm-l6-v2: {len(ENGLISH)} vectors from the cache" in second.out
    # The bench still ran for the times: the chosen model, once for the usual answer and once in a thread.
    assert "running" not in second.out and "all-minilm-l6-v2: measuring the speed" in second.out
    assert "one thread    " in second.out and "not measured" not in second.out
    # The same table and the same choice.
    assert table_of(second.out) == table_of(english_build.out)
    assert "Chosen: all-minilm-l6-v2" in second.out
    # The same folder, byte by byte: the vectors are the first build's, so the centres are.
    first_files, second_files = files_of(english_build.folder), files_of(second.folder)
    assert set(first_files) == set(second_files)
    assert [name for name in first_files if first_files[name] != second_files[name]
            and name not in REPORTS] == []
    first_report = json.loads(first_files["report.json"])
    second_report = json.loads(second_files["report.json"])
    assert without_times(first_report) == without_times(second_report)
    # The times are the second build's own: measured now, so a number, not a copy of the first.
    passed = next(row for row in second_report["candidates"] if row["verdict"] == "passed")
    assert passed["speed"]["medianMs"] > 0 and second_report["chosen"]["oneThread"]["medianMs"] > 0


@pytest.mark.e2e
def test_an_example_that_is_added_is_the_only_vector_the_second_build_computes(
        english_build, tmp_path, monkeypatch):
    assert english_build.code == 0, english_build.err
    extra = ("Do you have a size chart for the jackets?", "returns")

    again = build_again(tmp_path, (*ENGLISH, extra), english_build.results, monkeypatch)

    assert again.code == 0, again.err
    assert f"Vectors of the examples: {len(ENGLISH)} taken from the cache, 1 computed on the bench." in again.out
    assert f"running 1 examples on the processor, {len(ENGLISH)} more from the cache" in again.out
    assert "Chosen: all-minilm-l6-v2" in again.out

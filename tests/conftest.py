"""Fixtures shared by the tests: a cache of results of their own, and the end-to-end build."""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from test_build_e2e import ENGLISH, write_task

from cifarium import build, paths


@pytest.fixture(autouse=True)
def private_results(tmp_path, monkeypatch) -> Path:
    """Every test gets an empty cache of results (cifarium.results), never the one of the user:
    a build must not find vectors an earlier test, or an earlier run on this machine, left behind."""
    folder = tmp_path / "cifarium-results"
    monkeypatch.setattr(paths, "results_dir", lambda cache=None: folder)
    return folder


@dataclass(frozen=True)
class EnglishBuild:
    """One `cifarium build` on the English set, run once for all the tests of the session."""

    code: int
    out: str
    err: str
    folder: Path
    results: Path
    """The cache of results it filled: a second build that points `paths.results_dir` here finds the vectors."""
    seconds: float


@pytest.fixture(scope="session")
def english_build(tmp_path_factory) -> EnglishBuild:
    project = tmp_path_factory.mktemp("english")
    results = tmp_path_factory.mktemp("english-results")
    settings = write_task(project, ENGLISH)
    out, err = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(paths, "results_dir", lambda cache=None: results)
        started = time.perf_counter()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = build.run(settings)
        seconds = time.perf_counter() - started
    return EnglishBuild(code, out.getvalue(), err.getvalue(), project / "cifarium-out", results, seconds)

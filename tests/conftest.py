"""Fixtures shared by the end-to-end tests (`pytest -m e2e`)."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import pytest
from test_build_e2e import ENGLISH, write_task

from cifarium import build


@dataclass(frozen=True)
class EnglishBuild:
    """One `cifarium build` on the English set, run once for all the tests of the session."""

    code: int
    out: str
    err: str
    folder: Path


@pytest.fixture(scope="session")
def english_build(tmp_path_factory) -> EnglishBuild:
    project = tmp_path_factory.mktemp("english")
    settings = write_task(project, ENGLISH)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = build.run(settings)
    return EnglishBuild(code, out.getvalue(), err.getvalue(), project / "cifarium-out")

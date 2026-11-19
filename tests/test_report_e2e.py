"""The report of a build in the real Chrome: the English build's report against its folder, and the
report of a build that chooses nothing.

Not in ``tests/test_build_e2e.py`` on purpose: that file names what the end-to-end run downloads, so
the cache key of the job in .github/workflows/tests.yml is made of it, and a change to it makes the
runner download the engine and its license texts again. These tests download nothing the English
build does not (ARCHITECTURE.md, section 11).
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest
from publishable import example_leaks
from test_build_e2e import ENGLISH, write_task
from test_contracts import unknown_report_keys

from cifarium import build, bundle, catalog, paths

TEXTS = [text for text, _ in ENGLISH]


@pytest.mark.e2e
def test_the_two_files_of_the_report_are_in_the_published_folder_and_hold_no_text_of_the_examples(english_build):
    # `test_build_e2e.py` searches the whole folder for the texts; these are two files of it.
    assert english_build.code == 0, english_build.err

    assert (english_build.folder / "report.json").is_file() and (english_build.folder / "report.md").is_file()
    assert example_leaks(english_build.folder, TEXTS) == []


@pytest.mark.e2e
def test_the_report_of_a_real_build_has_the_keys_of_the_contract_and_the_weight_of_its_folder(english_build):
    assert english_build.code == 0, english_build.err
    folder = english_build.folder
    document = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "cifarium.json").read_text(encoding="utf-8"))

    assert unknown_report_keys(document) == []
    assert list(document)[:2] == ["format", "cifarium"] and document["cifarium"] == manifest["cifarium"]
    assert (document["outcome"], document["exitCode"], document["task"]) == ("built", 0, "classify")
    assert {"report.json", "report.md"} <= bundle.owned(manifest)
    # The weight is the folder's: the manifest, the files on disk, the table of the console.
    total, files = bundle.first_load(manifest, "cpu")
    chosen = document["chosen"]
    assert chosen["id"] == "all-minilm-l6-v2" and chosen["revision"] == manifest["model"]["revision"]
    assert chosen["firstLoad"]["bytes"] == total == sum((folder / name).stat().st_size for name in files)
    assert chosen["firstLoad"]["dataBytes"] == (folder / "data" / "labels.bin").stat().st_size
    parts = chosen["firstLoad"]
    assert parts["bytes"] == parts["engineBytes"] + parts["dataBytes"] + parts["modelBytes"]
    assert f"{total / 1e6:.1f} MB" in english_build.out
    # Every catalog candidate is there, in the order of the table, and the one that passed has its numbers.
    assert {row["id"] for row in document["candidates"]} == {c.entry.id for c in catalog.candidates("classify")}
    row = next(row for row in document["candidates"] if row["verdict"] == "passed")
    assert row["id"] == chosen["id"] and row["firstLoadBytes"] == total
    score = row["evaluation"]
    assert score["total"] == len(ENGLISH) and score["correct"] >= 0.85 * score["total"]
    assert score["margin"] == round(score["correct"] / score["total"] - 0.85, 4) >= 0
    assert len(score["misses"]) == score["total"] - score["correct"]
    labels = {label for _, label in ENGLISH}
    for miss in score["misses"]:
        assert 1 <= miss["line"] <= len(ENGLISH) and {miss["expected"], miss["predicted"]} <= labels
        assert miss["expected"] == ENGLISH[miss["line"] - 1][1], "the line is the line of the examples file"
    assert row["speed"]["medianMs"] > 0 and row["speed"]["threads"] >= 1 and row["speed"]["crossOriginIsolated"]
    assert chosen["oneThread"]["medianMs"] > 0 and chosen["oneThreadError"] is None
    device = document["device"]
    assert device["browser"]["name"] == "Chrome" and device["browser"]["version"] >= 100
    assert device["threads"] == row["speed"]["threads"] and device["crossOriginIsolated"] is True
    assert document["examples"]["count"] == len(ENGLISH)
    assert f"**Chosen: {chosen['id']}**" in (folder / "report.md").read_text(encoding="utf-8")


@pytest.mark.e2e
def test_a_real_build_that_chooses_nothing_leaves_its_report_and_no_folder(tmp_path: Path):
    # A limit of 40 MB sorts every model out before one runs: Chrome starts, the table is made, the
    # exit code is 1, and the only things in `output` are the two files of the report.
    settings = write_task(tmp_path, ENGLISH, "limits:\n  max_size_mb: 40\n")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = build.run(settings)

    assert code == 1, err.getvalue()
    assert "FAILED: no model passed" in err.getvalue() and "The report is in" in out.getvalue()
    folder = tmp_path / "cifarium-out"
    assert sorted(path.name for path in folder.iterdir()) == ["report.json", "report.md"]
    document = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert unknown_report_keys(document) == []
    assert (document["outcome"], document["exitCode"], document["chosen"]) == ("no-model", 1, None)
    verdicts = {row["verdict"] for row in document["candidates"]}
    assert verdicts <= {"license", "size"} and "size" in verdicts
    assert all(row["evaluation"] is None and row["speed"] is None for row in document["candidates"])
    assert document["device"]["browser"]["name"] == "Chrome" and document["device"]["threads"] is None
    assert "**No model passed.**" in (folder / "report.md").read_text(encoding="utf-8")
    assert example_leaks(folder, TEXTS) == []
    assert not paths.profile_dir().exists()

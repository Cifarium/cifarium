"""`cifarium build` and then `cifarium check` on the golden English task, in the real Chrome.

The folder is the one `build` writes for `examples/english` (all-MiniLM-L6-v2 on the processor). The
check serves it with the headers of each of the three hosting files, loads the model on a page through
`cifarium.js` and answers the examples; `heldout.jsonl` has the 100 texts the build never saw, which is
the held-back part of the golden task (ARCHITECTURE.md, section 3) measured on the runtime. Runs only
with `pytest -m e2e`; it downloads the same files as tests/test_build_e2e.py and lives in a file of its
own because a change of that file empties the cache of the end-to-end job on GitHub Actions (section 11).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from cifarium import build, bundle, cli, exitcodes, hosting

pytestmark = pytest.mark.e2e

GOLDEN = Path(__file__).resolve().parents[1] / "examples" / "english"
HELDOUT_SHARE = 83
"""The share of all-MiniLM on the 100 held-out texts of this build (95 on the 20 of the build, 83 on the
100): ARCHITECTURE.md, section 3, column s0. Chrome and ONNX Runtime may differ from one machine to
another in the last digits of a vector, so a text on the edge may go either way: the test allows a few."""


@dataclass(frozen=True)
class Golden:
    project: Path
    folder: Path
    out: str


@pytest.fixture(scope="module")
def golden(tmp_path_factory) -> Golden:
    project = tmp_path_factory.mktemp("golden")
    for name in ("cifarium.yaml", "examples.jsonl", "heldout.jsonl"):
        shutil.copyfile(GOLDEN / name, project / name)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = build.run(project / "cifarium.yaml")
    assert code == exitcodes.DONE, out.getvalue() + err.getvalue()
    return Golden(project, project / "cifarium-out", out.getvalue())


@pytest.fixture()
def copy(golden, tmp_path) -> Path:
    """A copy of the folder that a test may break."""
    target = tmp_path / "site"
    shutil.copytree(golden.folder, target)
    return target


def tree_hash(folder: Path) -> dict[str, str]:
    return {file.relative_to(folder).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(folder.rglob("*")) if file.is_file()}


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(["check", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def share_of(out: str) -> tuple[int, int]:
    match = re.search(r"examples\s+(\d+) of (\d+) examples right \(", out)
    assert match, out
    return int(match.group(1)), int(match.group(2))


@pytest.mark.parametrize("name", sorted(hosting.FILES))
def test_the_folder_a_build_wrote_passes_the_check_behind_each_hosting_file(golden, capsys, name):
    before = tree_hash(golden.folder)

    code, out, err = run(capsys, str(golden.folder), "--examples", str(golden.project / "examples.jsonl"),
                         "--hosting", name)

    assert code == exitcodes.DONE, out + err
    correct, total = share_of(out)
    assert (correct, total) == (20, 20), "the examples of the build are in the centres of its labels"
    assert re.search(r"ok    isolation   the page and the worker are cross-origin isolated, \d threads? on \d+ cores", out), out
    assert re.search(r"ok    first load  info\(\)\.bytes is \d+ \(\d+\.\d MB\), as report\.json says", out), out
    assert re.search(r"usual answer \d+\.\d ms \(median of 19\)", out), out
    assert "PASSED: the folder works as a site will see it." in out
    assert f"hosting/{hosting.FILES[name]}, read the way {name} reads it" in out
    # Nothing is written: the folder is published and is checked many times.
    assert tree_hash(golden.folder) == before


def test_without_arguments_the_folder_and_the_examples_are_those_of_the_settings(golden, capsys, monkeypatch):
    monkeypatch.chdir(golden.project)

    code, out, err = run(capsys)

    assert code == exitcodes.DONE, out + err
    assert "examples  20 in examples.jsonl, 4 labels (the examples of the build: it has seen them" in out
    assert "PASSED" in out


def test_on_the_texts_the_build_never_saw_the_check_prints_the_share_and_judges_it_by_the_report(golden, capsys):
    code, out, err = run(capsys, str(golden.folder), "--examples", str(golden.project / "heldout.jsonl"))

    correct, total = share_of(out)
    share = correct / total
    print(f"\nheld-out share of the English golden task on the runtime: {correct} of {total} ({share:.0%})")
    assert total == 100
    assert abs(correct - HELDOUT_SHARE) <= 3, (correct, out)
    # The pass rate is the 0.85 of the settings the report holds: 83 of 100 is under it, and the check
    # says so with the code of a folder that did not pass.
    passed = share + 1e-9 >= 0.85
    assert code == (exitcodes.DONE if passed else exitcodes.NOT_PASSED), (code, out, err)
    if not passed:
        assert "FAIL  examples" in out and "FAILED: the folder does not pass: examples." in err
    assert re.search(r"miss, line \d+: \w+ instead of \w+: ", out)
    # The other three judgements do not depend on the texts.
    assert "ok    isolation" in out and "ok    first load" in out


def test_the_pass_rate_is_the_one_of_the_report_so_a_lower_one_lets_the_same_answers_through(golden, capsys, copy):
    path = copy / "report.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["settings"]["passRate"] = 0.75
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    code, out, err = run(capsys, str(copy), "--examples", str(golden.project / "heldout.jsonl"))

    assert code == exitcodes.DONE, out + err
    assert "pass rate 75%" in out and "PASSED" in out


def test_a_report_that_gives_another_weight_than_the_folder_fails_the_first_load(golden, capsys, copy):
    path = copy / "report.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["chosen"]["firstLoad"]["bytes"] += 1
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    code, out, err = run(capsys, str(copy), "--examples", str(golden.project / "examples.jsonl"))

    assert code == exitcodes.NOT_PASSED, out + err
    assert "FAIL  first load" in out and "not of the same build" in out
    assert "FAILED: the folder does not pass: first load." in err


def test_a_hosting_file_that_sets_no_headers_fails_the_isolation_and_the_model_still_answers(golden, capsys, copy):
    (copy / "hosting" / "_headers").write_text("# no rules at all\n", encoding="utf-8")

    code, out, err = run(capsys, str(copy), "--examples", str(golden.project / "examples.jsonl"))

    assert code == exitcodes.NOT_PASSED, out + err
    assert "FAIL  isolation" in out and "not cross-origin isolated" in out
    assert "Cross-Origin-Opener-Policy" in out and "/cifarium-worker.js" in out
    assert share_of(out) == (20, 20), "one thread: slower, and still right"
    assert "FAILED: the folder does not pass: isolation." in err


def test_headers_on_the_page_alone_stop_the_model_and_the_check_says_the_page_could_not_load_it(golden, capsys, copy):
    # ARCHITECTURE.md, section 1.6: an isolated page refuses the thread's script that has no
    # Cross-Origin-Embedder-Policy of its own.
    (copy / "hosting" / "_headers").write_text(
        "/\n  Cross-Origin-Opener-Policy: same-origin\n  Cross-Origin-Embedder-Policy: require-corp\n"
        "/index.html\n  Cross-Origin-Opener-Policy: same-origin\n  Cross-Origin-Embedder-Policy: require-corp\n",
        encoding="utf-8")

    code, out, err = run(capsys, str(copy), "--examples", str(golden.project / "examples.jsonl"))

    assert code == exitcodes.NOT_PASSED, out + err
    assert "FAILED: the page could not load the model" in err
    assert "A site with this folder would fail the same way." in err
    assert "PASSED" not in out


def test_a_folder_without_the_runtime_is_a_failed_page_not_a_crash_of_the_command(golden, capsys, copy):
    (copy / "cifarium.js").unlink()

    code, out, err = run(capsys, str(copy), "--examples", str(golden.project / "examples.jsonl"))

    assert code == exitcodes.NOT_PASSED, out + err
    assert "FAILED: the page could not load cifarium.js as a module" in err


def test_the_manifest_of_a_newer_format_is_refused_before_a_browser_starts(golden, capsys, copy):
    path = copy / bundle.MANIFEST
    document = json.loads(path.read_text(encoding="utf-8"))
    document["format"] = bundle.FORMAT + 1
    path.write_text(json.dumps(document), encoding="utf-8")

    code, out, err = run(capsys, str(copy), "--examples", str(golden.project / "examples.jsonl"))

    assert code == exitcodes.BAD_INPUT and "newer" in err
    assert "browser" not in out

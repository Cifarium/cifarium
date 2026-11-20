"""The recipe zero-shot in the real Chrome on nli-deberta-v3-xsmall (87 MB at q8).

The end-to-end run is not part of the default run: `pytest -m e2e`. This file names a model the
English build does not download, so it is part of the cache key of the end-to-end job in
.github/workflows/tests.yml (ARCHITECTURE.md, section 11): without it the runner would download the
model on every run and never save it.

The model goes through :func:`build.choose` alone. In a whole `cifarium build` on the English set
all-MiniLM-L6-v2 passes first and nobody behind it runs, and the settings have no key that would
sort it out and keep this one. The tests of the folder (the end of the file) do run a whole
`build.run`, but on a catalog that holds this model only.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from publishable import example_leaks
from test_build_e2e import ENGLISH, write_task
from test_page_e2e import use_page

from cifarium import build, bundle, catalog, check, config, examples, exitcodes, hosting, paths, vendor
from cifarium import report as reporting
from cifarium.bench import BenchError, decode_scores, hypotheses, open_bench, zero_shot_job
from cifarium.server import SiteServer

pytestmark = pytest.mark.e2e

MODEL = "nli-deberta-v3-xsmall"


@pytest.fixture(scope="module")
def bench(built):
    """One Chrome for the whole module. It starts after the build of ``built`` has closed its own:
    the cache holds one Chrome profile, and two browsers at once would wipe each other's."""
    vendor.ensure_engine()
    with open_bench() as running:
        yield running


def candidate() -> catalog.Candidate:
    return next(c for c in catalog.candidates("classify") if c.entry.id == MODEL)


@dataclass(frozen=True)
class Built:
    """One `cifarium build` on the English set whose catalog holds this model only."""

    code: int
    out: str
    err: str
    project: Path
    folder: Path


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Built:
    project = tmp_path_factory.mktemp("zero-shot")
    results = tmp_path_factory.mktemp("zero-shot-results")
    settings = write_task(project, ENGLISH)
    only = candidate()
    out, err = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(paths, "results_dir", lambda cache=None: results)
        patch.setattr(catalog, "candidates", lambda task: (only,))
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = build.run(settings)
    return Built(code, out.getvalue(), err.getvalue(), project, project / "cifarium-out")


@pytest.mark.e2e
def test_the_english_set_gets_a_share_a_verdict_and_the_times_of_a_text_with_all_its_labels(bench, tmp_path, capsys):
    settings = config.load(write_task(tmp_path, ENGLISH))
    items = examples.load(settings)
    lines: list[str] = []
    embed = build.ChromeEmbed(bench, lines.append)

    result = build.choose(settings, items, [candidate()], embed, embed.speed)

    [row] = result.rows
    score = row.outcome.evaluation
    with capsys.disabled():
        print(f"\n{MODEL} on the English set: {score.correct} of {score.total}, {row.verdict}")
    assert row.verdict in ("passed", "accuracy"), row.message
    assert score.total == len(ENGLISH)
    # Three labels: a third is chance. The share itself is the report's, not the test's.
    assert score.correct > len(ENGLISH) / 2, row.message
    assert f"  {MODEL}: running {len(ENGLISH)} examples on the processor, 3 labels each" in lines
    # The times are of a text with all of its labels, from the same run, with the fields of the vectors.
    speed = embed.speed(candidate())
    assert speed.calls == len(ENGLISH) - 1 and speed.median_ms > 0 and speed.load_ms > 0
    assert speed.threads >= 1 and isinstance(speed.isolated, bool)
    assert result.chosen is None or result.chosen.speed == speed

    # The report takes the candidate as it takes any other.
    manifest = bundle.manifest_of(candidate(), items.labels, 0) if result.chosen else None
    document = reporting.of_build(settings, items, result, device=reporting.device_of(bench.device, bench.headless),
                                  manifest=manifest)
    [entry] = document["candidates"]
    assert (entry["id"], entry["recipe"], entry["verdict"]) == (MODEL, "zero-shot", row.verdict)
    assert (entry["evaluation"]["correct"], entry["evaluation"]["total"]) == (score.correct, score.total)


@pytest.mark.e2e
def test_one_thread_is_one_thread_for_the_zero_shot_job(bench):
    labels = [config.Label(name) for name in ("shipping", "returns", "technical")]
    build.download(candidate().entry, lambda message: None)

    speed = build.ChromeEmbed(bench, lambda message: None).single_thread(
        candidate(), [text for text, _ in ENGLISH], labels)

    assert speed.threads == 1 and speed.calls == build.SINGLE_THREAD_INPUTS - 1


@pytest.mark.e2e
def test_the_page_scores_every_hypothesis_as_written_and_in_the_order_of_the_job(bench):
    entry, recipe = candidate().entry, candidate().recipe
    build.download(entry, lambda message: None)
    # "$&" and "$1" mean something to String.replace, which the library uses to put a label into
    # its template; the page refuses a sentence that would come out changed, so an answer is proof.
    sentences = ["This example is about a price of $& and $1.", "This example is about the weather {}.",
                 "This example is about the weather {}."]

    scores = decode_scores(bench.run(zero_shot_job(entry, recipe, ["It costs $5.", "Rain all day."], sentences)))

    assert scores.shape == (2, 3)
    assert scores.sum(axis=1) == pytest.approx([1.0, 1.0])
    # The same sentence twice gets the same score, and the order is the job's, not the score's.
    assert scores[:, 1] == pytest.approx(scores[:, 2])
    assert scores[0, 0] > scores[0, 1] and scores[1, 1] > scores[1, 0]


@pytest.mark.e2e
def test_the_page_refuses_a_zero_shot_job_it_cannot_score(bench):
    job = zero_shot_job(candidate().entry, candidate().recipe, ["a text"], hypotheses(
        [config.Label("one"), config.Label("two")]))

    with pytest.raises(BenchError, match="at least two hypotheses"):
        bench.run({**job, "hypotheses": ["only one"]})
    with pytest.raises(BenchError, match="Unknown operation"):
        bench.run({**job, "operation": "summarize"})


# --- the folder of the recipe: manifest, runtime, report and check -------------------------------------

NAMES = ["shipping", "returns", "technical"]


def stand_scores(bench, texts, labels) -> np.ndarray:
    """What the bench page says for ``texts``: a row of scores for each, in the order of ``labels``."""
    build.download(candidate().entry, lambda message: None)
    job = zero_shot_job(candidate().entry, candidate().recipe, texts, hypotheses(labels))
    return decode_scores(bench.run(job, timeout=build.TIME_LIMIT_S))


def answers_of_the_runtime(bench, folder: Path, texts) -> dict:
    """What `cifarium.js` of ``folder`` answers on the page of `check`, behind the headers of Netlify."""
    page = {"/": check.PAGE, "/index.html": check.PAGE}
    with SiteServer(folder, hosting.headers_of(folder, "netlify"), replace=page) as server:
        outcome = check.drive(bench, server.url, texts)
    assert "result" in outcome, outcome
    return outcome["result"]


def test_a_build_that_chooses_the_model_writes_the_folder_and_the_report_and_ends_with_code_0(built):
    assert built.code == exitcodes.DONE, built.out + built.err
    manifest = json.loads((built.folder / bundle.MANIFEST).read_text(encoding="utf-8"))
    # No centres: the hypotheses of the labels are in `params`, and `data` is empty.
    assert (manifest["recipe"], manifest["data"]) == ("zero-shot", {})
    assert manifest["params"] == {"hypotheses": hypotheses([config.Label(name) for name in NAMES])}
    assert not (built.folder / "data").exists()
    document = json.loads((built.folder / bundle.REPORT_JSON).read_text(encoding="utf-8"))
    chosen = document["chosen"]
    assert (document["outcome"], chosen["id"], chosen["recipe"]) == ("built", MODEL, "zero-shot")
    assert chosen["firstLoad"]["dataBytes"] == 0
    assert chosen["firstLoad"]["bytes"] == bundle.first_load(manifest, "cpu")[0]
    assert "Wrote " in built.out and "FAILED" not in built.err
    # Everything in the folder may be published: no text of an example is in it.
    assert example_leaks(built.folder, [text for text, _ in ENGLISH]) == []


def test_the_runtime_answers_with_the_labels_and_the_scores_of_the_stand(bench, built):
    texts = [text for text, _ in ENGLISH]
    stand = stand_scores(bench, texts, [config.Label(name) for name in NAMES])

    result = answers_of_the_runtime(bench, built.folder, texts)

    answers = result["answers"]
    # The same call as the stand's, so the same label for every text and the same score, which is a
    # share of 1: the best of three is more than a third.
    assert [answer["label"] for answer in answers] == [NAMES[row] for row in stand.argmax(axis=1)]
    assert [answer["score"] for answer in answers] == pytest.approx(stand.max(axis=1), abs=1e-4)
    assert all(1 / 3 - 1e-6 <= answer["score"] <= 1 for answer in answers)
    info = result["info"]
    assert (info["recipe"], info["variant"], info["device"], info["state"]) == ("zero-shot", "cpu", "wasm", "ready")
    assert info["bytes"] == bundle.first_load(json.loads((built.folder / bundle.MANIFEST).read_text("utf-8")), "cpu")[0]


def test_a_dollar_sign_in_a_label_reaches_the_model_the_way_it_reached_the_stand(bench, built, tmp_path):
    # `String.replace`, which the library puts a label in a sentence with, would read "$&" and "$1";
    # the stand doubles every "$" and the runtime must do the same, or it asks another question.
    labels = [config.Label("price $& and $1"), config.Label("weather {}")]
    texts = ["It costs five dollars.", "Rain all day long."]
    folder = tmp_path / "dollars"
    bundle.write(folder, candidate(), labels)
    stand = stand_scores(bench, texts, labels)

    result = answers_of_the_runtime(bench, folder, texts)

    manifest = json.loads((folder / bundle.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["params"]["hypotheses"] == ["This example is about price $& and $1.",
                                                "This example is about weather {}."]
    names = [label.name for label in labels]
    assert [answer["label"] for answer in result["answers"]] == [names[row] for row in stand.argmax(axis=1)]
    assert [answer["score"] for answer in result["answers"]] == pytest.approx(stand.max(axis=1), abs=1e-4)


@pytest.mark.parametrize("change, words", [
    (lambda manifest: manifest["params"].pop("hypotheses"), "params: hypotheses, one sentence for each label"),
    (lambda manifest: manifest["params"]["hypotheses"].pop(), "params: hypotheses, one sentence for each label"),
    (lambda manifest: manifest["params"]["hypotheses"].__setitem__(0, ""), "params: hypotheses"),
    (lambda manifest: manifest.__setitem__("data", None), "data: an object"),
])
def test_a_manifest_of_zero_shot_without_a_sentence_for_every_label_is_refused_before_any_download(
        bench, built, tmp_path, change, words):
    folder = tmp_path / "broken"
    folder.mkdir()
    for name in ("cifarium.js", "cifarium-worker.js"):
        (folder / name).write_bytes((built.folder / name).read_bytes())
    manifest = json.loads((built.folder / bundle.MANIFEST).read_text(encoding="utf-8"))
    change(manifest)
    (folder / bundle.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    page = {"/": check.PAGE, "/index.html": check.PAGE}

    with SiteServer(folder, lambda path: {}, replace=page) as server:
        outcome = check.drive(bench, server.url, ["a text"])

    failure = outcome["failure"]
    assert (failure["stage"], failure["code"]) == ("create", "manifest")
    assert words in failure["message"]


def test_a_model_whose_labels_have_no_entailment_is_a_failed_load_and_not_a_silent_logit_2(bench, built, tmp_path):
    # The library would take logit 2 of a model that names no "entailment" and tell only the
    # console; the runtime refuses. The copy of the folder has the label of the model renamed.
    folder = tmp_path / "renamed"
    shutil.copytree(built.folder, folder)
    [config_file] = folder.rglob("nli-deberta-v3-xsmall/config.json")
    text = config_file.read_text(encoding="utf-8")
    assert text.lower().count("entailment") == 2  # in id2label and in label2id
    config_file.write_text(re.sub("(?i)entailment", "agreement", text), encoding="utf-8")
    page = {"/": check.PAGE, "/index.html": check.PAGE}

    with SiteServer(folder, hosting.headers_of(folder, "netlify"), replace=page) as server:
        outcome = check.drive(bench, server.url, ["a text"])

    failure = outcome["failure"]
    assert (failure["stage"], failure["code"]) == ("load", "load")
    assert 'no "entailment" label (it has' in failure["message"], failure["message"]


def test_a_model_that_is_not_a_classifier_at_all_is_a_failed_load_too(bench, built, tmp_path):
    # all-MiniLM, put into a zero-shot manifest by hand, has no labels in its config: here the
    # library itself fails to build the pipeline, and the runtime reports it as a model that did not start.
    embedder = next(c for c in catalog.candidates("classify") if c.entry.id == "all-minilm-l6-v2")
    build.download(embedder.entry, lambda message: None)
    liar = catalog.Candidate(embedder.entry, catalog.Recipe("classify", "zero-shot", {}))
    folder = tmp_path / "liar"
    bundle.write(folder, liar, [config.Label("shipping"), config.Label("returns")])
    page = {"/": check.PAGE, "/index.html": check.PAGE}

    with SiteServer(folder, hosting.headers_of(folder, "netlify"), replace=page) as server:
        outcome = check.drive(bench, server.url, ["a text"])

    failure = outcome["failure"]
    assert (failure["stage"], failure["code"]) == ("load", "load")
    assert failure["message"].startswith("The model did not start: cpu (wasm, q8)"), failure["message"]


def test_the_example_page_of_the_folder_names_the_label_and_says_what_the_score_means(bench, built):
    text = "Where is my package? The tracking page has not updated in four days."
    expected = NAMES[int(stand_scores(bench, [text], [config.Label(name) for name in NAMES]).argmax())]

    with SiteServer(built.folder, hosting.headers_of(built.folder, "netlify")) as server:
        shown = use_page(bench, server.url + "/", text)

    assert shown["label"] == expected == "shipping"
    assert re.fullmatch(rf"{expected} \([01]\.\d\d\)", shown["shown"]), shown["shown"]
    listed = re.findall(r"(\w+) ([01]\.\d\d)", shown["scores"])
    values = [float(value) for _, value in listed]
    assert sorted(name for name, _ in listed) == sorted(NAMES) and values == sorted(values, reverse=True)
    assert sum(values) == pytest.approx(1.0, abs=0.02)  # the scores add up to 1, each rounded to two digits
    info = shown["info"]
    assert "labels        shipping, returns, technical" in info
    assert "score         how well a sentence about each label fits the text, from 0 to 1, all labels together 1" in info
    assert shown["error_hidden"]


def test_check_passes_the_folder_and_its_share_on_the_examples_of_the_build_is_the_builds(
        bench, built, monkeypatch, capsys):
    monkeypatch.chdir(built.project)  # no arguments: the folder and the examples come from cifarium.yaml

    @contextlib.contextmanager
    def open_this_one(*, headed: bool = False):
        # Playwright's sync API starts one browser per thread, and the module already has one.
        yield bench

    monkeypatch.setattr(check, "open_bench", open_this_one)

    code = check.run()

    out, err = capsys.readouterr()
    assert code == exitcodes.DONE, out + err
    document = json.loads((built.folder / bundle.REPORT_JSON).read_text(encoding="utf-8"))
    [row] = document["candidates"]
    build_share = (row["evaluation"]["correct"], row["evaluation"]["total"])
    match = re.search(r"examples\s+(\d+) of (\d+) examples right \(", out)
    assert match, out
    # The model learned nothing from the examples, so the page must get what the build got.
    assert (int(match.group(1)), int(match.group(2))) == build_share
    assert "the model has not learned from them" in out and "it has seen them" not in out
    for line in ("ok    isolation", "ok    first load", "PASSED: the folder works as a site will see it."):
        assert line in out, line

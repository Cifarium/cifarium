"""The recipe zero-shot in the real Chrome on nli-deberta-v3-xsmall (87 MB at q8).

The end-to-end run is not part of the default run: `pytest -m e2e`. This file names a model the
English build does not download, so it is part of the cache key of the end-to-end job in
.github/workflows/tests.yml (ARCHITECTURE.md, section 11): without it the runner would download the
model on every run and never save it.

The model goes through :func:`build.choose` alone. In a whole `cifarium build` on the English set
all-MiniLM-L6-v2 passes first and nobody behind it runs, and the settings have no key that would
sort it out and keep this one.
"""

from __future__ import annotations

import pytest
from test_build_e2e import ENGLISH, write_task

from cifarium import build, bundle, catalog, config, examples, vendor
from cifarium import report as reporting
from cifarium.bench import BenchError, decode_scores, hypotheses, open_bench, zero_shot_job

MODEL = "nli-deberta-v3-xsmall"


@pytest.fixture(scope="module")
def bench():
    vendor.ensure_engine()
    with open_bench() as running:
        yield running


def candidate() -> catalog.Candidate:
    return next(c for c in catalog.candidates("classify") if c.entry.id == MODEL)


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

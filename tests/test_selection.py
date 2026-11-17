"""Selection of the lightest model: order, filters, the stop, and what the bench may do wrong.

The bench is a fake that returns ready vectors, so nothing here starts a browser. The
catalog entries are made up, so the numbers do not move when `catalog-update` refreshes sizes.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cifarium import bundle, catalog, selection, vendor
from cifarium.bench import BenchError
from cifarium.config import Config, Label, Limits
from cifarium.examples import Example, Examples
from cifarium.selection import Verdict, select

MB = 1_000_000
ENGINE = MB
ENGINE_SIZES = {vendor.LIBRARY_FILE: ENGINE - 2, vendor.WASM_LOADER_FILE: 1, vendor.WASM_BINARY_FILE: 1}
"""The engine of the tests weighs one megabyte, so a model of 9 MB makes a first load of 10."""

LABELS = ("alpha", "beta")
EXPECTED = ["alpha", "beta"] * 10
DIMENSIONS = len(LABELS)
"""The fake vectors have one axis for each label; the centres of two labels weigh 16 bytes."""
CENTRES = 4 * len(LABELS) * DIMENSIONS


# --- what the tests are made of --------------------------------------------------------------


def make(entry_id: str, model_bytes: int = 10 * MB, *, license: str = "apache-2.0",
         languages: tuple[str, ...] = ("multi",), kind: str = "embed-centroid",
         gpu_bytes: int | None = None) -> catalog.Candidate:
    """A candidate whose first load is exactly ``model_bytes``, without the engine, and the
    centres of the labels that a build of it writes: they are inside the number."""
    data = CENTRES if kind == "embed-centroid" else 0
    variants = {"cpu": catalog.Variant("cpu", "wasm", "q8", {"onnx/model_quantized.onnx": model_bytes - 1 - data})}
    if gpu_bytes is not None:
        variants["gpu"] = catalog.Variant("gpu", "webgpu", "q4f16", {"onnx/model_q4f16.onnx": gpu_bytes - 1})
    entry = catalog.Entry(Path(f"{entry_id}.yaml"), entry_id, f"org/{entry_id}", "0" * 40, license,
                          "org/source", languages, {"config.json": 1}, variants, ())
    params = {"pooling": "mean", "prefix": "", "dimensions": DIMENSIONS} if kind == "embed-centroid" else {}
    return catalog.Candidate(entry, catalog.Recipe("classify", kind, params))


def examples_of(texts: list[str] | None = None, expected: list[str] | None = None,
                labels: tuple[str, ...] = LABELS) -> Examples:
    expected = EXPECTED if expected is None else expected
    texts = [f"text {k}" for k in range(len(expected))] if texts is None else texts
    items = tuple(Example(k + 1, text, label) for k, (text, label) in enumerate(zip(texts, expected)))
    return Examples(Path("examples.jsonl"), items, tuple(Label(name) for name in labels))


def settings(**changes) -> Config:
    config = Config(Path("cifarium.yaml"), "classify", Path("examples.jsonl"),
                    limits=Limits(max_size_mb=300), license="permissive")
    return replace(config, **changes)


def vectors(flipped: int = 0, expected: list[str] | None = None) -> np.ndarray:
    """One axis per label, so the centres separate the labels. The first ``flipped`` examples
    of alpha get the vector of beta: with 20 examples 3 of them cost 3 misses (17 of 20 is
    85 %) and 4 cost 4 (80 %)."""
    expected = EXPECTED if expected is None else expected
    rows = np.zeros((len(expected), len(LABELS)))
    seen = 0
    for k, label in enumerate(expected):
        axis = LABELS.index(label)
        if label == "alpha":
            seen += 1
            if seen <= flipped:
                axis = 1
        rows[k, axis] = 1.0
    return rows


class FakeBench:
    """Answers per catalog id (a vector table, or an exception to raise) and a log of the calls."""

    def __init__(self, answers: dict[str, object]) -> None:
        self.answers = answers
        self.calls: list[str] = []
        self.texts: list[list[str]] = []

    def __call__(self, candidate: catalog.Candidate, texts) -> np.ndarray:
        self.calls.append(candidate.entry.id)
        self.texts.append(list(texts))
        answer = self.answers[candidate.entry.id]
        if isinstance(answer, BaseException):
            raise answer
        # A model of the real catalog returns as many numbers as the catalog says: pad the two
        # axes of the fake vectors with zeros, which keeps their length and their angles.
        promised = candidate.recipe.params.get("dimensions")
        if isinstance(answer, np.ndarray) and promised and self.pad and answer.shape[1] < promised:
            answer = np.pad(answer, ((0, 0), (0, promised - answer.shape[1])))
        return answer

    pad = True


def run(candidates, answers, *, config: Config | None = None, examples: Examples | None = None, **options):
    bench = FakeBench(answers)
    result = select(config or settings(), examples or examples_of(), candidates, bench,
                    engine_sizes=ENGINE_SIZES, **options)
    return result, bench


def verdicts(result: selection.Selection) -> dict[str, Verdict]:
    return {outcome.candidate.entry.id: outcome.verdict for outcome in result.outcomes}


# --- order and the stop ------------------------------------------------------------------------


def test_the_lightest_candidate_runs_first_and_the_walk_stops_at_the_first_that_passes():
    candidates = [make("heavy", 200 * MB), make("light", 10 * MB), make("middle", 20 * MB)]
    answers = {"light": vectors(flipped=5), "middle": vectors(), "heavy": vectors()}

    result, bench = run(candidates, answers)

    assert bench.calls == ["light", "middle"]
    assert [outcome.candidate.entry.id for outcome in result.outcomes] == ["light", "middle", "heavy"]
    assert verdicts(result) == {"light": Verdict.ACCURACY, "middle": Verdict.PASSED, "heavy": Verdict.NOT_TRIED}
    assert result.chosen.candidate.entry.id == "middle"
    assert result.chosen.message == "got 20 of 20 examples (100%)"
    assert result.outcomes[2].message == "a lighter model already passed"


def test_the_bench_gets_the_texts_of_the_examples_and_the_chosen_vectors_come_back():
    texts = [f"example {k}" for k in range(20)]
    answer = vectors()

    result, bench = run([make("only")], {"only": answer}, examples=examples_of(texts=texts))

    assert bench.texts == [texts]
    assert result.vectors is answer or np.array_equal(result.vectors, answer)


def test_the_weight_is_the_model_and_the_engine_and_is_told_in_every_outcome():
    result, _ = run([make("a", 12 * MB), make("b", 400 * MB)], {"a": vectors()})

    assert [outcome.size_bytes for outcome in result.outcomes] == [13 * MB, 401 * MB]


def test_the_weight_of_a_model_with_two_variants_is_that_of_the_processors_variant():
    # The device is not known before the build, and the processor's variant is the one every
    # visitor can run, so the graphics card's variant does not count, heavy or light.
    candidates = [make("both", 10 * MB, gpu_bytes=50 * MB), make("plain", 30 * MB)]

    result, bench = run(candidates, {"plain": vectors(), "both": vectors()})

    assert [outcome.candidate.entry.id for outcome in result.outcomes] == ["both", "plain"]
    assert result.outcomes[0].size_bytes == 11 * MB


def test_the_weight_is_the_one_definition_of_the_build_and_counts_the_centres_of_the_labels():
    # selection.first_load_bytes is bundle.first_load of the manifest of a build: the engine, the
    # centres (four bytes for every number of every label), the files of the model and nothing else.
    candidate = make("m", 10 * MB)
    for names in (("a", "b"), ("a", "b", "c"), tuple("abcdefg")):
        labels = [Label(name) for name in names]
        model = sum(candidate.entry.files().values())
        centres = 4 * len(names) * DIMENSIONS

        assert selection.first_load_bytes(candidate, labels, ENGINE_SIZES) == ENGINE + model + centres

        manifest = bundle.manifest_of(candidate, labels, centres, engine_sizes=ENGINE_SIZES)
        assert bundle.first_load(manifest, "cpu")[0] == ENGINE + model + centres


def test_a_vector_of_another_length_than_the_catalog_says_does_not_pass():
    # The weight was computed from `dimensions`; a model that returns another length would sit
    # under the limit with a wrong weight, so it is refused with the numbers named.
    bench = FakeBench({"liar": vectors(), "honest": vectors()})
    bench.pad = False
    liar = make("liar", 10 * MB)
    wrong = catalog.Candidate(liar.entry, catalog.Recipe("classify", "embed-centroid",
                                                         {**liar.recipe.params, "dimensions": 384}))

    result = select(settings(), examples_of(), [wrong, make("honest", 20 * MB)], bench,
                    engine_sizes=ENGINE_SIZES)

    assert verdicts(result) == {"liar": Verdict.VECTORS, "honest": Verdict.PASSED}
    assert "the catalog says 384 numbers to a vector, the bench returned 2" in result.outcomes[0].message


def test_equal_weights_keep_the_order_of_the_catalog_and_the_recipes_of_one_model():
    first, second = make("aaa", 10 * MB), make("bbb", 10 * MB)
    ready, later = make("ccc", 10 * MB), make("ccc", 10 * MB, kind="zero-shot")

    result, bench = run([first, second, ready, later], {name: vectors(flipped=5) for name in "aaa bbb ccc".split()})

    assert [(o.candidate.entry.id, o.candidate.recipe.kind) for o in result.outcomes] == [
        ("aaa", "embed-centroid"), ("bbb", "embed-centroid"),
        ("ccc", "embed-centroid"), ("ccc", "zero-shot"),
    ]
    assert bench.calls == ["aaa", "bbb", "ccc"]


def test_the_default_engine_weight_is_that_of_the_pinned_files():
    model = 30 * MB - vendor.engine_bytes()  # the first load is exactly 30 MB

    result = select(settings(limits=Limits(max_size_mb=30)), examples_of(), [make("fits", model)],
                    FakeBench({"fits": vectors()}))

    assert result.chosen is not None and result.chosen.size_bytes == 30 * MB


def test_no_candidates_choose_nobody():
    result, bench = run([], {})

    assert result.outcomes == () and result.chosen is None and result.vectors is None
    assert bench.calls == []


def test_the_report_hears_every_outcome_in_the_order_of_the_walk():
    heard = []
    candidates = [make("b", 20 * MB), make("a", 10 * MB, license="gemma"), make("c", 30 * MB)]

    result, _ = run(candidates, {"b": vectors(), "c": vectors()}, report=heard.append)

    assert heard == list(result.outcomes)
    assert [outcome.verdict for outcome in heard] == [Verdict.LICENSE, Verdict.PASSED, Verdict.NOT_TRIED]


# --- nobody passes -----------------------------------------------------------------------------


def test_when_nobody_passes_every_candidate_says_how_far_it_got():
    candidates = [make("one", 10 * MB), make("two", 20 * MB)]

    result, bench = run(candidates, {"one": vectors(flipped=4), "two": vectors(flipped=5)})

    assert result.chosen is None and result.vectors is None
    assert bench.calls == ["one", "two"]
    one, two = result.outcomes
    assert one.message == "got 16 of 20 examples (80%), below the pass rate of 85%"
    assert one.evaluation.correct == 16 and len(one.evaluation.misses) == 4
    assert two.evaluation.correct == 15
    assert all(outcome.tried for outcome in result.outcomes)


def test_seventeen_of_twenty_pass_eighty_five_percent():
    result, _ = run([make("edge")], {"edge": vectors(flipped=3)})

    assert result.chosen is not None
    assert result.chosen.evaluation.correct == 17


def test_the_rule_of_passing_lives_in_one_function(monkeypatch):
    candidates = [make("good", 10 * MB), make("bad", 20 * MB)]

    monkeypatch.setattr(selection, "passes", lambda evaluation: False)
    refused, _ = run(candidates, {"good": vectors(), "bad": vectors(flipped=5)})
    monkeypatch.setattr(selection, "passes", lambda evaluation: evaluation.correct < 20)
    reversed_, _ = run(candidates, {"good": vectors(), "bad": vectors(flipped=5)})

    assert refused.chosen is None
    assert verdicts(refused) == {"good": Verdict.ACCURACY, "bad": Verdict.ACCURACY}
    assert reversed_.chosen.candidate.entry.id == "bad"


# --- what the settings sort out ----------------------------------------------------------------


def test_license_permissive_sorts_out_a_model_that_is_not_permissive_and_license_any_does_not():
    gemma = make("gemma", 10 * MB, license="gemma")

    permissive, bench = run([gemma], {"gemma": vectors()})
    anything, _ = run([gemma], {"gemma": vectors()}, config=settings(license="any"))

    assert bench.calls == []
    assert permissive.outcomes[0].verdict is Verdict.LICENSE
    assert permissive.outcomes[0].message == "license gemma does not pass `license: permissive`"
    assert not permissive.outcomes[0].tried
    assert anything.chosen is not None


@pytest.mark.parametrize("license", ["apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause"])
def test_permissive_licenses_pass_license_permissive(license):
    result, _ = run([make("m", license=license)], {"m": vectors()})

    assert result.chosen is not None


def test_a_first_load_exactly_at_the_limit_is_allowed_and_one_byte_over_is_not():
    config = settings(limits=Limits(max_size_mb=10))
    at_limit, over = make("at", 9 * MB), make("over", 9 * MB + 1)

    result, bench = run([at_limit, over], {"at": vectors(flipped=5), "over": vectors()}, config=config)

    assert bench.calls == ["at"]
    assert verdicts(result) == {"at": Verdict.ACCURACY, "over": Verdict.SIZE}
    assert result.outcomes[1].message == (
        "first load is 10.0 MB (9.0 MB model and 1.0 MB engine), over `limits.max_size_mb: 10`")


def test_a_heavy_model_is_sorted_out_but_a_lighter_one_behind_it_in_the_catalog_is_still_run():
    result, bench = run([make("big", 500 * MB), make("small", 20 * MB)], {"small": vectors()})

    assert bench.calls == ["small"]
    assert verdicts(result) == {"big": Verdict.SIZE, "small": Verdict.PASSED}


def test_the_recipe_that_is_not_ready_never_reaches_the_bench():
    result, bench = run([make("nli", 10 * MB, kind="zero-shot"), make("embed", 20 * MB)], {"embed": vectors()})

    assert bench.calls == ["embed"]
    assert result.outcomes[0].verdict is Verdict.RECIPE
    assert result.outcomes[0].message == "the recipe zero-shot is not ready yet"


def test_the_first_reason_found_is_told_license_before_size_before_language_before_recipe():
    russian = examples_of(texts=["привет мир"] * 20)
    everything_wrong = make("x", 500 * MB, license="unknown", languages=("en",), kind="zero-shot")
    config = settings()

    assert run([everything_wrong], {}, config=config, examples=russian)[0].outcomes[0].verdict is Verdict.LICENSE
    config = settings(license="any")
    assert run([everything_wrong], {}, config=config, examples=russian)[0].outcomes[0].verdict is Verdict.SIZE
    config = settings(license="any", limits=Limits(max_size_mb=1000))
    assert run([everything_wrong], {}, config=config, examples=russian)[0].outcomes[0].verdict is Verdict.LANGUAGE
    assert run([everything_wrong], {}, config=config)[0].outcomes[0].verdict is Verdict.RECIPE


# --- language -----------------------------------------------------------------------------------


def test_a_model_for_english_is_sorted_out_when_the_examples_are_not_in_latin_script():
    russian = examples_of(texts=["Где моя посылка, трек не обновляется"] * 20)
    candidates = [make("english", 10 * MB, languages=("en",)), make("multi", 20 * MB)]

    result, bench = run(candidates, {"multi": vectors()}, examples=russian)

    assert bench.calls == ["multi"]
    assert result.outcomes[0].verdict is Verdict.LANGUAGE
    assert result.outcomes[0].message == "made for English, and 100% of the letters in the examples are not Latin"
    assert result.chosen.candidate.entry.id == "multi"


def test_a_model_for_english_runs_on_english_examples():
    result, bench = run([make("english", 10 * MB, languages=("en",))], {"english": vectors()})

    assert bench.calls == ["english"] and result.chosen is not None


@pytest.mark.parametrize(
    ("texts", "share"),
    [
        ([], 0.0),
        (["", "123 456 !?"], 0.0),  # no letters at all: nobody is sorted out
        (["Where is my parcel"], 0.0),
        (["café, łódź, señor, Straße, ÅÄÖ"], 0.0),  # accented Latin letters are Latin
        (["привет"], 1.0),
        (["abc привет"], 6 / 9),  # three Latin letters, six Cyrillic
        (["abc", "привет"], 6 / 9),  # texts are counted together, not averaged
        (["你好 世界"], 1.0),
        (["مرحبا"], 1.0),
    ],
)
def test_the_share_of_letters_that_are_not_latin(texts, share):
    assert selection.non_latin_share(texts) == pytest.approx(share)


def test_the_threshold_is_half_of_the_letters_and_only_more_than_half_sorts_a_model_out():
    english = make("english", 10 * MB, languages=("en",))
    half = examples_of(texts=["ab ыю"] * 20)  # two Latin letters, two Cyrillic
    mostly = examples_of(texts=["ab ыюя"] * 20)  # two Latin, three Cyrillic

    assert run([english], {"english": vectors()}, examples=half)[0].outcomes[0].verdict is Verdict.PASSED
    assert run([english], {}, examples=mostly)[0].outcomes[0].verdict is Verdict.LANGUAGE


def test_a_model_for_every_language_is_never_sorted_out_by_the_script():
    chinese = examples_of(texts=["你好世界"] * 20)

    result, _ = run([make("multi", languages=("multi",))], {"multi": vectors()}, examples=chinese)

    assert result.chosen is not None


# --- what the bench may do wrong ---------------------------------------------------------------


def test_a_bench_error_on_one_candidate_is_its_reason_and_the_next_one_still_runs():
    candidates = [make("broken", 10 * MB), make("fine", 20 * MB)]
    failure = BenchError("The job failed in the browser: out of memory\nconsole error: details")

    result, bench = run(candidates, {"broken": failure, "fine": vectors()})

    assert bench.calls == ["broken", "fine"]
    broken = result.outcomes[0]
    assert broken.verdict is Verdict.BENCH
    assert broken.message == "the bench failed: The job failed in the browser: out of memory"
    assert broken.error == str(failure) and broken.evaluation is None and broken.tried
    assert result.chosen.candidate.entry.id == "fine"


def test_a_bench_error_on_every_candidate_leaves_nobody_chosen_without_raising():
    result, _ = run([make("a"), make("b", 20 * MB)], {"a": BenchError("x"), "b": BenchError("y")})

    assert result.chosen is None
    assert [outcome.verdict for outcome in result.outcomes] == [Verdict.BENCH, Verdict.BENCH]


def test_an_exception_that_is_not_a_bench_error_is_a_defect_and_goes_up():
    with pytest.raises(RuntimeError, match="bug"):
        run([make("a")], {"a": RuntimeError("bug")})


@pytest.mark.parametrize(
    ("bad", "said"),
    [
        (np.full((20, 2), np.nan), "NaN or infinity"),
        (np.full((20, 2), 2.0), "length"),
        (vectors()[:-1], "one vector per example"),
    ],
    ids=["nan", "not-unit-length", "wrong-count"],
)
def test_vectors_that_cannot_be_scored_fail_the_candidate_and_the_walk_goes_on(bad, said):
    result, bench = run([make("odd", 10 * MB), make("fine", 20 * MB)], {"odd": bad, "fine": vectors()})

    assert bench.calls == ["odd", "fine"]
    odd = result.outcomes[0]
    assert odd.verdict is Verdict.VECTORS
    assert odd.message.startswith("the bench returned vectors that cannot be scored: ")
    assert said in odd.error
    assert result.chosen.candidate.entry.id == "fine"


def test_examples_that_do_not_fit_their_labels_stop_everything_before_the_bench_runs():
    stray = examples_of(expected=["alpha", "gamma"] * 10)

    with pytest.raises(ValueError, match="gamma"):
        run([make("a")], {}, examples=stray)

    one_label = examples_of(expected=["alpha"] * 20, labels=("alpha",))
    with pytest.raises(ValueError, match="two labels"):
        run([make("a")], {}, examples=one_label)


# --- the real catalog --------------------------------------------------------------------------


def test_on_the_real_catalog_russian_examples_reach_only_the_multilingual_models_with_a_ready_recipe():
    candidates = catalog.candidates("classify")
    russian = examples_of(texts=["Где моя посылка, трек не обновляется"] * 20)
    bench = FakeBench({entry_id: vectors(flipped=5) for entry_id in {c.entry.id for c in candidates}})

    result = select(settings(license="any", limits=Limits(max_size_mb=10_000)), russian, candidates, bench)

    reached = {(c.entry.id, c.recipe.kind) for c in candidates if "multi" in c.entry.languages
               and c.recipe.kind == "embed-centroid"}
    assert {(o.candidate.entry.id, o.candidate.recipe.kind) for o in result.outcomes if o.tried} == reached
    assert [o.size_bytes for o in result.outcomes] == sorted(o.size_bytes for o in result.outcomes)
    assert result.chosen is None


def test_on_the_real_catalog_permissive_english_work_starts_from_the_lightest_model():
    candidates = catalog.candidates("classify")
    bench = FakeBench({entry_id: vectors() for entry_id in {c.entry.id for c in candidates}})

    result = select(settings(), examples_of(), candidates, bench)

    assert bench.calls == ["all-minilm-l6-v2"]
    assert result.chosen.candidate.entry.id == "all-minilm-l6-v2"
    by_verdict = {outcome.candidate.entry.id: outcome.verdict for outcome in result.outcomes}
    assert by_verdict["embeddinggemma-300m"] is Verdict.LICENSE  # gemma is not a permissive license
    assert by_verdict["multilingual-e5-small"] is Verdict.NOT_TRIED

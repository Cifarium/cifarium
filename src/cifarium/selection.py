"""Selection: the lightest catalog model that gets the examples right.

Plain functions, no browser. The bench comes in from outside as a function that
turns a candidate and the example texts into vectors, so the tests give it ready
numbers and the ``build`` command gives it Chrome. ARCHITECTURE.md, section 9
(``build``, steps 2-5), has the choices behind it.

The candidates are put in order of the weight a visitor downloads on the first
visit, the lightest first, and walked once. Each one is sorted out by the
settings (license, size limit, a model made for English, a recipe that is not
ready) or run on the bench and scored. The walk stops at the first candidate that
passes; the ones behind it are not run. Every candidate ends up with an
:class:`Outcome` that says why it was not taken, so a report can show the whole table.
"""

from __future__ import annotations

import enum
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from . import catalog, evaluate, vendor
from .bench import BenchError
from .config import Config
from .examples import Examples

BYTES_PER_MB = 1_000_000
"""The megabyte of ``limits.max_size_mb`` and of the reports: a million bytes."""

# Recipes the build can evaluate. A candidate with another recipe is sorted out before the
# walk, so the bench never gets it (``bench.embed_job`` refuses such a recipe).
READY_RECIPES = frozenset({"embed-centroid"})

# A model made for English is sorted out when more than this share of the letters in the
# examples is not Latin (ARCHITECTURE.md, section 12, question 8).
NON_LATIN_SHARE = 0.5

Bench = Callable[[catalog.Candidate, Sequence[str]], np.ndarray]
"""Candidate and example texts -> one vector of unit length per text, as the bench returns
them. Raises :class:`~cifarium.bench.BenchError` when the candidate cannot run."""


class Verdict(enum.Enum):
    """What became of one candidate. The value is a stable word for machine-readable reports."""

    PASSED = "passed"
    LICENSE = "license"
    SIZE = "size"
    LANGUAGE = "language"
    RECIPE = "recipe"
    ACCURACY = "accuracy"
    VECTORS = "vectors"
    BENCH = "bench"
    NOT_TRIED = "not-tried"
    """A lighter candidate had already passed."""


@dataclass(frozen=True)
class Outcome:
    candidate: catalog.Candidate
    verdict: Verdict
    message: str
    """One sentence for a person, saying why the candidate was not taken (or that it was)."""
    size_bytes: int
    """What a visitor downloads on the first visit: the model and the engine."""
    evaluation: evaluate.Evaluation | None = None
    """Set when the candidate was scored: its share, and the examples it missed."""
    error: str | None = None
    """Set when the bench failed on the candidate: what it said."""

    @property
    def tried(self) -> bool:
        """Whether the candidate ran on the bench."""
        return self.verdict in (Verdict.PASSED, Verdict.ACCURACY, Verdict.VECTORS, Verdict.BENCH)


@dataclass(frozen=True)
class Selection:
    outcomes: tuple[Outcome, ...]
    """One per candidate, in the order of the walk: the lightest first."""
    vectors: np.ndarray | None
    """Vectors of the examples from the chosen candidate, so the centres of the labels can be
    computed without running the bench again; None when nothing was chosen."""

    @property
    def chosen(self) -> Outcome | None:
        return next((outcome for outcome in self.outcomes if outcome.verdict is Verdict.PASSED), None)


def select(
    config: Config,
    examples: Examples,
    candidates: Sequence[catalog.Candidate],
    bench: Bench,
    *,
    engine_bytes: int | None = None,
    report: Callable[[Outcome], object] | None = None,
) -> Selection:
    """Walk the candidates from the lightest and stop at the first one that passes.

    ``report`` is called with every outcome as soon as it is known, so a command can show
    progress while the bench is busy. Only :class:`~cifarium.bench.BenchError` of the bench is
    taken as the candidate's failure; any other exception is a defect and goes up. Raises
    :class:`ValueError` before anything runs if the examples do not fit their labels.
    """
    engine = vendor.engine_bytes() if engine_bytes is None else engine_bytes
    texts = [item.input for item in examples.items]
    expected = [item.expected for item in examples.items]
    names = [label.name for label in examples.labels]
    _check_examples(expected, names, config.pass_rate)
    share = non_latin_share(texts)

    outcomes: list[Outcome] = []
    vectors: np.ndarray | None = None
    for candidate in ordered(candidates, engine):
        size = first_load_bytes(candidate, engine)
        sorted_out = skip(candidate, config, size, engine, share)
        if sorted_out is not None:
            outcome = Outcome(candidate, *sorted_out, size)
        elif vectors is not None:
            outcome = Outcome(candidate, Verdict.NOT_TRIED, "a lighter model already passed", size)
        else:
            outcome, vectors = _run(candidate, bench, texts, expected, names, config.pass_rate, size)
        outcomes.append(outcome)
        if report is not None:
            report(outcome)
    return Selection(tuple(outcomes), vectors)


def passes(evaluation: evaluate.Evaluation) -> bool:
    """The one rule for "this candidate got the examples right".

    Today it is the share of the settings, as ``Evaluation.passed`` says. A check on a golden
    task may make it stricter, for instance a margin over ``pass_rate``; the walk asks only here.
    """
    return evaluation.passed


def first_load_bytes(candidate: catalog.Candidate, engine_bytes: int) -> int:
    """Weight of the first visit: the engine and the heaviest variant of the model."""
    entry = candidate.entry
    return engine_bytes + max(entry.first_load_bytes(name) for name in entry.variants)


def ordered(candidates: Iterable[catalog.Candidate], engine_bytes: int) -> list[catalog.Candidate]:
    """The lightest first. Equal weights keep the order they came in, which is the order of the
    catalog: by id, and for one model by its recipes."""
    return sorted(candidates, key=lambda candidate: first_load_bytes(candidate, engine_bytes))


def skip(
    candidate: catalog.Candidate, config: Config, size: int, engine_bytes: int, non_latin: float
) -> tuple[Verdict, str] | None:
    """Why the settings sort the candidate out before it runs, or None. The first reason found
    is the one told: license, size, language, then a recipe that is not ready."""
    entry = candidate.entry
    if config.license == "commercial" and not catalog.is_commercial(entry.license):
        return Verdict.LICENSE, f"license {entry.license} does not pass `license: commercial`"
    limit = config.limits.max_size_mb
    if size > limit * BYTES_PER_MB:
        return Verdict.SIZE, (
            f"first load is {size / BYTES_PER_MB:.1f} MB ({(size - engine_bytes) / BYTES_PER_MB:.1f} MB "
            f"model and {engine_bytes / BYTES_PER_MB:.1f} MB engine), over `limits.max_size_mb: {limit:g}`"
        )
    if "multi" not in entry.languages and non_latin > NON_LATIN_SHARE:
        return Verdict.LANGUAGE, (
            f"made for English, and {_percent(non_latin)} of the letters in the examples are not Latin"
        )
    if candidate.recipe.kind not in READY_RECIPES:
        return Verdict.RECIPE, f"the recipe {candidate.recipe.kind} is not ready yet"
    return None


def non_latin_share(texts: Iterable[str]) -> float:
    """Share of the letters in ``texts`` that are not Latin, from 0 to 1; 0 when there are no
    letters. Accented Latin letters (é, ł, ñ) are Latin. The rule cannot tell English from other
    Latin-script languages: the score on the examples settles those."""
    letters = other = 0
    for text in texts:
        for char in text:
            if char.isalpha():
                letters += 1
                if not char.isascii() and not unicodedata.name(char, "").startswith("LATIN"):
                    other += 1
    return other / letters if letters else 0.0


def _check_examples(expected: Sequence[str], names: Sequence[str], pass_rate: float) -> None:
    """Fail at once, and not once per candidate, when the labels do not fit the examples.

    A dry run of the scoring on dummy vectors of unit length: it reaches every check of
    :mod:`evaluate` that does not depend on the vectors.
    """
    evaluate.leave_one_out(np.ones((len(expected), 1)), expected, names, pass_rate)


def _run(
    candidate: catalog.Candidate,
    bench: Bench,
    texts: Sequence[str],
    expected: Sequence[str],
    names: Sequence[str],
    pass_rate: float,
    size: int,
) -> tuple[Outcome, np.ndarray | None]:
    """Run one candidate on the bench and score it. The vectors come back only when it passes."""
    try:
        vectors = bench(candidate, texts)
    except BenchError as error:
        return Outcome(candidate, Verdict.BENCH, f"the bench failed: {_first_line(error)}", size,
                       error=str(error)), None
    try:
        evaluation = evaluate.leave_one_out(vectors, expected, names, pass_rate)
    except ValueError as error:
        # The labels were checked before the walk, so what is left is the vectors: a count that
        # does not match, NaN, a length other than 1. They belong to this candidate.
        return Outcome(candidate, Verdict.VECTORS,
                       f"the bench returned vectors that cannot be scored: {error}", size,
                       error=str(error)), None
    if not passes(evaluation):
        return Outcome(
            candidate, Verdict.ACCURACY,
            f"got {evaluation.correct} of {evaluation.total} examples ({_percent(evaluation.accuracy)}), "
            f"below the pass rate of {_percent(evaluation.pass_rate)}",
            size, evaluation=evaluation,
        ), None
    return Outcome(
        candidate, Verdict.PASSED,
        f"got {evaluation.correct} of {evaluation.total} examples ({_percent(evaluation.accuracy)})",
        size, evaluation=evaluation,
    ), np.asarray(vectors)


def _percent(share: float) -> str:
    return f"{share * 100:.1f}".removesuffix(".0") + "%"


def _first_line(error: BaseException) -> str:
    lines = str(error).strip().splitlines()
    return lines[0] if lines else type(error).__name__

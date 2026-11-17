"""The ``build`` command: the lightest model that gets the examples right and answers in time.

Reads the settings and the examples, downloads the engine, starts the bench and walks the
catalog with :func:`selection.select`, the lightest candidate first. The candidate that
passes on accuracy is then checked on speed: the usual answer from the same run on the bench
(the median of the calls after the first) against ``limits.max_latency_ms``. A candidate that
is too slow does not end the build: the walk goes on with the candidates behind it. The
chosen model is run once more in a single thread, on a few examples, to show what a site
without the isolation headers gets. ARCHITECTURE.md, section 9 (``build``), has the choices.

The result is printed and the folder for the site is written to ``output`` by :mod:`bundle`.
"""

from __future__ import annotations

import math
import os
import re
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import bundle, catalog, config, evaluate, examples, hub, selection, vendor
from .bench import Bench, BenchError, decode_vectors, embed_job, open_bench
from .selection import BYTES_PER_MB, Outcome, Verdict

TIME_LIMIT_S = 300.0
"""Time limit of one job on the bench: loading the model and every example."""

SINGLE_THREAD_INPUTS = 6
"""Examples of the single-thread run: the first call warms up, the median is of the other five.
Few on purpose: no vectors are needed, and a large model in one thread takes seconds a call."""

LATENCY = "latency"
"""Verdict word of a candidate that passed on accuracy and was too slow. It lives here and not
in :class:`selection.Verdict`, because the speed check is not part of the selection."""

SpeedOf = Callable[[catalog.Candidate], "Speed"]
"""The times of a candidate's last run on the bench."""


@dataclass(frozen=True)
class Speed:
    """Times of one run on the bench, in milliseconds."""

    load_ms: float
    """From creating the pipeline to ready, with an empty browser store."""
    first_ms: float
    """The first call, warm-up included."""
    median_ms: float
    """The usual answer: the median of the calls after the first. The speed check uses it."""
    p95_ms: float
    """95th percentile of the calls after the first, by nearest rank."""
    calls: int
    """Calls after the first, the ones the median is of."""
    threads: int
    isolated: bool
    """Whether the page was crossOriginIsolated."""

    @classmethod
    def of(cls, result: dict[str, Any]) -> Speed:
        """Read the times out of the page's answer (``EmbedResult`` in bench.js)."""
        times = [float(value) for value in result["timesMs"]]
        if not times:
            raise ValueError("the bench returned no times")
        usual = times[1:] or times
        ranked = sorted(usual)
        return cls(
            load_ms=float(result["loadMs"]),
            first_ms=times[0],
            median_ms=float(statistics.median(usual)),
            p95_ms=ranked[math.ceil(0.95 * len(ranked)) - 1],
            calls=len(usual),
            threads=int(result["threads"]),
            isolated=bool(result["crossOriginIsolated"]),
        )


@dataclass(frozen=True)
class Row:
    """One candidate of the build: what the selection said, and the speed of one that passed."""

    outcome: Outcome
    speed: Speed | None = None
    """Set for a candidate that passed on accuracy."""
    limit_ms: float | None = None
    """``limits.max_latency_ms`` of the settings; None when there is no limit."""

    @property
    def fast_enough(self) -> bool:
        return self.speed is not None and (self.limit_ms is None or self.speed.median_ms <= self.limit_ms)

    @property
    def chosen(self) -> bool:
        return self.outcome.verdict is Verdict.PASSED and self.fast_enough

    @property
    def verdict(self) -> str:
        """A stable word: one of :class:`selection.Verdict`, or :data:`LATENCY`."""
        if self.outcome.verdict is Verdict.PASSED and not self.fast_enough:
            return LATENCY
        return self.outcome.verdict.value

    @property
    def message(self) -> str:
        if self.verdict == LATENCY:
            assert self.speed is not None and self.limit_ms is not None
            return (f"{self.outcome.message}, but the usual answer takes {_ms(self.speed.median_ms)}, "
                    f"over `limits.max_latency_ms: {self.limit_ms:g}`")
        return self.outcome.message


@dataclass(frozen=True)
class Result:
    rows: tuple[Row, ...]
    """One per candidate, in the order of the walk: the lightest first."""
    vectors: np.ndarray | None
    """Vectors of the examples from the chosen candidate; None when nothing was chosen."""

    @property
    def chosen(self) -> Row | None:
        return next((row for row in self.rows if row.chosen), None)


def choose(
    settings: config.Config,
    items: examples.Examples,
    candidates: Sequence[catalog.Candidate],
    embed: selection.Embed,
    speed_of: SpeedOf,
    *,
    engine_bytes: int | None = None,
    report: Callable[[Row], object] | None = None,
) -> Result:
    """The lightest candidate that passes on accuracy and on speed. No browser here: the bench
    comes in as ``embed``, and ``speed_of`` gives the times of a candidate it has run.

    A candidate that passes on accuracy but is too slow gets the verdict ``latency``, and the
    selection runs again on the candidates after it in the walk: the lighter ones have already
    failed. ``report`` gets every candidate that ran on the bench as soon as its verdict is
    known; the ones the settings sorted out are only in the result.
    """
    engine = vendor.engine_bytes() if engine_bytes is None else engine_bytes
    limit = settings.limits.max_latency_ms

    def row_of(outcome: Outcome) -> Row:
        if outcome.verdict is Verdict.PASSED:
            return Row(outcome, speed_of(outcome.candidate), limit)
        return Row(outcome)

    def on_outcome(outcome: Outcome) -> None:
        if report is not None and outcome.tried:
            report(row_of(outcome))

    walk = selection.ordered(candidates, engine)
    rows: list[Row] = []
    while True:
        picked = selection.select(settings, items, walk, embed, engine_bytes=engine, report=on_outcome)
        # The selection keeps the order it was given: ``walk`` is already the order of the walk.
        passed = next((k for k, outcome in enumerate(picked.outcomes) if outcome.verdict is Verdict.PASSED), None)
        if passed is None:
            rows.extend(Row(outcome) for outcome in picked.outcomes)
            return Result(tuple(rows), None)
        rows.extend(Row(outcome) for outcome in picked.outcomes[:passed])
        row = row_of(picked.outcomes[passed])
        rows.append(row)
        if row.fast_enough:
            rows.extend(Row(outcome) for outcome in picked.outcomes[passed + 1:])
            return Result(tuple(rows), picked.vectors)
        # By position, not by weight: two candidates may weigh the same.
        walk = walk[passed + 1:]


class ChromeEmbed:
    """The selection's bench function on a running bench.

    Downloads the files of a candidate when they are not in the cache yet, runs the job of its
    recipe in a new tab and keeps the times, which :meth:`speed` gives back.
    """

    def __init__(self, bench: Bench, say: Callable[[str], None]) -> None:
        self.bench = bench
        self.say = say
        self._speeds: dict[tuple[str, str], Speed] = {}

    def __call__(self, candidate: catalog.Candidate, texts: Sequence[str]) -> np.ndarray:
        entry = candidate.entry
        download(entry, self.say)
        self.say(f"  {entry.id}: running {len(texts)} examples on the processor")
        result = self.bench.run(embed_job(entry, candidate.recipe, texts), timeout=TIME_LIMIT_S)
        self._speeds[_key(candidate)] = Speed.of(result)
        return decode_vectors(result)

    def speed(self, candidate: catalog.Candidate) -> Speed:
        return self._speeds[_key(candidate)]

    def single_thread(self, candidate: catalog.Candidate, texts: Sequence[str]) -> Speed:
        """Times of the candidate in one thread, on the first :data:`SINGLE_THREAD_INPUTS` texts."""
        job = embed_job(candidate.entry, candidate.recipe, texts[:SINGLE_THREAD_INPUTS], threads=1)
        speed = Speed.of(self.bench.run(job, timeout=TIME_LIMIT_S))
        if speed.threads != 1:
            # The page reports what the engine used; anything else means the job's field was lost.
            raise BenchError(f"asked the page for 1 thread, it computed in {speed.threads}")
        return speed


def download(entry: catalog.Entry, say: Callable[[str], None]) -> None:
    """Put the files of the entry's processor variant into the cache, saying so if any is missing."""
    files = entry.files()
    folder = hub.model_dir(entry.repo)
    missing = sum(size for name, size in files.items() if not (folder / name).is_file())
    if missing:
        say(f"  {entry.id}: downloading {_mb(missing)} from Hugging Face")
    hub.download_files(entry.repo, entry.revision, files)


def run(path: str | os.PathLike[str] = config.DEFAULT_FILE, *, headed: bool = False) -> int:
    """Run the command and return the process exit code: 0 when a model was chosen and the
    folder written, 1 otherwise."""
    _tolerate_any_text()
    try:
        settings = config.load(path)
        items = examples.load(settings)
        candidates = catalog.candidates(settings.task)
    except config.TaskFileError as error:
        print(error, file=sys.stderr)
        return 1
    if settings.host != "bundle":
        # Before the bench: a long run must not end in a folder the settings did not ask for.
        print(f"{settings.path}: host: {settings.host} is not supported yet: the folder always carries "
              "the model. Use `host: bundle` or leave the key out.", file=sys.stderr)
        return 1

    texts = [item.input for item in items.items]
    limit = settings.limits.max_latency_ms
    _say("cifarium build")
    _say(f"  settings  {settings.path}")
    _say(f"  examples  {len(items.items)} in {settings.examples.name}, {len(items.labels)} labels, "
         f"pass rate {_percent(settings.pass_rate)}")
    _say(f"  limits    first load up to {settings.limits.max_size_mb:g} MB, usual answer "
         + ("without a limit" if limit is None else f"up to {limit:g} ms")
         + f", license {settings.license}")
    try:
        vendor.ensure_engine(report=lambda message: _say(f"  {message}"))
        _say(f"  engine    Transformers.js {vendor.TRANSFORMERS_VERSION}, "
             f"onnxruntime-web {vendor.ORT_WEB_VERSION}")
        with open_bench(headed=headed) as bench:
            _say(f"  browser   {describe_browser(bench)}")
            _say("\nCandidates on the bench, the lightest first:")
            embed = ChromeEmbed(bench, _say)
            result = choose(settings, items, candidates, embed, embed.speed, report=_report)
            chosen = result.chosen
            single: Speed | str | None = None
            if chosen is not None:
                _say(f"  {chosen.outcome.candidate.entry.id}: measuring one thread "
                     f"on {min(len(texts), SINGLE_THREAD_INPUTS)} examples")
                try:
                    single = embed.single_thread(chosen.outcome.candidate, texts)
                except BenchError as error:
                    single = _first_line(error)
    except (vendor.VendorError, hub.HubError, BenchError) as error:
        print(f"\nFAILED: {error}", file=sys.stderr)
        return 1

    _say("\nEvery candidate:")
    for line in describe_table(result.rows):
        _say(f"  {line}")
    if chosen is None:
        _say("\nNo model was chosen: the table says why each one was not taken.")
        print("\nFAILED: no model passed. More examples per label, a lower `pass_rate`, wider "
              "`limits` or `license: any` in the settings may let one through.", file=sys.stderr)
        return 1
    _say("")
    for line in describe_chosen(chosen, single, items.items):
        _say(line)

    assert result.vectors is not None
    names = [label.name for label in items.labels]
    centres = evaluate.centroids(result.vectors, [item.expected for item in items.items], names)
    try:
        built = bundle.write(settings.output, chosen.outcome.candidate, items.labels, centres)
    except bundle.BundleError as error:
        print(f"\nFAILED: {error}", file=sys.stderr)
        return 1
    _say("")
    for line in describe_built(built):
        _say(line)
    return 0


def describe_built(built: bundle.Built) -> list[str]:
    """What was written, and how to look at it."""
    where = _shown(built.folder)
    lines = [
        f"Wrote {where}: {len(built.files)} files, {_mb(built.size_bytes)} on disk; "
        f"a first visit downloads {_mb(built.first_load_bytes)}",
    ]
    if built.removed:
        lines.append(f"  removed {len(built.removed)} files of the previous build that this one does not use")
    lines += [
        f"  {bundle.PAGE} is an example page and every build writes it again: copy it before you change it",
        f"  hosting/ has the headers for Netlify, Vercel and nginx; {bundle.LICENSES} the licenses",
        "",
        "To look at it, serve the folder over http (a page opened from file:// cannot start the model):",
        f"  python -m http.server 8000 --directory {_quoted(where)}",
        "and open http://localhost:8000/ . That server sends no isolation headers, so the model runs in "
        "one thread there.",
    ]
    return lines


def describe_table(rows: Sequence[Row]) -> list[str]:
    """The table of reasons: one line per candidate, in the order of the walk."""
    width = max((len(row.outcome.candidate.entry.id) for row in rows), default=5)
    lines = [f"{'model':<{width}}  {'recipe':<14}  {'first load':>10}  {'verdict':<9}  reason"]
    for row in rows:
        candidate = row.outcome.candidate
        lines.append(f"{candidate.entry.id:<{width}}  {candidate.recipe.kind:<14}  "
                     f"{_mb(row.outcome.size_bytes):>10}  {row.verdict:<9}  {row.message}")
    return lines


def describe_chosen(row: Row, single: Speed | str | None, items: Sequence[examples.Example]) -> list[str]:
    """What the build chose: share, weight, times, threads, the speed check and the misses."""
    outcome, speed = row.outcome, row.speed
    assert speed is not None and outcome.evaluation is not None
    entry, score = outcome.candidate.entry, outcome.evaluation
    variant = entry.variants["cpu"]
    engine = vendor.engine_bytes()
    isolation = "crossOriginIsolated" if speed.isolated else "not crossOriginIsolated"
    lines = [
        f"Chosen: {entry.id} ({entry.repo}, {variant.dtype} on the processor, license {entry.license})",
        f"  accuracy      {score.correct} of {score.total} examples ({_percent(score.accuracy)}), "
        f"pass rate {_percent(score.pass_rate)}",
        f"  first load    {_mb(outcome.size_bytes)}: {_mb(outcome.size_bytes - engine)} model "
        f"and {_mb(engine)} engine",
        f"  model load    {speed.load_ms / 1000:.2f} s",
        f"  per example   {_ms(speed.median_ms)} median, 95th percentile {_ms(speed.p95_ms)}, "
        f"first call {_ms(speed.first_ms)}",
        f"  threads       {speed.threads}, {isolation}",
    ]
    if isinstance(single, Speed):
        lines.append(f"  one thread    {_ms(single.median_ms)} median of {single.calls} examples, "
                     "as on a site without the isolation headers")
    elif single is not None:
        lines.append(f"  one thread    not measured: {single}")
    if row.limit_ms is None:
        lines.append("  speed limit   none: `limits.max_latency_ms` is not set")
    else:
        lines.append(f"  speed limit   {_ms(speed.median_ms)} is within `limits.max_latency_ms: {row.limit_ms:g}`")
        if isinstance(single, Speed) and single.median_ms > row.limit_ms:
            lines.append(f"  warning       in one thread the usual answer takes {_ms(single.median_ms)}, over the "
                         "limit: a host without the isolation headers (GitHub Pages) misses it")
    lines.extend(f"  {line}" for line in describe_misses(score, items))
    return lines


def describe_misses(score: evaluate.Evaluation, items: Sequence[examples.Example]) -> list[str]:
    """One line per miss: its line in the examples file, the label named and its score, the
    expected one and its score, and the start of the text."""
    lines = []
    for miss in score.misses:
        item = items[miss.index]
        own = "no centre" if miss.expected_score is None else f"{miss.expected_score:.3f}"
        lines.append(f"miss, line {item.line}: {miss.predicted} {miss.score:.3f} "
                     f"instead of {miss.expected} {own}: {item.input[:60]}")
    return lines


def describe_browser(bench: Bench) -> str:
    version = re.search(r"Chrome/(\d+)", bench.device["userAgent"])
    name = f"Chrome {version.group(1)}" if version else "Chrome"
    window = "without a window" if bench.headless else "with a window"
    gpu = bench.device["webgpu"]
    if not gpu:
        adapter = "none"
    else:
        adapter = " ".join(part for part in (gpu["vendor"], gpu["architecture"]) if part) or "unnamed"
        if gpu["isFallbackAdapter"]:
            adapter += " (software)"
    return f"{name} {window}, WebGPU adapter: {adapter}"


def _report(row: Row) -> None:
    extra = f", usual answer {_ms(row.speed.median_ms)}" if row.chosen and row.speed is not None else ""
    _say(f"  {row.outcome.candidate.entry.id}: {row.verdict}, {row.message}{extra}")


def _shown(folder: os.PathLike[str] | str) -> str:
    """The folder as the user can type it: relative to the current folder when it lies inside."""
    path = os.path.abspath(folder)
    try:
        inside = os.path.commonpath([path, os.getcwd()]) == os.getcwd()
    except ValueError:  # another drive on Windows
        inside = False
    return os.path.relpath(path) if inside and path != os.getcwd() else path


def _quoted(text: str) -> str:
    return f'"{text}"' if " " in text else text


def _key(candidate: catalog.Candidate) -> tuple[str, str]:
    return candidate.entry.id, candidate.recipe.kind


def _tolerate_any_text() -> None:
    """Labels and examples may hold any script; a console or a pipe in a narrower code page
    must not end the build with UnicodeEncodeError. What it cannot show becomes ``?``."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def _say(message: str = "") -> None:
    print(message, flush=True)


def _mb(size: int) -> str:
    return f"{size / BYTES_PER_MB:.1f} MB"


def _ms(value: float) -> str:
    return f"{value:.1f} ms"


def _percent(share: float) -> str:
    return f"{share * 100:.1f}".removesuffix(".0") + "%"


def _first_line(error: BaseException) -> str:
    lines = str(error).strip().splitlines()
    return lines[0] if lines else type(error).__name__

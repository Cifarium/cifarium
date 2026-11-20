"""Does the build's score hold on examples the build did not see?

    python examples/measure.py [--part english|russian] [--refresh] [--report FILE]

For each part of the golden task (``english/``, ``russian/``) it takes every example, the 20 of
``examples.jsonl`` and the 100 of ``heldout.jsonl``, and turns each into a vector on the bench once per
catalog model. The vectors are kept outside git, in the cache of cifarium (``golden/``), so a second run
needs no browser. Everything after that is numpy and the code of the tool itself:

- sample 0 is the 20 examples of the repository; samples 1 to 5 are random 20 that keep the share of every
  label (a fixed seed); the rest of the 120 examples is the held-out part of each sample;
- for every sample and model: the score the build would see on the 20 (``evaluate.leave_one_out``) and the
  score of the centres of those 20 on the held-out part (``evaluate.held_out``);
- for every sample and both settings of ``license``: the model ``selection.select`` takes, and the
  held-out score of that model.

Then it prints the tables and the two cases of the fork in row 5 of the queue (NEXT.md).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cifarium import build, catalog, config, evaluate, examples, paths, selection, vendor
from cifarium.bench import decode_vectors, embed_job, open_bench

HERE = Path(__file__).resolve().parent
PARTS = ("english", "russian")
LICENSES = ("permissive", "any")
VECTOR_RECIPE = "embed-centroid"

SEED = 2026
RANDOM_SAMPLES = 5
SAMPLE_SIZE = 20
CHUNK = 25
"""Examples per job on the bench. A job loads the model once and has 300 s; Gemma needs about 2 s an example."""
MARGIN = 0.05
"""How far below ``pass_rate`` the held-out score of the chosen model may fall before the fork (case B) applies:
room for the noise of the held-out part itself."""
CASE_B_SAMPLES = 2
"""Case B applies when the margin is broken in at least this many of the random samples."""


@dataclass(frozen=True)
class Pool:
    """Every example of a part: the 20 of the repository first, then the held-out ones."""

    part: str
    settings: config.Config
    shipped: examples.Examples
    texts: tuple[str, ...]
    expected: tuple[str, ...]
    names: tuple[str, ...]
    build_count: int

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.texts, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Scores:
    """One model on one sample."""

    build_correct: int
    build_total: int
    held_correct: int
    held_total: int

    @property
    def build(self) -> float:
        return self.build_correct / self.build_total

    @property
    def held(self) -> float:
        return self.held_correct / self.held_total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--part", choices=PARTS, help="one part only (default: both)")
    parser.add_argument("--refresh", action="store_true", help="count the vectors again, ignoring the cache")
    parser.add_argument("--report", type=Path, help="also write the tables to this file (UTF-8)")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    pools = [load_pool(part) for part in ((args.part,) if args.part else PARTS)]
    candidates = vector_candidates()
    vectors, timings = collect(pools, candidates, args.refresh)

    lines: list[str] = []
    for pool in pools:
        lines += report_part(pool, candidates, vectors, timings)
    text = "\n".join(lines)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8", newline="\n")
    return 0


def vector_candidates() -> list[catalog.Candidate]:
    """The models this measurement is about: the ones that turn a text into a vector (``embed-centroid``).

    Not everything ``selection.READY_RECIPES`` lets the build run: ``zero-shot`` is ready there too, but
    its job gives scores of labels, not vectors (``embed_job`` refuses it) and the selection asks its
    function for them with the labels. The measurement keeps vectors and takes centres from them, so
    only these models go in, to ``collect`` and to ``selection.select`` alike."""
    return [c for c in catalog.candidates("classify") if c.recipe.kind == VECTOR_RECIPE]


def load_pool(part: str) -> Pool:
    folder = HERE / part
    settings = config.load(folder / "cifarium.yaml")
    shipped = examples.load(settings)
    held = examples.read(folder / "heldout.jsonl", settings.labels)
    items = [*shipped.items, *held.items]
    texts = tuple(item.input for item in items)
    if len({text.strip() for text in texts}) != len(texts):
        raise SystemExit(f"{part}: a text is in both files or twice in one")
    return Pool(part, settings, shipped, texts, tuple(item.expected for item in items),
                tuple(label.name for label in shipped.labels), len(shipped.items))


# --- the vectors ---------------------------------------------------------------------------------


def collect(
    pools: Sequence[Pool], candidates: Sequence[catalog.Candidate], refresh: bool
) -> tuple[dict[tuple[str, str, str], np.ndarray], dict[tuple[str, str, str], list[dict]]]:
    """Vectors and bench times of every candidate on every part: from the cache, or counted now.

    The bench is started only when something is missing.
    """
    vectors: dict[tuple[str, str, str], np.ndarray] = {}
    timings: dict[tuple[str, str, str], list[dict]] = {}
    missing = []
    for pool in pools:
        for candidate in candidates:
            key = (pool.part, *candidate_key(candidate))
            loaded = None if refresh else read_cache(pool, candidate)
            if loaded is None:
                missing.append((pool, candidate))
            else:
                vectors[key], timings[key] = loaded
    if missing:
        vendor.ensure_engine(report=lambda message: say(f"  {message}"))
        with open_bench() as bench:
            say(f"browser: {build.describe_browser(bench)}")
            for pool, candidate in missing:
                key = (pool.part, *candidate_key(candidate))
                vectors[key], timings[key] = embed_all(bench, pool, candidate)
                write_cache(pool, candidate, vectors[key], timings[key])
    return vectors, timings


def embed_all(bench, pool: Pool, candidate: catalog.Candidate) -> tuple[np.ndarray, list[dict]]:
    """Every text of the pool through the candidate. The first job is the 20 examples of the repository,
    so its time is the time the build spends on one candidate; the held-out ones follow in chunks."""
    entry = candidate.entry
    build.download(entry, say)
    n = len(pool.texts)
    bounds = [(0, pool.build_count)] + [(s, min(s + CHUNK, n)) for s in range(pool.build_count, n, CHUNK)]
    pieces, chunks = [], []
    for start, stop in bounds:
        job = embed_job(entry, candidate.recipe, pool.texts[start:stop])
        began = time.perf_counter()
        result = bench.run(job, timeout=build.TIME_LIMIT_S)
        wall = time.perf_counter() - began
        speed = build.Speed.of(result)
        pieces.append(decode_vectors(result))
        chunks.append({"count": stop - start, "wall_s": wall, "load_ms": speed.load_ms,
                       "median_ms": speed.median_ms, "threads": speed.threads})
        say(f"  {pool.part}/{entry.id}: {stop - start} examples in {wall:.1f} s "
            f"(model {speed.load_ms / 1000:.1f} s, {speed.median_ms:.0f} ms an example)")
    return np.concatenate(pieces).astype(np.float32), chunks


def candidate_key(candidate: catalog.Candidate) -> tuple[str, str]:
    return candidate.entry.id, candidate.recipe.kind


def cache_file(pool: Pool, candidate: catalog.Candidate) -> Path:
    return paths.cache_dir() / "golden" / pool.part / f"{candidate.entry.id}.npz"


def cache_key(pool: Pool, candidate: catalog.Candidate) -> str:
    """Names everything the vectors depend on: the texts, the model file, the recipe and the engine."""
    entry = candidate.entry
    what = [pool.fingerprint, entry.repo, entry.revision, sorted(entry.files().items()),
            entry.variants["cpu"].dtype, candidate.recipe.params, vendor.TRANSFORMERS_VERSION,
            vendor.ORT_WEB_VERSION]
    return hashlib.sha256(json.dumps(what, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_cache(pool: Pool, candidate: catalog.Candidate):
    path = cache_file(pool, candidate)
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["key"]) != cache_key(pool, candidate):
                return None
            return data["vectors"], json.loads(str(data["timing"]))
    except (OSError, KeyError, ValueError):
        return None


def write_cache(pool: Pool, candidate: catalog.Candidate, vectors: np.ndarray, timing: list[dict]) -> None:
    path = cache_file(pool, candidate)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vectors=vectors, key=np.array(cache_key(pool, candidate)), timing=np.array(json.dumps(timing)))


# --- the samples ----------------------------------------------------------------------------------


def stratified(expected: Sequence[str], names: Sequence[str], size: int, rng: random.Random) -> list[int]:
    """``size`` positions out of ``expected`` that keep the share of every label (largest remainders)."""
    by_label = {name: [i for i, label in enumerate(expected) if label == name] for name in names}
    quota = {name: size * len(found) / len(expected) for name, found in by_label.items()}
    counts = {name: int(share) for name, share in quota.items()}
    order = sorted(names, key=lambda name: (counts[name] - quota[name], names.index(name)))
    for name in order[: size - sum(counts.values())]:
        counts[name] += 1
    if min(counts.values()) < examples.MIN_EXAMPLES_PER_LABEL:
        raise SystemExit(f"a sample of {size} cannot hold {examples.MIN_EXAMPLES_PER_LABEL} examples per label: {counts}")
    return sorted(i for name in names for i in rng.sample(by_label[name], counts[name]))


def samples_of(pool: Pool) -> list[list[int]]:
    """Sample 0 is the repository's own 20; 1 to 5 are drawn with the fixed seed."""
    drawn = [list(range(pool.build_count))]
    for number in range(1, RANDOM_SAMPLES + 1):
        rng = random.Random(f"{SEED}/{pool.part}/sample/{number}")
        drawn.append(stratified(pool.expected, pool.names, SAMPLE_SIZE, rng))
    return drawn


def score(pool: Pool, vectors: np.ndarray, sample: Sequence[int]) -> Scores:
    """The build's score on the sample, and the score of its centres on every other example."""
    inside = list(sample)
    outside = [i for i in range(len(pool.texts)) if i not in set(inside)]
    expected = np.array(pool.expected, dtype=object)
    rate = pool.settings.pass_rate
    seen = evaluate.leave_one_out(vectors[inside], list(expected[inside]), pool.names, rate)
    centres = evaluate.centroids(vectors[inside], list(expected[inside]), pool.names)
    held = evaluate.held_out(centres, vectors[outside], list(expected[outside]), pool.names, rate)
    return Scores(seen.correct, seen.total, held.correct, held.total)


def choose(pool: Pool, sample: Sequence[int], license: str, candidates, vectors) -> str | None:
    """The id of the model ``selection.select`` takes on the sample, or None."""
    items = examples.Examples(
        path=pool.shipped.path, labels=pool.shipped.labels,
        items=tuple(examples.Example(n + 1, pool.texts[i], pool.expected[i]) for n, i in enumerate(sample)),
    )
    settings = dataclasses.replace(pool.settings, license=license)
    row_of = {text: i for i, text in enumerate(pool.texts)}

    def embed(candidate: catalog.Candidate, texts: Sequence[str]) -> np.ndarray:
        return vectors[(pool.part, *candidate_key(candidate))][[row_of[text] for text in texts]]

    chosen = selection.select(settings, items, candidates, embed).chosen
    return chosen.candidate.entry.id if chosen else None


# --- the report -------------------------------------------------------------------------------------


def report_part(pool, candidates, vectors, timings) -> list[str]:
    rate = pool.settings.pass_rate
    samples = samples_of(pool)
    ids = [c.entry.id for c in candidates]
    scores = {(s, c.entry.id): score(pool, vectors[(pool.part, *candidate_key(c))], sample)
              for s, sample in enumerate(samples) for c in candidates}
    held_total = len(pool.texts) - SAMPLE_SIZE
    counts = {name: pool.expected.count(name) for name in pool.names}
    out = [f"## {pool.part}: {len(pool.texts)} examples, labels {counts}; "
           f"{SAMPLE_SIZE} to build from, {held_total} held out per sample, pass rate {rate:g}", ""]

    out += ["Score on the 20 (leave one out) / score on the held-out part, in per cent. "
            "Sample 0 is the repository's 20; 1 to 5 are random.", ""]
    head = ["model"] + [f"s{s}" for s in range(len(samples))] + ["mean 1-5", "passes 1-5", "of them held < "
            f"{(rate - MARGIN) * 100:g}"]
    rows = []
    for model in ids:
        random_scores = [scores[(s, model)] for s in range(1, len(samples))]
        passing = [x for x in random_scores if x.build + 1e-9 >= rate]
        rows.append([model]
                    + [f"{scores[(s, model)].build * 100:.0f} / {scores[(s, model)].held * 100:.0f}"
                       for s in range(len(samples))]
                    + [f"{np.mean([x.build for x in random_scores]) * 100:.0f} / "
                       f"{np.mean([x.held for x in random_scores]) * 100:.0f}",
                       f"{len(passing)}/{len(random_scores)}",
                       str(sum(1 for x in passing if x.held + 1e-9 < rate - MARGIN))])
    out += table(head, rows) + [""]

    picks = {(s, lic): choose(pool, sample, lic, candidates, vectors)
             for s, sample in enumerate(samples) for lic in LICENSES}
    out += ["Model taken by `selection.select`, and its score on the held-out part:", ""]
    rows = []
    for s in range(len(samples)):
        row = [f"s{s}"]
        for lic in LICENSES:
            model = picks[(s, lic)]
            row.append("none" if model is None else
                       f"{model}: {scores[(s, model)].build * 100:.0f} / {scores[(s, model)].held * 100:.0f}")
        rows.append(row)
    out += table(["sample"] + [f"license: {lic}" for lic in LICENSES], rows) + [""]

    out += verdict(pool, samples, picks, scores)
    out += timing_lines(pool, candidates, timings)
    return out


def verdict(pool, samples, picks, scores) -> list[str]:
    rate = pool.settings.pass_rate
    out = []
    for lic in LICENSES:
        random_range = range(1, len(samples))
        none = [s for s in random_range if picks[(s, lic)] is None]
        broken = [s for s in random_range if picks[(s, lic)] is not None
                  and scores[(s, picks[(s, lic)])].held + 1e-9 < rate - MARGIN]
        out.append(f"license {lic}: no model passes on the 20 in {len(none)} of {len(random_range)} "
                   f"random samples {none}; the chosen model has a held-out score below "
                   f"{(rate - MARGIN) * 100:g}% in {len(broken)} of them {broken}")
        if lic == "any" and none:
            out.append(f"  CASE A ({pool.part}): no model of the catalog reaches the pass rate in samples {none}")
        if len(broken) >= CASE_B_SAMPLES:
            out.append(f"  CASE B ({pool.part}, license {lic}): the model taken falls more than "
                       f"{MARGIN * 100:g} points below the pass rate on held-out examples in samples {broken}")
    return out + [""]


def timing_lines(pool, candidates, timings) -> list[str]:
    out = [f"Time to check one candidate on the repository's {pool.build_count} examples "
           "(model files already in the cache; a new tab, the model, the examples), and over the rest:", ""]
    rows = []
    for candidate in candidates:
        chunks = timings[(pool.part, *candidate_key(candidate))]
        first, rest = chunks[0], chunks[1:]
        per_example = [chunk["median_ms"] for chunk in chunks]
        rows.append([candidate.entry.id, f"{first['wall_s']:.1f} s", f"{first['load_ms'] / 1000:.1f} s",
                     f"{first['median_ms']:.0f} ms", f"{np.median(per_example):.0f} ms",
                     f"{np.mean([chunk['wall_s'] / chunk['count'] for chunk in rest]):.2f} s"])
    return table(["model", "20 examples, wall", "model load", "an example (first job)",
                  "an example (all jobs)", "wall per example, held-out jobs"], rows) + [""]


def table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

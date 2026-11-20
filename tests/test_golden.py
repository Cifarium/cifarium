"""The golden task in ``examples/``: both parts read as the build reads them, and the scripts that
make and measure them are deterministic. No browser; the measurement itself is ``examples/measure.py``."""

from __future__ import annotations

import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from cifarium import catalog, config, examples
from cifarium.bench import embed_job

FOLDER = Path(__file__).resolve().parents[1] / "examples"
PARTS = ("english", "russian")


def script(name: str):
    """Load ``examples/<name>.py``; the folder is not a package."""
    module_name = f"golden_{name}"
    spec = importlib.util.spec_from_file_location(module_name, FOLDER / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("part", PARTS)
def test_the_part_reads_as_a_task_of_the_build(part):
    settings = config.load(FOLDER / part / "cifarium.yaml")
    built = examples.load(settings)

    assert settings.task == "classify"
    assert 3 <= len(built.labels) <= 6
    assert all(label.description for label in built.labels)
    assert len(built.items) == 20
    counts = Counter(item.expected for item in built.items)
    assert set(counts) == {label.name for label in built.labels}
    assert min(counts.values()) >= examples.MIN_EXAMPLES_PER_LABEL


@pytest.mark.parametrize("part", PARTS)
def test_the_held_out_part_has_the_same_labels_and_no_text_the_build_sees(part):
    settings = config.load(FOLDER / part / "cifarium.yaml")
    built = examples.load(settings)
    held = examples.read(FOLDER / part / "heldout.jsonl", settings.labels)

    assert len(held.items) == 100
    assert {label.name for label in held.labels} == {label.name for label in built.labels}
    assert not {item.input.strip() for item in built.items} & {item.input.strip() for item in held.items}
    # Every label has the same number of examples: 5 to build from and 25 held out.
    assert set(Counter(item.expected for item in built.items).values()) == {5}
    assert set(Counter(item.expected for item in held.items).values()) == {25}


def test_the_readme_names_the_sources_and_the_licence_of_both_parts():
    text = (FOLDER / "README.md").read_text(encoding="utf-8")

    for word in ("Banking77", "MASSIVE", "CC BY 4.0", "not support requests", "english", "russian"):
        assert word in text


def fake_rows(prepare, labels):
    """Enough made-up texts for every source group of ``labels``."""
    return [prepare.Text(f"{group} {n}", group, f"fake/{group}/{n}")
            for label in labels for group in label.groups for n in range(60)]


@pytest.mark.parametrize("which", ["ENGLISH_LABELS", "RUSSIAN_LABELS"])
def test_the_draw_is_balanced_spread_over_the_intents_and_repeatable(which):
    prepare = script("prepare")
    labels = getattr(prepare, which)
    rows = fake_rows(prepare, labels)

    drawn = prepare.draw("test", labels, rows)

    assert drawn == prepare.draw("test", labels, list(rows))
    assert Counter(label for _, label, _ in drawn) == {label.name: 30 for label in labels}
    assert Counter(label for _, label, built in drawn if built) == {label.name: 5 for label in labels}
    assert len({row.text for row, _, _ in drawn}) == 120
    for label in labels:
        per_group = Counter(row.group for row, name, _ in drawn if name == label.name)
        assert max(per_group.values()) - min(per_group.values()) <= 1
        assert set(per_group) == set(label.groups)


def test_the_draw_depends_on_the_part_so_that_changing_one_leaves_the_other():
    prepare = script("prepare")
    rows = fake_rows(prepare, prepare.ENGLISH_LABELS)

    assert prepare.draw("a", prepare.ENGLISH_LABELS, rows) != prepare.draw("b", prepare.ENGLISH_LABELS, rows)


def test_unambiguous_drops_repeats_and_texts_of_two_groups():
    prepare = script("prepare")
    rows = [prepare.Text("one", "a", "s1"), prepare.Text("one", "a", "s2"),
            prepare.Text("two", "a", "s3"), prepare.Text("two", "b", "s4"),
            prepare.Text("", "a", "s5"), prepare.Text("three", "b", "s6")]

    assert [row.text for row in prepare.unambiguous(rows)] == ["one", "three"]


@pytest.mark.parametrize("part", PARTS)
def test_the_samples_keep_the_labels_and_do_not_change_between_runs(part):
    measure = script("measure")
    pool = measure.load_pool(part)

    samples = measure.samples_of(pool)

    assert samples == measure.samples_of(measure.load_pool(part))
    assert len(samples) == 1 + measure.RANDOM_SAMPLES
    assert samples[0] == list(range(20))
    for sample in samples:
        assert len(sample) == len(set(sample)) == measure.SAMPLE_SIZE
        assert Counter(pool.expected[i] for i in sample) == {name: 5 for name in pool.names}
    assert len({tuple(sample) for sample in samples}) == len(samples)


def test_stratified_keeps_the_share_of_every_label_by_largest_remainders():
    measure = script("measure")
    expected = ["a"] * 50 + ["b"] * 30 + ["c"] * 20

    picked = measure.stratified(expected, ["a", "b", "c"], 20, random.Random(1))

    assert Counter(expected[i] for i in picked) == {"a": 10, "b": 6, "c": 4}
    assert picked == sorted(picked)
    # Equal remainders: the extra examples go to the labels listed first.
    even = ["a"] * 7 + ["b"] * 7 + ["c"] * 7
    assert Counter(even[i] for i in measure.stratified(even, ["a", "b", "c"], 20, random.Random(1))) == {
        "a": 7, "b": 7, "c": 6}


def test_the_measurement_takes_the_models_that_make_vectors_and_no_other_recipe():
    measure = script("measure")
    everything = catalog.candidates("classify")

    taken = measure.vector_candidates()

    # The catalog has a model of the recipe zero-shot, which the build runs but this script cannot.
    assert {"embed-centroid", "zero-shot"} <= {c.recipe.kind for c in everything}
    assert taken and {c.recipe.kind for c in taken} == {"embed-centroid"}
    assert [c.entry.id for c in taken] == [c.entry.id for c in everything if c.recipe.kind == "embed-centroid"]
    for candidate in taken:  # `embed_job` refuses a recipe that is not vectors, with ValueError
        embed_job(candidate.entry, candidate.recipe, ["a text"])


@pytest.mark.parametrize("license", ["permissive", "any"])
def test_the_choice_of_the_measurement_never_asks_for_the_scores_of_a_zero_shot_model(license):
    measure = script("measure")
    pool = measure.load_pool("english")
    candidates = measure.vector_candidates()
    rng = np.random.default_rng(0)
    vectors = {}
    for candidate in candidates:
        raw = rng.normal(size=(len(pool.texts), candidate.recipe.params["dimensions"]))
        vectors[("english", *measure.candidate_key(candidate))] = (
            raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)

    # Random vectors pass nobody, so the walk reaches every candidate and asks for its vectors; a
    # zero-shot one would get the labels as a third argument, or no key in `vectors`.
    assert measure.choose(pool, list(range(measure.SAMPLE_SIZE)), license, candidates, vectors) is None


def test_stratified_refuses_a_sample_that_leaves_a_label_with_one_example():
    measure = script("measure")
    expected = ["a"] * 95 + ["b"] * 5

    with pytest.raises(SystemExit, match="per label"):
        measure.stratified(expected, ["a", "b"], 20, random.Random(1))

"""The build command without a browser: the speed check, the walk after a slow candidate, what
is printed, and the exit codes. The bench is a fake in both layers: :func:`build.choose` gets a
fake bench function, and :func:`build.run` a fake Chrome behind the real catalog.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from publishable import example_leaks

import cifarium
from cifarium import build, bundle, catalog, cli, exitcodes, hub, paths, results, vendor
from cifarium.bench import BenchError
from cifarium.build import LATENCY, Speed, choose
from cifarium.config import Config, Label, Limits
from cifarium.examples import Example, Examples
from cifarium.selection import Verdict

MB = 1_000_000
ENGINE = MB
ENGINE_SIZES = {vendor.LIBRARY_FILE: ENGINE - 2, vendor.WASM_LOADER_FILE: 1, vendor.WASM_BINARY_FILE: 1}
"""The engine of the tests weighs one megabyte in all."""
LABELS = ("alpha", "beta")
EXPECTED = ["alpha", "beta"] * 10
DIMENSIONS = len(LABELS)
"""The fake vectors have one axis for each label; the centres of two labels weigh 16 bytes."""
CENTRES = 4 * len(LABELS) * DIMENSIONS


# --- choose, on a fake bench function ----------------------------------------------------------


def make(entry_id: str, model_mb: int, *, license: str = "apache-2.0") -> catalog.Candidate:
    """A candidate whose first load is ``model_mb`` megabytes, engine and centres of the labels apart."""
    files = {"onnx/model_quantized.onnx": model_mb * MB - 1 - CENTRES}
    variants = {"cpu": catalog.Variant("cpu", "wasm", "q8", files)}
    entry = catalog.Entry(Path(f"{entry_id}.yaml"), entry_id, f"org/{entry_id}", "0" * 40, license,
                          "org/source", ("multi",), {"config.json": 1}, variants, ())
    params = {"pooling": "mean", "prefix": "", "dimensions": DIMENSIONS}
    return catalog.Candidate(entry, catalog.Recipe("classify", "embed-centroid", params))


def task() -> Examples:
    items = tuple(Example(k + 1, f"text {k}", label) for k, label in enumerate(EXPECTED))
    return Examples(Path("examples.jsonl"), items, tuple(Label(name) for name in LABELS))


def settings(limit_ms: float | None = None, **changes) -> Config:
    config = Config(Path("cifarium.yaml"), "classify", Path("examples.jsonl"),
                    limits=Limits(max_size_mb=300, max_latency_ms=limit_ms), license="permissive")
    return replace(config, **changes)


def good() -> np.ndarray:
    """One axis per label: every example right."""
    return np.array([[1.0, 0.0] if label == "alpha" else [0.0, 1.0] for label in EXPECTED])


def bad() -> np.ndarray:
    """Every example on the same vector: half of them wrong."""
    return np.tile([1.0, 0.0], (len(EXPECTED), 1))


def speed(median_ms: float) -> Speed:
    return Speed(load_ms=100.0, first_ms=3 * median_ms, median_ms=median_ms, p95_ms=median_ms,
                 calls=len(EXPECTED) - 1, threads=4, isolated=True)


class FakeBench:
    """Answers per catalog id: vectors and the usual answer in milliseconds, or an exception."""

    def __init__(self, answers: dict[str, tuple[np.ndarray, float] | Exception]) -> None:
        self.answers = answers
        self.calls: list[str] = []

    def __call__(self, candidate: catalog.Candidate, texts, *labels) -> np.ndarray:
        self.calls.append(candidate.entry.id)
        answer = self.answers[candidate.entry.id]
        if isinstance(answer, Exception):
            raise answer
        return answer[0]

    def speed(self, candidate: catalog.Candidate) -> Speed:
        answer = self.answers[candidate.entry.id]
        assert not isinstance(answer, Exception) and candidate.entry.id in self.calls
        return speed(answer[1])


def run_choose(candidates, bench: FakeBench, config: Config) -> tuple[build.Result, list[build.Row]]:
    reported: list[build.Row] = []
    result = choose(config, task(), candidates, bench, bench.speed, engine_sizes=ENGINE_SIZES,
                    report=reported.append)
    return result, reported


def verdicts(rows) -> list[tuple[str, str]]:
    return [(row.outcome.candidate.entry.id, row.verdict) for row in rows]


def test_without_a_limit_the_lightest_passing_model_is_chosen_however_slow():
    bench = FakeBench({"a": (bad(), 5.0), "b": (good(), 5000.0), "c": (good(), 1.0)})

    result, _ = run_choose([make("c", 30), make("b", 20), make("a", 10)], bench, settings())

    assert verdicts(result.rows) == [("a", "accuracy"), ("b", "passed"), ("c", "not-tried")]
    assert result.chosen.outcome.candidate.entry.id == "b"
    assert np.array_equal(result.vectors, good())
    assert bench.calls == ["a", "b"]


def test_a_model_too_slow_hands_over_to_the_next_heavier_one():
    bench = FakeBench({"a": (good(), 80.0), "b": (bad(), 1.0), "c": (good(), 20.0), "d": (good(), 1.0)})

    result, reported = run_choose([make("a", 10), make("b", 20), make("c", 30), make("d", 40)],
                                  bench, settings(limit_ms=50))

    assert verdicts(result.rows) == [("a", LATENCY), ("b", "accuracy"), ("c", "passed"), ("d", "not-tried")]
    assert result.chosen.outcome.candidate.entry.id == "c"
    assert bench.calls == ["a", "b", "c"]
    # The slow one is reported as slow, never as chosen, and every run is reported once.
    assert verdicts(reported) == [("a", LATENCY), ("b", "accuracy"), ("c", "passed")]
    slow = result.rows[0]
    assert slow.outcome.verdict is Verdict.PASSED and not slow.chosen
    assert "the usual answer takes 80.0 ms, over `limits.max_latency_ms: 50`" in slow.message


def test_candidates_sorted_out_by_the_settings_appear_once_and_are_not_reported():
    bench = FakeBench({"a": (good(), 80.0), "c": (good(), 20.0)})
    candidates = [make("a", 10), make("gpl", 15, license="gpl-3.0"), make("big", 20, license="gpl-3.0"),
                  make("c", 30)]

    result, reported = run_choose(candidates, bench, settings(limit_ms=50))

    assert verdicts(result.rows) == [("a", LATENCY), ("gpl", "license"), ("big", "license"), ("c", "passed")]
    assert verdicts(reported) == [("a", LATENCY), ("c", "passed")]


def test_the_limit_is_inclusive():
    bench = FakeBench({"a": (good(), 50.0)})

    result, _ = run_choose([make("a", 10)], bench, settings(limit_ms=50))

    assert result.chosen is not None


def test_nothing_is_chosen_when_the_last_passing_model_is_too_slow():
    bench = FakeBench({"a": (bad(), 1.0), "b": (good(), 51.0)})

    result, _ = run_choose([make("a", 10), make("b", 20)], bench, settings(limit_ms=50))

    assert verdicts(result.rows) == [("a", "accuracy"), ("b", LATENCY)]
    assert result.chosen is None and result.vectors is None


def test_after_a_slow_model_the_walk_goes_on_by_position_even_at_equal_weight():
    bench = FakeBench({"x": (good(), 90.0), "y": (good(), 10.0)})

    result, _ = run_choose([make("x", 10), make("y", 10)], bench, settings(limit_ms=50))

    assert verdicts(result.rows) == [("x", LATENCY), ("y", "passed")]
    assert bench.calls == ["x", "y"]


def test_a_failed_bench_is_a_refusal_and_the_walk_goes_on():
    bench = FakeBench({"a": BenchError("out of memory"), "b": (good(), 1.0)})

    result, _ = run_choose([make("a", 10), make("b", 20)], bench, settings(limit_ms=50))

    assert verdicts(result.rows) == [("a", "bench"), ("b", "passed")]


def test_a_zero_shot_candidate_is_checked_on_speed_like_any_other():
    # Rows of 1 and 0 on two labels are a softmax over them as well as unit vectors.
    zero = replace(make("z", 10), recipe=catalog.Recipe("classify", "zero-shot", {}))
    bench = FakeBench({"z": (good(), 80.0), "b": (good(), 1.0)})

    result, _ = run_choose([zero, make("b", 20)], bench, settings(limit_ms=50))

    assert verdicts(result.rows) == [("z", LATENCY), ("b", "passed")]
    assert result.rows[0].outcome.evaluation.correct == 20


def test_the_usual_answer_is_the_median_of_the_calls_after_the_first():
    measured = Speed.of({"timesMs": [400.0, 10.0, 30.0, 20.0, 12.0], "loadMs": 900.0,
                         "threads": 4, "crossOriginIsolated": True})

    assert (measured.first_ms, measured.median_ms, measured.p95_ms, measured.calls) == (400.0, 16.0, 30.0, 4)
    assert (measured.load_ms, measured.threads, measured.isolated) == (900.0, 4, True)


def test_a_single_call_is_its_own_usual_answer():
    measured = Speed.of({"timesMs": [7.0], "loadMs": 1.0, "threads": 1, "crossOriginIsolated": False})

    assert (measured.median_ms, measured.calls) == (7.0, 1)


# --- run, on a fake Chrome behind the real catalog ---------------------------------------------


TOPICS = ("shipping", "returns", "technical")


class FakeChrome:
    """The page of the bench: one axis per topic word found in the text, steady times."""

    headless = True
    device = {"userAgent": "Mozilla/5.0 Chrome/157.0.0.0 Safari/537.36", "webgpu": None}

    def __init__(self, ms: float = 10.0, single_thread: Exception | None = None) -> None:
        self.ms = ms
        self.single_thread = single_thread
        self.ignores_threads = False
        self.removed: tuple[str, ...] = ()
        self.jobs: list[dict] = []

    def run(self, job: dict, *, timeout: float) -> dict:
        self.jobs.append(job)
        if job.get("threads") == 1 and self.single_thread is not None:
            raise self.single_thread
        times = [3 * self.ms] + [self.ms * (2 if job.get("threads") == 1 else 1)] * (len(job["inputs"]) - 1)
        measured = {"loadMs": 500.0, "timesMs": times, "threads": 4 if self.ignores_threads else job.get("threads", 4),
                    "crossOriginIsolated": True, "events": []}
        if job["operation"] == "zero-shot":
            # A hypothesis that names the topic of the text wins: a softmax over the labels.
            logits = np.array([[2.0 if any(topic in text and topic in sentence for topic in TOPICS) else 0.0
                                for sentence in job["hypotheses"]] for text in job["inputs"]])
            scores = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            return {"count": len(scores), "labels": len(job["hypotheses"]), "scores": scores.tolist(), **measured}
        rows = np.array([[1.0 if topic in text else 0.0 for topic in TOPICS] for text in job["inputs"]])
        rows = (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype("<f4")
        # The model of the real catalog returns as many numbers as the catalog says it does;
        # zeros after the topic axes keep the length and the angles of the vectors.
        promised = next(recipe.params["dimensions"] for candidate in catalog.candidates("classify")
                        for recipe in [candidate.recipe]
                        if candidate.entry.repo == job["model"] and recipe.kind == "embed-centroid")
        rows = np.pad(rows, ((0, 0), (0, promised - rows.shape[1])))
        return {"count": len(rows), "dims": promised,
                "vectors": base64.b64encode(rows.tobytes()).decode("ascii"), **measured}


@dataclass
class Written:
    """What the build asked :func:`bundle.write` to put into the folder."""

    folder: Path
    candidate: catalog.Candidate
    labels: list[Label]
    centres: np.ndarray | None
    """None for the recipe zero-shot, whose folder has no centres."""
    reports: dict[str, bytes]

    @property
    def names(self) -> list[str]:
        return [label.name for label in self.labels]


@pytest.fixture
def chrome(monkeypatch):
    """Replace the engine download, the model download, Chrome and the writing of the folder
    (tests/test_bundle.py checks that); return the fake Chrome with what it was asked to do."""
    fake = FakeChrome()
    downloads: list[str] = []
    written: list[Written] = []

    @contextmanager
    def fake_open_bench(*, headed: bool = False):
        yield fake

    def fake_write(folder, candidate, labels, centres, **kwargs):
        written.append(Written(Path(folder), candidate, list(labels), None if centres is None else np.array(centres),
                               dict(kwargs.get("reports") or {})))
        return bundle.Built(Path(folder), {}, ("cifarium.json", "index.html"), fake.removed,
                            first_load_bytes=51_200_000, size_bytes=52_000_000)

    monkeypatch.setattr(vendor, "ensure_engine", lambda **kwargs: None)
    monkeypatch.setattr(hub, "download_files", lambda repo, revision, files: downloads.append(repo))
    monkeypatch.setattr(build, "open_bench", fake_open_bench)
    monkeypatch.setattr(bundle, "write", fake_write)
    fake.downloads = downloads
    fake.written = written
    return fake


def write_task(folder: Path, settings: str = "", *, extra: tuple[tuple[str, str], ...] = ()) -> Path:
    lines = [json.dumps({"input": f"a question about {topic}, number {k}", "expected": topic})
             for k in range(4) for topic in TOPICS]
    lines += [json.dumps({"input": text, "expected": label}, ensure_ascii=False) for text, label in extra]
    (folder / "examples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    path = folder / "cifarium.yaml"
    path.write_text(f"task: classify\nexamples: examples.jsonl\n{settings}", encoding="utf-8")
    return path


def test_run_prints_the_table_and_the_chosen_model(tmp_path, chrome, capsys):
    code = build.run(write_task(tmp_path))

    out, err = capsys.readouterr()
    assert code == 0, err
    assert "Chosen: all-minilm-l6-v2 (Xenova/all-MiniLM-L6-v2, q8 on the processor" in out
    assert "accuracy      12 of 12 examples (100%), pass rate 85%" in out
    assert "per example   10.0 ms median" in out
    assert "threads       4, crossOriginIsolated" in out
    assert "one thread    20.0 ms median of 5 examples" in out
    assert "speed limit   none: `limits.max_latency_ms` is not set" in out
    # Every catalog candidate is in the table, the sorted-out ones with their reason.
    for candidate in catalog.candidates("classify"):
        assert candidate.entry.id in out
    assert "license gemma does not pass `license: permissive`" in out
    # One run of the chosen model, one single-thread run on a few examples.
    assert chrome.downloads == ["Xenova/all-MiniLM-L6-v2"]
    assert [job.get("threads") for job in chrome.jobs] == [None, 1]
    assert len(chrome.jobs[1]["inputs"]) == build.SINGLE_THREAD_INPUTS


def test_run_writes_the_folder_of_the_chosen_model_with_the_centres_of_its_labels(tmp_path, chrome, capsys):
    code = build.run(write_task(tmp_path))

    out, err = capsys.readouterr()
    assert code == 0, err
    [written] = chrome.written
    assert written.folder.resolve() == (tmp_path / "cifarium-out").resolve()
    assert written.candidate.entry.id == "all-minilm-l6-v2"
    # The labels in the order of their first example; the fake page puts each topic on its own axis.
    assert written.names == list(TOPICS)
    # 384 numbers, the length the catalog names for all-MiniLM: the topic axes first, then zeros.
    assert written.centres.shape == (3, 384)
    assert np.allclose(written.centres[:, :3], np.eye(3)) and not written.centres[:, 3:].any()
    assert "Wrote " in out and "2 files, 52.0 MB on disk; a first visit downloads 51.2 MB" in out
    assert "removed" not in out
    assert "index.html is an example page and every build writes it again" in out
    assert "python -m http.server 8000 --directory " in out and "http://localhost:8000/" in out
    assert "sends no isolation headers" in out


def test_run_hands_the_descriptions_of_the_labels_to_the_folder_and_nothing_of_the_examples(tmp_path, chrome, capsys):
    settings = ("labels:\n  - shipping: Where the parcel is\n  - returns\n  - technical: Something is broken\n")

    code = build.run(write_task(tmp_path, settings))

    _, err = capsys.readouterr()
    assert code == 0, err
    [written] = chrome.written
    # The folder gets the labels of the settings and the centres, which are numbers: no example.
    assert written.labels == [Label("shipping", "Where the parcel is"), Label("returns"),
                              Label("technical", "Something is broken")]
    assert written.centres.dtype.kind == "f"


def test_run_writes_to_the_output_of_the_settings_and_says_what_it_removed(tmp_path, chrome, capsys):
    chrome.removed = ("models/old/a", "models/old/b")

    code = build.run(write_task(tmp_path, "output: ./public/ml\n"))

    out, _ = capsys.readouterr()
    assert code == 0
    assert chrome.written[0].folder.resolve() == (tmp_path / "public" / "ml").resolve()
    assert "removed 2 files of the previous build that this one does not use" in out


def test_a_folder_that_cannot_be_written_ends_with_its_message_and_code_3(tmp_path, chrome, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise bundle.BundleError("cannot write the folder: Access is denied")

    monkeypatch.setattr(bundle, "write", fail)

    code = build.run(write_task(tmp_path))

    out, err = capsys.readouterr()
    assert code == exitcodes.ENVIRONMENT == 3
    assert "FAILED: cannot write the folder: Access is denied" in err
    assert "Traceback" not in err and "Wrote " not in out


def test_host_hub_is_refused_before_the_bench_starts_with_code_2(tmp_path, chrome, capsys):
    code = build.run(write_task(tmp_path, "host: hub\n"))

    out, err = capsys.readouterr()
    assert code == exitcodes.BAD_INPUT == 2
    assert "host: hub is not supported yet" in err and "host: bundle" in err
    assert out == "" and chrome.jobs == [] and chrome.downloads == [] and chrome.written == []


def test_the_folder_is_shown_as_the_user_can_type_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert build._shown(tmp_path / "out") == "out"
    assert build._shown(tmp_path / "public" / "ml") == os.path.join("public", "ml")
    assert build._shown(tmp_path) == str(tmp_path)
    assert build._shown(tmp_path.parent / "elsewhere") == str(tmp_path.parent / "elsewhere")
    assert build._quoted("my site") == '"my site"' and build._quoted("site") == "site"


def test_a_miss_of_the_chosen_model_names_its_line_in_the_examples_file(tmp_path, chrome, capsys):
    # Line 13: about shipping, but labelled technical. 12 of 13 is 92 %, above the pass rate.
    code = build.run(write_task(tmp_path, extra=(("доставка: a question about shipping", "technical"),)))

    out, _ = capsys.readouterr()
    assert code == 0
    assert "accuracy      12 of 13 examples (92.3%)" in out
    assert "miss, line 13: shipping 1.000 instead of technical 0.000: доставка: a question about shipping" in out


def test_run_goes_past_slow_models_and_fails_when_none_is_fast_enough(tmp_path, chrome, capsys):
    code = build.run(write_task(tmp_path, "license: permissive\nlimits:\n  max_latency_ms: 5\n"))

    out, err = capsys.readouterr()
    assert code == exitcodes.NO_MODEL == 1
    assert "FAILED: no model passed" in err
    assert "Chosen" not in out and "Wrote " not in out and chrome.written == []
    # Every vector model of the catalog that the settings let through ran once and was too slow.
    ran = [job["model"] for job in chrome.jobs]
    assert len(ran) == len(set(ran)) > 1
    assert out.count(f"  {LATENCY}  ") == len(ran)


# --- the report, which the folder gets or which a refusal leaves behind ---------------------------------


def texts_of(folder: Path) -> list[str]:
    return [json.loads(line)["input"] for line in (folder / "examples.jsonl").read_text(encoding="utf-8").splitlines()]


def test_run_hands_both_files_of_the_report_to_the_folder_with_the_weight_of_its_manifest(tmp_path, chrome, capsys):
    code = build.run(write_task(tmp_path, "limits:\n  max_latency_ms: 15\n"))

    out, err = capsys.readouterr()
    assert code == 0, err
    [written] = chrome.written
    assert set(written.reports) == {"report.json", "report.md"}
    document = json.loads(written.reports["report.json"])
    assert (document["outcome"], document["exitCode"]) == ("built", 0)
    assert document["settings"]["limits"] == {"maxSizeMb": 300, "maxLatencyMs": 15}
    assert document["device"] == {"browser": {"name": "Chrome", "version": 157, "headless": True}, "webgpu": None,
                                  "threads": 4, "crossOriginIsolated": True}
    chosen = document["chosen"]
    assert chosen["id"] == "all-minilm-l6-v2" and chosen["model"] == "Xenova/all-MiniLM-L6-v2"
    # The weight is that of the manifest the folder gets, which is the number of the table.
    manifest = bundle.manifest_of(written.candidate, written.labels, written.centres.astype("<f4").nbytes)
    assert chosen["firstLoad"]["bytes"] == bundle.first_load(manifest, "cpu")[0]
    row = next(row for row in document["candidates"] if row["id"] == "all-minilm-l6-v2")
    assert row["verdict"] == "passed" and row["firstLoadBytes"] == chosen["firstLoad"]["bytes"]
    assert row["evaluation"]["correct"] == row["evaluation"]["total"] == 12
    assert chosen["oneThread"]["medianMs"] == 20.0 and chosen["oneThreadError"] is None
    assert "report.md says why each model was taken or not" in out
    # The report is made of the same numbers the console showed.
    assert [row["id"] for row in document["candidates"]] == [
        candidate.entry.id for candidate in selection_order(written.labels)]


def selection_order(labels):
    from cifarium import selection
    return selection.ordered(catalog.candidates("classify"), labels)


def test_the_report_of_a_run_holds_no_text_of_an_example(tmp_path, chrome, capsys):
    write_task(tmp_path, extra=(("доставка: a question about shipping", "technical"),))

    code = build.run(tmp_path / "cifarium.yaml")

    assert code == 0, capsys.readouterr().err
    [written] = chrome.written
    folder = tmp_path / "published"
    bundle.write_reports(folder, written.reports)
    # A miss is there, by its line (13) and its labels, and so is every file of the report.
    assert '"line": 13' in written.reports["report.json"].decode("utf-8")
    assert example_leaks(folder, texts_of(tmp_path)) == []


def test_a_refusal_writes_its_report_into_output_and_ends_with_code_1(tmp_path, chrome, capsys):
    code = build.run(write_task(tmp_path, "license: permissive\nlimits:\n  max_latency_ms: 5\n"))

    out, err = capsys.readouterr()
    assert code == exitcodes.NO_MODEL == 1
    assert "FAILED: no model passed" in err
    folder = tmp_path / "cifarium-out"
    assert sorted(path.name for path in folder.iterdir()) == ["report.json", "report.md"]
    assert chrome.written == []
    document = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert (document["outcome"], document["exitCode"], document["chosen"]) == ("no-model", 1, None)
    verdicts = {row["id"]: row["verdict"] for row in document["candidates"]}
    assert {candidate.entry.id for candidate in catalog.candidates("classify")} == set(verdicts)
    assert LATENCY in verdicts.values() and "license" in verdicts.values()
    assert "No model passed" in (folder / "report.md").read_text(encoding="utf-8")
    assert f"The report is in {build._shown(folder)}: report.md for you, report.json for scripts." in out
    assert "Wrote " not in out and "earlier build" not in out
    assert example_leaks(folder, texts_of(tmp_path)) == []


def test_a_refusal_leaves_the_folder_of_an_earlier_build_as_it_was_and_says_so(tmp_path, chrome, capsys):
    folder = tmp_path / "cifarium-out"
    folder.mkdir()
    (folder / "cifarium.json").write_text('{"format": 1}', encoding="utf-8")
    (folder / "index.html").write_text("<p>mine</p>", encoding="utf-8")
    (folder / "report.json").write_text("an older report", encoding="utf-8")

    code = build.run(write_task(tmp_path, "limits:\n  max_latency_ms: 5\n"))

    out, _ = capsys.readouterr()
    assert code == 1
    assert (folder / "cifarium.json").read_text(encoding="utf-8") == '{"format": 1}'
    assert (folder / "index.html").read_text(encoding="utf-8") == "<p>mine</p>"
    assert json.loads((folder / "report.json").read_text(encoding="utf-8"))["outcome"] == "no-model"
    assert f"{build._shown(folder)} also holds the folder of an earlier build: this run did not change it." in out


def test_a_refusal_whose_report_cannot_be_written_is_code_3_and_not_1(tmp_path, chrome, monkeypatch, capsys):
    def fail(folder, reports):
        raise bundle.BundleError("cannot write the folder: Access is denied")

    monkeypatch.setattr(bundle, "write_reports", fail)

    code = build.run(write_task(tmp_path, "limits:\n  max_latency_ms: 5\n"))

    out, err = capsys.readouterr()
    assert code == exitcodes.ENVIRONMENT == 3
    assert "FAILED: cannot write the folder: Access is denied" in err and "no model passed" not in err
    assert "Every candidate:" in out, "the table is on the console whatever happens to the file"
    assert "The report is in" not in out


def test_an_output_that_is_a_file_is_a_mistake_of_the_settings_before_anything_runs(tmp_path, chrome, capsys):
    (tmp_path / "cifarium-out").write_text("in the way", encoding="utf-8")

    code = build.run(write_task(tmp_path))

    _, err = capsys.readouterr()
    assert code == exitcodes.BAD_INPUT == 2 and "cifarium-out" in err
    assert chrome.jobs == [] and (tmp_path / "cifarium-out").read_text(encoding="utf-8") == "in the way"


def test_a_build_that_fails_before_the_selection_ends_writes_no_report(tmp_path, chrome, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise vendor.VendorError("Could not download onnxruntime-web: offline")

    monkeypatch.setattr(vendor, "ensure_engine", fail)

    assert build.run(write_task(tmp_path)) == 3
    assert not (tmp_path / "cifarium-out").exists()
    capsys.readouterr()


def test_run_warns_when_one_thread_misses_the_limit(tmp_path, chrome, capsys):
    code = build.run(write_task(tmp_path, "limits:\n  max_latency_ms: 15\n"))

    out, _ = capsys.readouterr()
    assert code == 0
    assert "speed limit   10.0 ms is within `limits.max_latency_ms: 15`" in out
    assert "in one thread the usual answer takes 20.0 ms, over the limit" in out


def test_a_failed_single_thread_run_changes_nothing_but_the_line(tmp_path, chrome, capsys):
    chrome.single_thread = BenchError("The job exceeded its time limit of 300 s.")

    code = build.run(write_task(tmp_path))

    out, _ = capsys.readouterr()
    assert code == 0
    assert "one thread    not measured: The job exceeded its time limit of 300 s." in out


def test_a_page_that_ignores_one_thread_is_not_taken_for_a_single_thread_measurement(tmp_path, chrome, capsys):
    chrome.ignores_threads = True

    code = build.run(write_task(tmp_path))

    out, _ = capsys.readouterr()
    assert code == 0
    assert "one thread    not measured: asked the page for 1 thread, it computed in 4" in out


@pytest.mark.parametrize(
    "where, error",
    [("download", hub.HubError("Could not download onnx/model_quantized.onnx of Xenova/all-MiniLM-L6-v2: offline")),
     ("engine", vendor.VendorError("Could not download onnxruntime-web: offline")),
     ("chrome", BenchError("Could not start Google Chrome: no such file"))],
)
def test_a_download_or_chrome_failure_ends_with_its_message_and_code_3(tmp_path, chrome, monkeypatch, capsys,
                                                                       where, error):
    def fail(*args, **kwargs):
        raise error

    target = {"download": (hub, "download_files"), "engine": (vendor, "ensure_engine"),
              "chrome": (build, "open_bench")}[where]
    monkeypatch.setattr(*target, fail)

    code = build.run(write_task(tmp_path))

    _, err = capsys.readouterr()
    assert code == exitcodes.ENVIRONMENT == 3
    assert f"FAILED: {error}" in err
    assert "Traceback" not in err


def test_a_disk_or_permission_error_in_the_cache_is_the_environment_too(tmp_path, chrome, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(vendor, "ensure_engine", fail)

    code = build.run(write_task(tmp_path))

    _, err = capsys.readouterr()
    assert code == 3 and "FAILED: " in err and "Permission denied" in err and "Traceback" not in err


def test_settings_that_do_not_load_end_with_their_problems_and_code_2(tmp_path, chrome, capsys):
    code = build.run(tmp_path / "cifarium.yaml")

    _, err = capsys.readouterr()
    assert code == exitcodes.BAD_INPUT == 2
    assert "cifarium.yaml" in err
    assert chrome.jobs == []


def test_examples_that_do_not_load_end_with_their_problems_and_code_2(tmp_path, chrome, capsys):
    settings = write_task(tmp_path)
    (tmp_path / "examples.jsonl").write_text('{"input": "a"}\n', encoding="utf-8")

    code = build.run(settings)

    _, err = capsys.readouterr()
    assert code == 2 and "line 1" in err
    assert chrome.jobs == [] and chrome.downloads == []


@pytest.mark.parametrize("line", ["license: commercial", "documents: ./d.jsonl", "pass_rate: 85", "host: cdn"])
def test_every_kind_of_wrong_setting_is_code_2_and_starts_nothing(tmp_path, chrome, capsys, line):
    code = build.run(write_task(tmp_path, line + "\n"))

    _, err = capsys.readouterr()
    assert code == 2 and err.strip()
    assert chrome.jobs == [] and chrome.downloads == [] and chrome.written == []


def test_a_catalog_that_cannot_be_read_is_a_damaged_installation_code_3(tmp_path, chrome, monkeypatch, capsys):
    def broken(task):
        raise catalog.CatalogError(Path("catalog/x.yaml"), ["revision: missing"])

    monkeypatch.setattr(catalog, "candidates", broken)

    code = build.run(write_task(tmp_path))

    _, err = capsys.readouterr()
    assert code == 3
    assert "x.yaml: revision: missing" in err and "install cifarium again" in err
    assert chrome.jobs == []


def test_what_cifarium_did_not_expect_is_code_3_with_the_traceback_and_never_1(tmp_path, chrome, monkeypatch, capsys):
    def surprise(*args, **kwargs):
        raise KeyError("a bug")

    monkeypatch.setattr(build, "choose", surprise)

    code = cli.main(["build", str(write_task(tmp_path))])

    _, err = capsys.readouterr()
    assert code == 3
    assert "Traceback" in err and "KeyError" in err and "did not expect" in err and "issues" in err


def test_an_unknown_command_or_flag_is_code_2(capsys):
    for argv in (["nonsense"], ["build", "--bogus"]):
        with pytest.raises(SystemExit) as caught:
            cli.main(argv)
        assert caught.value.code == 2
    capsys.readouterr()


def test_text_a_narrow_code_page_cannot_show_becomes_question_marks(monkeypatch):
    raw = io.BytesIO()
    narrow = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", narrow)

    build._tolerate_any_text()
    print("label доставка")
    narrow.flush()

    assert raw.getvalue() == b"label ????????" + os.linesep.encode()


def test_the_command_line_has_build_and_no_smoke(tmp_path, chrome, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    listed = capsys.readouterr().out
    assert "build" in listed and "smoke" not in listed

    settings = write_task(tmp_path)
    assert cli.main(["build", str(settings)]) == 0
    monkeypatch.chdir(tmp_path)
    assert cli.main(["build"]) == 0


# --- the cache of results: the same answer, the vectors kept ----------------------------------


def table_of(out: str) -> list[str]:
    """The lines of the table of every candidate, as printed."""
    lines = out.splitlines()
    start = lines.index("Every candidate:") + 1
    return lines[start:lines.index("", start)]


def report_of(written: Written) -> dict:
    return json.loads(written.reports["report.json"])


def chosen_speed(written: Written) -> dict:
    return next(row for row in report_of(written)["candidates"] if row["verdict"] == "passed")["speed"]


def jobs_of(chrome: FakeChrome) -> list[tuple[int | None, int]]:
    """What the bench was asked: the number of threads, if fixed, and the number of texts."""
    return [(job.get("threads"), len(job["inputs"])) for job in chrome.jobs]


def minilm() -> catalog.Candidate:
    return next(c for c in catalog.candidates("classify") if c.entry.id == "all-minilm-l6-v2")


def test_a_second_build_on_the_same_examples_computes_no_vector_and_answers_the_same(tmp_path, chrome, capsys):
    path = write_task(tmp_path)
    assert build.run(path) == 0
    first, _ = capsys.readouterr()
    assert build.run(path) == 0
    second, err = capsys.readouterr()

    assert "all-minilm-l6-v2: running 12 examples on the processor" in first
    assert "Vectors of the examples: 12 computed on the bench." in first
    assert "all-minilm-l6-v2: 12 vectors from the cache" in second
    assert "Vectors of the examples: all 12 taken from the cache." in second
    assert "running" not in second
    assert f"The cache is {paths.results_dir()}; delete that folder to compute every vector again." in second
    # What the bench was asked in the second build: the times of the chosen model, one thread.
    assert jobs_of(chrome) == [(None, 12), (1, 6), (None, 12), (1, 6)]
    # The same table, the same choice, the same centres, so the same folder.
    assert table_of(first) == table_of(second)
    assert "Chosen: all-minilm-l6-v2" in second
    assert chrome.written[0].centres.tobytes() == chrome.written[1].centres.tobytes()
    assert chrome.written[0].names == chrome.written[1].names


def test_the_times_are_measured_in_the_build_that_prints_them_and_never_taken_from_the_cache(
        tmp_path, chrome, capsys):
    path = write_task(tmp_path)
    assert build.run(path) == 0
    capsys.readouterr()
    chrome.ms = 25.0  # the machine got slower between the builds

    assert build.run(path) == 0
    out, _ = capsys.readouterr()

    assert "all-minilm-l6-v2: measuring the speed on 12 examples" in out
    assert "per example   25.0 ms median" in out and "10.0 ms" not in out
    assert chosen_speed(chrome.written[0])["medianMs"] == 10.0
    assert chosen_speed(chrome.written[1])["medianMs"] == 25.0


def test_the_run_for_the_times_of_cached_vectors_is_short_on_a_long_set(tmp_path, chrome, capsys, monkeypatch):
    monkeypatch.setattr(build, "SPEED_INPUTS", 5)
    path = write_task(tmp_path)
    assert build.run(path) == 0
    capsys.readouterr()

    assert build.run(path) == 0
    out, _ = capsys.readouterr()

    assert "measuring the speed on 5 examples" in out
    # The first build, with nothing cached, measured on all of them: as it always did.
    assert jobs_of(chrome) == [(None, 12), (1, 6), (None, 5), (1, 6)]


def test_an_example_that_changes_is_the_only_vector_computed(tmp_path, chrome, capsys):
    path = write_task(tmp_path)
    assert build.run(path) == 0
    capsys.readouterr()
    examples_file = tmp_path / "examples.jsonl"
    examples_file.write_text(examples_file.read_text(encoding="utf-8").replace("number 0", "number 100", 1),
                             encoding="utf-8")

    assert build.run(path) == 0
    out, _ = capsys.readouterr()

    assert "all-minilm-l6-v2: running 1 examples on the processor, 11 more from the cache" in out
    assert "Vectors of the examples: 11 taken from the cache, 1 computed on the bench." in out
    assert chrome.jobs[2]["inputs"] == ["a question about shipping, number 100"]
    # One text is too few for a median: the times come from a run of their own.
    assert "measuring the speed on 12 examples" in out
    assert jobs_of(chrome)[2:] == [(None, 1), (None, 12), (1, 6)]


def test_examples_that_are_added_are_the_only_vectors_computed(tmp_path, chrome, capsys):
    assert build.run(write_task(tmp_path)) == 0
    capsys.readouterr()
    path = write_task(tmp_path, extra=(("one more question about returns", "returns"),
                                       ("and another about shipping", "shipping")))

    assert build.run(path) == 0
    out, _ = capsys.readouterr()

    assert chrome.jobs[2]["inputs"] == ["one more question about returns", "and another about shipping"]
    assert "Vectors of the examples: 12 taken from the cache, 2 computed on the bench." in out


def test_a_change_of_anything_the_vector_depends_on_computes_everything_again(tmp_path, chrome, capsys, monkeypatch):
    path = write_task(tmp_path)
    assert build.run(path) == 0
    capsys.readouterr()

    for patch in (lambda: monkeypatch.setattr(vendor, "TRANSFORMERS_VERSION", "9.9.9"),
                  lambda: monkeypatch.setattr(cifarium, "__version__", "9.9.9"),
                  lambda: monkeypatch.setattr(results, "page_digest", lambda: "another page")):
        patch()
        assert build.run(path) == 0
        out, _ = capsys.readouterr()
        assert "running 12 examples on the processor" in out and "from the cache" not in out
        assert "Vectors of the examples: 12 computed on the bench." in out


def test_a_cache_file_that_is_damaged_does_not_stop_the_build_and_is_replaced(
        tmp_path, chrome, capsys, private_results):
    path = write_task(tmp_path)
    assert build.run(path) == 0
    capsys.readouterr()
    [kept] = list(private_results.glob("*.npz"))
    kept.write_bytes(os.urandom(2048))

    assert build.run(path) == 0
    again, err = capsys.readouterr()
    assert build.run(path) == 0
    third, _ = capsys.readouterr()

    assert "running 12 examples on the processor" in again and err == ""
    assert "all 12 taken from the cache" in third


def test_a_cache_that_cannot_be_written_does_not_fail_the_build(tmp_path, chrome, capsys, monkeypatch):
    blocker = tmp_path / "results"
    blocker.write_text("a file where the folder should be", encoding="utf-8")
    monkeypatch.setattr(paths, "results_dir", lambda cache=None: blocker)

    code = build.run(write_task(tmp_path))

    out, err = capsys.readouterr()
    assert code == 0, err
    assert "all-minilm-l6-v2: the vectors could not be saved to the cache" in out
    assert "the next build computes them again" in out
    assert "Chosen: all-minilm-l6-v2" in out


def test_a_build_removes_the_files_of_the_cache_nobody_used_for_long_and_says_so(
        tmp_path, chrome, capsys, private_results):
    private_results.mkdir()
    old = private_results / "gone-0123456789abcdef0123.npz"
    old.write_bytes(b"x")
    long_ago = old.stat().st_mtime - (results.MAX_AGE_DAYS + 1) * 24 * 3600
    os.utime(old, (long_ago, long_ago))

    assert build.run(write_task(tmp_path)) == 0

    out, _ = capsys.readouterr()
    assert f"cache     removed 1 old file of earlier builds from {private_results}" in out
    assert not old.exists() and len(list(private_results.glob("*.npz"))) == 1


def test_a_failed_bench_is_not_kept(tmp_path, chrome, capsys, monkeypatch, private_results):
    def fail(job, *, timeout):
        raise BenchError("the tab crashed")

    monkeypatch.setattr(chrome, "run", fail)

    assert build.run(write_task(tmp_path)) == exitcodes.NO_MODEL

    capsys.readouterr()
    assert not private_results.exists() or list(private_results.iterdir()) == []


def test_a_run_for_the_times_that_fails_is_a_refusal_of_that_candidate_and_the_walk_goes_on():
    bench = FakeBench({"a": (good(), 1.0), "b": (good(), 1.0)})
    asked: list[str] = []

    def speed_of(candidate: catalog.Candidate) -> Speed:
        asked.append(candidate.entry.id)
        if candidate.entry.id == "a":
            raise BenchError("the tab crashed\nwith a second line")
        return bench.speed(candidate)

    reported: list[build.Row] = []
    result = choose(settings(), task(), [make("a", 10), make("b", 20)], bench, speed_of,
                    engine_sizes=ENGINE_SIZES, report=reported.append)

    assert verdicts(result.rows) == [("a", "bench"), ("b", "passed")]
    assert result.chosen.outcome.candidate.entry.id == "b"
    assert "the bench failed while measuring the speed: the tab crashed" in result.rows[0].message
    assert result.rows[0].outcome.error == "the tab crashed\nwith a second line"
    assert result.rows[0].outcome.evaluation is None and result.rows[0].speed is None
    assert verdicts(reported) == [("a", "bench"), ("b", "passed")]
    # Each candidate was measured once, though the report and the walk both ask for its row.
    assert asked == ["a", "b"]


def test_a_page_that_returns_other_vectors_than_it_was_asked_for_is_a_failed_bench_when_some_came_from_the_cache(
        tmp_path, chrome, monkeypatch):
    embed = build.ChromeEmbed(chrome, lambda message: None, results.Cache(tmp_path / "results"))
    texts = ["a question about shipping", "a question about returns"]
    embed(minilm(), texts)
    real = chrome.run

    def doubled(job, *, timeout):
        answer = real(job, timeout=timeout)
        rows = np.frombuffer(base64.b64decode(answer["vectors"]), dtype="<f4").reshape(answer["count"], -1)
        return {**answer, "count": 2 * len(rows), "vectors": base64.b64encode(np.vstack([rows, rows]).tobytes())}

    monkeypatch.setattr(chrome, "run", doubled)

    with pytest.raises(BenchError, match="asked the page for 1 vectors, it returned 2"):
        embed(minilm(), [*texts, "a question about technical things"])
    # With nothing from the cache the answer goes on as the page gave it, and the selection judges it.
    assert len(build.ChromeEmbed(chrome, lambda message: None)(minilm(), texts)) == 4


def test_the_same_text_twice_is_computed_once_and_both_places_get_its_vector(tmp_path, chrome):
    embed = build.ChromeEmbed(chrome, lambda message: None, results.Cache(tmp_path / "results"))

    vectors = embed(minilm(), ["about shipping", "about returns", "about shipping"])

    assert chrome.jobs[0]["inputs"] == ["about shipping", "about returns"]
    assert vectors.shape[0] == 3 and np.array_equal(vectors[0], vectors[2])
    assert (embed.reused, embed.computed) == (0, 2)


def test_without_a_cache_the_bench_gets_every_text_and_nothing_is_written(tmp_path, chrome):
    lines: list[str] = []
    embed = build.ChromeEmbed(chrome, lines.append)
    texts = ["about shipping", "about returns"]

    embed(minilm(), texts)
    embed(minilm(), texts)

    assert [job["inputs"] for job in chrome.jobs] == [texts, texts]
    assert lines.count("  all-minilm-l6-v2: running 2 examples on the processor") == 2
    assert not (tmp_path / "results").exists()


def test_what_the_build_says_about_the_cache():
    assert build.describe_cache(0, 0, "x") == []
    assert build.describe_cache(0, 5, "x")[1] == "Vectors of the examples: 5 computed on the bench."
    assert build.describe_cache(5, 0, "x")[1] == "Vectors of the examples: all 5 taken from the cache."
    assert build.describe_cache(4, 1, "x")[1] == (
        "Vectors of the examples: 4 taken from the cache, 1 computed on the bench.")
    assert build.describe_cache(4, 1, "some folder")[2] == (
        "  The cache is some folder; delete that folder to compute every vector again.")


# --- the recipe zero-shot -----------------------------------------------------------------------


def nli() -> catalog.Candidate:
    return next(c for c in catalog.candidates("classify") if c.entry.id == "nli-deberta-v3-xsmall")


def test_a_zero_shot_candidate_runs_every_text_with_the_hypotheses_of_its_labels_and_keeps_nothing(tmp_path, chrome):
    lines: list[str] = []
    embed = build.ChromeEmbed(chrome, lines.append, results.Cache(tmp_path / "results"))
    texts = ["about shipping", "about returns", "about shipping"]
    labels = [Label("shipping", "where the shipping is"), Label("returns"), Label("technical")]

    scores = embed(nli(), texts, labels)

    [job] = chrome.jobs
    assert job["operation"] == "zero-shot" and job["inputs"] == texts and "threads" not in job
    assert job["hypotheses"] == ["This example is about shipping.", "This example is about returns.",
                                 "This example is about technical."]
    assert scores.shape == (3, 3) and np.argmax(scores, axis=1).tolist() == [0, 1, 0]
    # "downloading" comes before it only on a machine without the model in its cache (a runner).
    assert lines[-1] == "  nli-deberta-v3-xsmall: running 3 examples on the processor, 3 labels each"
    assert all("running" not in line for line in lines[:-1])
    # The cache keeps vectors only: nothing is read, counted or written for the scores.
    assert (embed.reused, embed.computed) == (0, 0)
    assert not (tmp_path / "results").exists()
    # The times of this run are the candidate's in this build: no second run for them.
    assert embed.speed(nli()).median_ms == 10.0 and len(chrome.jobs) == 1
    embed(nli(), texts, labels)
    assert len(chrome.jobs) == 2


def test_the_single_thread_run_of_a_zero_shot_candidate_carries_its_labels(chrome):
    embed = build.ChromeEmbed(chrome, lambda message: None)

    speed = embed.single_thread(nli(), [f"about shipping {k}" for k in range(8)], [Label(t) for t in TOPICS])

    [job] = chrome.jobs
    assert (job["operation"], job["threads"], len(job["inputs"])) == ("zero-shot", 1, build.SINGLE_THREAD_INPUTS)
    assert len(job["hypotheses"]) == len(TOPICS) and speed.threads == 1


def test_a_zero_shot_candidate_asked_without_its_labels_is_a_defect(chrome):
    with pytest.raises(ValueError, match="needs the labels"):
        build.ChromeEmbed(chrome, lambda message: None)(nli(), ["about shipping"])


def test_a_chosen_zero_shot_candidate_is_printed_and_its_folder_and_report_are_written_with_code_0(
        tmp_path, chrome, monkeypatch, capsys):
    only = nli()
    monkeypatch.setattr(catalog, "candidates", lambda task: (only,))

    code = build.run(write_task(tmp_path))

    out, err = capsys.readouterr()
    assert code == exitcodes.DONE == 0, err
    assert "nli-deberta-v3-xsmall: running 12 examples on the processor, 3 labels each" in out
    assert any(line.strip().startswith("nli-deberta-v3-xsmall  zero-shot") and "passed" in line
               for line in table_of(out))
    assert "Chosen: nli-deberta-v3-xsmall" in out and "accuracy      12 of 12 examples (100%)" in out
    assert "one thread    20.0 ms median of 5 examples" in out
    assert "Vectors of the examples" not in out and "cannot write the folder" not in err
    assert "Wrote " in out
    assert [(job["operation"], job.get("threads")) for job in chrome.jobs] == [("zero-shot", None), ("zero-shot", 1)]
    # The folder is asked for with the labels and no centres: a zero-shot folder has none to write.
    [written] = chrome.written
    assert written.candidate is only and written.names == list(TOPICS) and written.centres is None
    # The report of the build is the one every other recipe gets, and its weight is the folder's.
    document = json.loads(written.reports[bundle.REPORT_JSON])
    assert (document["outcome"], document["exitCode"]) == ("built", 0)
    chosen = document["chosen"]
    assert (chosen["id"], chosen["recipe"]) == ("nli-deberta-v3-xsmall", "zero-shot")
    load = chosen["firstLoad"]
    assert load["dataBytes"] == 0 and load["bytes"] == load["engineBytes"] + load["modelBytes"]
    assert load["bytes"] == bundle.first_load(
        bundle.manifest_of(only, [Label(topic) for topic in TOPICS], 0), "cpu")[0]
    [entry] = document["candidates"]
    assert (entry["verdict"], entry["evaluation"]["correct"], entry["evaluation"]["total"]) == ("passed", 12, 12)


def test_the_centres_of_a_folder_are_those_of_the_examples_for_vectors_and_none_for_scores(tmp_path):
    items = Examples(path=tmp_path / "e.jsonl", labels=(Label("a"), Label("b")),
                     items=(Example(1, "x", "a"), Example(2, "y", "a"), Example(3, "z", "b"), Example(4, "w", "b")))
    vectors = np.array([[1, 0], [0, 1], [1, 1], [1, -1]], dtype=np.float64) / np.sqrt([[1], [1], [2], [2]])

    centres = build.centres_of(make("v", 20), vectors, items)

    assert centres is not None and centres.dtype == np.dtype("<f4") and centres.shape == (2, 2)
    assert np.linalg.norm(centres, axis=1) == pytest.approx([1.0, 1.0], abs=1e-6)
    assert centres[0] == pytest.approx(np.array([1, 1]) / np.sqrt(2), abs=1e-6)
    # What the selection keeps for zero-shot is the scores of the labels: no centres come from them.
    assert build.centres_of(nli(), np.array([[0.7, 0.3], [0.2, 0.8]]), items) is None

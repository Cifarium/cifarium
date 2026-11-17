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
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cifarium import build, catalog, cli, hub, vendor
from cifarium.bench import BenchError
from cifarium.build import LATENCY, Speed, choose
from cifarium.config import Config, Label, Limits
from cifarium.examples import Example, Examples
from cifarium.selection import Verdict

MB = 1_000_000
ENGINE = MB
LABELS = ("alpha", "beta")
EXPECTED = ["alpha", "beta"] * 10


# --- choose, on a fake bench function ----------------------------------------------------------


def make(entry_id: str, model_mb: int, *, license: str = "apache-2.0") -> catalog.Candidate:
    """A candidate whose first load is ``model_mb`` megabytes without the engine."""
    variants = {"cpu": catalog.Variant("cpu", "wasm", "q8", {"onnx/model_quantized.onnx": model_mb * MB - 1})}
    entry = catalog.Entry(Path(f"{entry_id}.yaml"), entry_id, f"org/{entry_id}", "0" * 40, license,
                          "org/source", ("multi",), {"config.json": 1}, variants, ())
    return catalog.Candidate(entry, catalog.Recipe("classify", "embed-centroid", {"pooling": "mean", "prefix": ""}))


def task() -> Examples:
    items = tuple(Example(k + 1, f"text {k}", label) for k, label in enumerate(EXPECTED))
    return Examples(Path("examples.jsonl"), items, tuple(Label(name) for name in LABELS))


def settings(limit_ms: float | None = None, **changes) -> Config:
    config = Config(Path("cifarium.yaml"), "classify", Path("examples.jsonl"),
                    limits=Limits(max_size_mb=300, max_latency_ms=limit_ms), license="commercial")
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

    def __call__(self, candidate: catalog.Candidate, texts) -> np.ndarray:
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
    result = choose(config, task(), candidates, bench, bench.speed, engine_bytes=ENGINE, report=reported.append)
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
        self.jobs: list[dict] = []

    def run(self, job: dict, *, timeout: float) -> dict:
        self.jobs.append(job)
        if job.get("threads") == 1 and self.single_thread is not None:
            raise self.single_thread
        rows = np.array([[1.0 if topic in text else 0.0 for topic in TOPICS] for text in job["inputs"]])
        rows = (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype("<f4")
        times = [3 * self.ms] + [self.ms * (2 if job.get("threads") == 1 else 1)] * (len(rows) - 1)
        return {"count": len(rows), "dims": len(TOPICS), "loadMs": 500.0, "timesMs": times,
                "vectors": base64.b64encode(rows.tobytes()).decode("ascii"),
                "threads": 4 if self.ignores_threads else job.get("threads", 4),
                "crossOriginIsolated": True, "events": []}


@pytest.fixture
def chrome(monkeypatch):
    """Replace the engine download, the model download and Chrome; return the fake Chrome."""
    fake = FakeChrome()
    downloads: list[str] = []

    @contextmanager
    def fake_open_bench(*, headed: bool = False):
        yield fake

    monkeypatch.setattr(vendor, "ensure_engine", lambda **kwargs: None)
    monkeypatch.setattr(hub, "download_files", lambda repo, revision, files: downloads.append(repo))
    monkeypatch.setattr(build, "open_bench", fake_open_bench)
    fake.downloads = downloads
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
    assert "license gemma does not pass `license: commercial`" in out
    # One run of the chosen model, one single-thread run on a few examples, nothing written.
    assert chrome.downloads == ["Xenova/all-MiniLM-L6-v2"]
    assert [job.get("threads") for job in chrome.jobs] == [None, 1]
    assert len(chrome.jobs[1]["inputs"]) == build.SINGLE_THREAD_INPUTS
    assert not (tmp_path / "cifarium-out").exists()


def test_a_miss_of_the_chosen_model_names_its_line_in_the_examples_file(tmp_path, chrome, capsys):
    # Line 13: about shipping, but labelled technical. 12 of 13 is 92 %, above the pass rate.
    code = build.run(write_task(tmp_path, extra=(("доставка: a question about shipping", "technical"),)))

    out, _ = capsys.readouterr()
    assert code == 0
    assert "accuracy      12 of 13 examples (92.3%)" in out
    assert "miss, line 13: shipping 1.000 instead of technical 0.000: доставка: a question about shipping" in out


def test_run_goes_past_slow_models_and_fails_when_none_is_fast_enough(tmp_path, chrome, capsys):
    code = build.run(write_task(tmp_path, "license: commercial\nlimits:\n  max_latency_ms: 5\n"))

    out, err = capsys.readouterr()
    assert code == 1
    assert "FAILED: no model passed" in err
    assert "Chosen" not in out
    # Every vector model of the catalog that the settings let through ran once and was too slow.
    ran = [job["model"] for job in chrome.jobs]
    assert len(ran) == len(set(ran)) > 1
    assert out.count(f"  {LATENCY}  ") == len(ran)


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
def test_a_download_or_chrome_failure_ends_with_its_message_and_code_1(tmp_path, chrome, monkeypatch, capsys,
                                                                       where, error):
    def fail(*args, **kwargs):
        raise error

    target = {"download": (hub, "download_files"), "engine": (vendor, "ensure_engine"),
              "chrome": (build, "open_bench")}[where]
    monkeypatch.setattr(*target, fail)

    code = build.run(write_task(tmp_path))

    _, err = capsys.readouterr()
    assert code == 1
    assert f"FAILED: {error}" in err
    assert "Traceback" not in err


def test_settings_that_do_not_load_end_with_their_problems_and_code_1(tmp_path, chrome, capsys):
    code = build.run(tmp_path / "cifarium.yaml")

    _, err = capsys.readouterr()
    assert code == 1
    assert "cifarium.yaml" in err
    assert chrome.jobs == []


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

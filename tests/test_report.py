"""The report of a build without a browser: what ``report.json`` and ``report.md`` say about the
candidates, the weight of the first load, the margin over the pass rate and the misses, that the two
files agree, that no text of an example and no folder of the machine reaches them, and that the files
are written the way the folder writes everything (ARCHITECTURE.md, sections 7, 9 and 14).
The keys of ``report.json`` as a contract are in ``tests/test_contracts.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from publishable import example_leaks
from test_build import ENGINE_SIZES, FakeBench, bad, good, make, settings, speed
from test_bundle import Sources, fill, fill_engine, MODEL_FILES, GRAPH

from cifarium import bundle, exitcodes, report
from cifarium.bench import BenchError
from cifarium.build import LATENCY, Result, choose
from cifarium.config import Label
from cifarium.examples import Example, Examples

NAMES = ("alpha", "beta")
EXPECTED = ["alpha", "beta"] * 10
SENTENCES = [f"Zebra {k} quartz marmalade xylophone telegram {k * 7919}" for k in range(len(EXPECTED))]
"""Texts no word of the report may repeat: whole, cut into pieces of 20 characters, or escaped."""

CHROME = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/157.0.0.0 Safari/537.36")
BENCH_DEVICE = {
    "userAgent": CHROME, "hardwareConcurrency": 8, "deviceMemory": 8.0, "crossOriginIsolated": True,
    "webgpu": {"vendor": "intel", "architecture": "gen-12lp", "device": "", "isFallbackAdapter": False,
               "description": "Intel(R) Iris(R) Xe Graphics", "shaderF16": True},
    "webgpuError": None,
}
DEVICE = report.device_of(BENCH_DEVICE, True)


def items() -> Examples:
    """The examples of the other test files, with texts of their own and the line numbers of a file
    that starts with two comment lines, so that a line number is not an index."""
    made = tuple(Example(k + 3, text, label) for k, (text, label) in enumerate(zip(SENTENCES, EXPECTED)))
    return Examples(Path("examples.jsonl"), made, tuple(Label(name) for name in NAMES))


def two_misses():
    """The vectors of ``good`` with two examples on the other label's axis: 18 of 20 right."""
    vectors = good()
    vectors[0], vectors[1] = [0.0, 1.0], [1.0, 0.0]
    return vectors


def build(answers: dict, limit_ms=None, **changes) -> Result:
    candidates = [make(name, 10 * (k + 1)) for k, name in enumerate(answers)]
    bench = FakeBench(answers)
    return choose(settings(limit_ms, **changes), items(), candidates, bench, bench.speed, engine_sizes=ENGINE_SIZES)


def manifest_for(result: Result) -> dict:
    candidate = result.chosen.outcome.candidate
    return bundle.manifest_of(candidate, items().labels, bundle.centres_bytes(candidate, 2),
                              engine_sizes=ENGINE_SIZES)


def of(result: Result, *, limit_ms=None, single=None, **changes) -> dict:
    manifest = None if result.chosen is None else manifest_for(result)
    return report.of_build(settings(limit_ms, **changes), items(), result, device=DEVICE, single=single,
                           manifest=manifest)


def success() -> dict:
    """A slow model, one that is too inaccurate, the chosen one with two misses and one that was never
    tried; a limit, so that the slow one has the word `latency`, and a single thread that misses it."""
    result = build({"a": (good(), 80.0), "b": (bad(), 1.0), "c": (two_misses(), 20.0), "d": (good(), 1.0)},
                   limit_ms=50)
    return of(result, limit_ms=50, single=speed(60.0))


def refusal() -> dict:
    return of(build({"a": (bad(), 5.0), "b": (bad(), 5.0)}))


def by_id(document: dict) -> dict:
    return {row["id"]: row for row in document["candidates"]}


# --- what the report says -------------------------------------------------------------------------------


def test_the_report_of_a_build_says_what_became_of_every_candidate():
    document = success()

    assert document["outcome"] == "built" and document["exitCode"] == exitcodes.DONE == 0
    assert [(row["id"], row["verdict"]) for row in document["candidates"]] == [
        ("a", LATENCY), ("b", "accuracy"), ("c", "passed"), ("d", "not-tried")]
    rows = by_id(document)
    assert rows["a"]["reason"].endswith("over `limits.max_latency_ms: 50`")
    assert rows["d"]["reason"] == "a lighter model already passed"
    # Only a candidate that ran on the bench and passed on accuracy has times; only a scored one a share.
    assert [row["speed"] is not None for row in document["candidates"]] == [True, False, True, False]
    assert [row["evaluation"] is not None for row in document["candidates"]] == [True, True, True, False]
    assert document["chosen"]["id"] == "c"
    assert (document["task"], document["settings"]["passRate"]) == ("classify", 0.85)
    assert document["settings"]["limits"] == {"maxSizeMb": 300, "maxLatencyMs": 50}
    assert document["examples"] == {"file": "examples.jsonl", "count": 20, "labels": ["alpha", "beta"]}
    assert rows["c"]["speed"] == {"loadMs": 100.0, "firstMs": 60.0, "medianMs": 20.0, "p95Ms": 20.0,
                                  "calls": 19, "threads": 4, "crossOriginIsolated": True}


def test_the_first_load_is_the_one_weight_given_by_parts():
    result = build({"a": (good(), 1.0)})
    document = of(result)

    weight = document["chosen"]["firstLoad"]
    # The table of the selection, the manifest and the report count the same bytes.
    assert weight["bytes"] == result.chosen.outcome.size_bytes == document["candidates"][0]["firstLoadBytes"]
    assert weight["bytes"] == bundle.first_load(manifest_for(result), "cpu")[0]
    assert weight["bytes"] == weight["engineBytes"] + weight["dataBytes"] + weight["modelBytes"]
    assert weight["engineBytes"] == sum(ENGINE_SIZES.values())
    assert weight["dataBytes"] == 4 * 2 * 2  # two labels, two numbers each, four bytes a number


def test_the_margin_over_the_pass_rate_is_shown_and_is_negative_for_a_model_that_fell_short():
    rows = by_id(success())

    assert rows["c"]["evaluation"]["correct"] == 18 and rows["c"]["evaluation"]["total"] == 20
    assert rows["c"]["evaluation"]["accuracy"] == 0.9
    assert rows["c"]["evaluation"]["margin"] == 0.05
    assert rows["b"]["evaluation"]["accuracy"] == 0.5
    assert rows["b"]["evaluation"]["margin"] == -0.35
    # A share exactly on the pass rate has margin 0.0, never -0.0.
    assert report._round(0.85 - 0.85000000001, 4) == 0.0
    assert str(report._round(-0.00001, 4)) == "0.0"


def test_the_pass_rate_stays_a_plain_share_and_the_margin_only_shows():
    # 17 of 20 is exactly 0.85: it passes, with margin 0 (decided on 17.11.2026).
    vectors = good()
    for row in (0, 1, 2):
        vectors[row] = [vectors[row][1], vectors[row][0]]

    document = of(build({"a": (vectors, 1.0)}))

    row = document["candidates"][0]
    assert (row["verdict"], row["evaluation"]["correct"], row["evaluation"]["margin"]) == ("passed", 17, 0.0)


def test_a_miss_is_a_line_of_the_examples_file_and_two_labels_and_no_text():
    row = by_id(success())["c"]

    misses = row["evaluation"]["misses"]
    # Examples 0 and 1 are lines 3 and 4 of the file.
    assert [(miss["line"], miss["expected"], miss["predicted"]) for miss in misses] == [
        (3, "alpha", "beta"), (4, "beta", "alpha")]
    assert all(set(miss) == {"line", "expected", "predicted", "score", "expectedScore"} for miss in misses)
    assert all(-1.0 <= miss["score"] <= 1.0 for miss in misses)
    assert example_leaks_in(success(), SENTENCES) == []


def test_the_report_of_a_refusal_has_every_reason_and_no_model():
    document = refusal()

    assert document["outcome"] == "no-model" and document["exitCode"] == exitcodes.NO_MODEL == 1
    assert document["chosen"] is None
    assert [row["verdict"] for row in document["candidates"]] == ["accuracy", "accuracy"]
    assert all(row["reason"] for row in document["candidates"])
    assert all(row["evaluation"]["margin"] < 0 and row["speed"] is None for row in document["candidates"])


def test_a_chosen_model_without_the_manifest_of_its_folder_is_refused():
    result = build({"a": (good(), 1.0)})

    with pytest.raises(ValueError, match="manifest"):
        report.of_build(settings(), items(), result, device=DEVICE)


def test_the_one_thread_run_is_given_or_its_reason():
    result = build({"a": (good(), 1.0)})

    measured = of(result, single=speed(40.0))["chosen"]
    assert measured["oneThread"] == {"firstMs": 120.0, "medianMs": 40.0, "p95Ms": 40.0, "calls": 19}
    assert measured["oneThreadError"] is None

    failed = of(result, single="The job exceeded its time limit of 300 s.")["chosen"]
    assert failed["oneThread"] is None and failed["oneThreadError"] == "The job exceeded its time limit of 300 s."

    skipped = of(result)["chosen"]
    assert skipped["oneThread"] is None and skipped["oneThreadError"] is None


# --- the device and the machine -------------------------------------------------------------------------


def test_the_device_is_cut_to_the_version_of_chrome_and_the_kind_of_the_adapter():
    assert DEVICE == {
        "browser": {"name": "Chrome", "version": 157, "headless": True},
        "webgpu": {"vendor": "intel", "architecture": "gen-12lp", "isFallbackAdapter": False},
    }
    text = report.json_text(success()).decode("utf-8")
    # The user agent, the name of the card, the cores and the memory are not for a published file.
    for left_out in ("Windows NT", "Mozilla", "Iris", "hardwareConcurrency", "deviceMemory", "shaderF16"):
        assert left_out not in text


def test_a_machine_without_an_adapter_or_with_a_software_one_is_told_so():
    none = report.device_of({"userAgent": "Chrome/150.1.2.3", "webgpu": None}, False)
    software = report.device_of({"userAgent": "Chrome/150", "webgpu": {
        "vendor": "google", "architecture": "swiftshader", "isFallbackAdapter": True}}, True)
    unknown = report.device_of({"userAgent": "Firefox/1"}, True)

    assert none == {"browser": {"name": "Chrome", "version": 150, "headless": False}, "webgpu": None}
    assert software["webgpu"] == {"vendor": "google", "architecture": "swiftshader", "isFallbackAdapter": True}
    assert unknown["browser"]["version"] is None
    document = success()
    assert (document["device"]["threads"], document["device"]["crossOriginIsolated"]) == (4, True)
    assert refusal()["device"]["threads"] is None and refusal()["device"]["crossOriginIsolated"] is None


def test_the_folders_of_the_machine_do_not_reach_the_report():
    home = Path.home()
    crash = BenchError(f"The job failed in the browser: cannot open {home}\\cache\\model.onnx "
                       f"({home.as_posix()}/cache/model.onnx)")
    bench = FakeBench({"a": crash, "b": (good(), 1.0)})
    result = choose(settings(), items(), [make("a", 10), make("b", 20)], bench, bench.speed,
                    engine_sizes=ENGINE_SIZES)

    document = of(result, single=f"could not start in {home}")

    text = report.json_text(document).decode("utf-8") + report.markdown_text(document).decode("utf-8")
    for spelling in (str(home), home.as_posix(), json.dumps(str(home))[1:-1]):
        assert spelling.lower() not in text.lower()
    assert "~" in by_id(document)["a"]["reason"] and "~" in document["chosen"]["oneThreadError"]
    assert by_id(document)["a"]["verdict"] == "bench"


# --- the two files ---------------------------------------------------------------------------------------


def test_the_json_is_utf8_with_the_format_and_the_version_first_and_the_same_every_time():
    raw = report.json_text(success())

    assert raw == report.json_text(success())
    assert raw.endswith(b"\n") and b"\r" not in raw
    document = json.loads(raw)
    assert list(document)[:2] == ["format", "cifarium"] and document["format"] == report.FORMAT == 1
    assert document == success()


def test_the_markdown_says_what_the_json_says():
    document = success()

    text = report.markdown_text(document).decode("utf-8")
    assert text.startswith("# cifarium build report\n") and text.endswith("\n") and "\r" not in text
    assert "**Chosen: c**" in text
    score = by_id(document)["c"]["evaluation"]
    assert f"Examples right: {score['correct']} of {score['total']} (90%)" in text
    assert "margin is +5.0 points (one example is worth 5.0 points)" in text
    weight = document["chosen"]["firstLoad"]
    assert f"First load: {weight['bytes'] / 1e6:.1f} MB = " in text
    assert "One thread: median 60.0 ms of 19 examples" in text
    assert "- line 3: `beta`" in text and "instead of `alpha`" in text and "Misses of c" in text
    assert "Warning: in one thread the usual answer takes 60.0 ms, over the limit" in text
    for row in document["candidates"]:
        assert f"| {row['id']} | embed-centroid | " in text and row["verdict"] in text
    assert "Chrome 157 without a window. WebGPU adapter: intel gen-12lp." in text
    assert "text of an example is not in this file" in text


def test_the_markdown_of_a_refusal_says_that_no_model_passed_and_why_each_was_not_taken():
    document = refusal()

    text = report.markdown_text(document).decode("utf-8")
    assert "**No model passed.**" in text and "Chosen" not in text
    assert text.count("| accuracy |") == 2 and "-35.0 points" in text
    assert "Threads of the engine: not measured." in text


def test_a_pipe_or_a_line_break_in_a_reason_does_not_break_the_table():
    document = refusal()
    document["candidates"][0]["reason"] = "one | two\nthree"

    text = report.markdown_text(document).decode("utf-8")
    assert "one \\| two three |" in text


# --- the folder ------------------------------------------------------------------------------------------


@pytest.fixture
def sources(tmp_path: Path) -> Sources:
    engine, model = tmp_path / "cache" / "engine", tmp_path / "cache" / "model"
    fill_engine(engine)
    fill(model, {**MODEL_FILES, **GRAPH})
    return Sources(engine, model)


def example_leaks_in(document: dict, texts) -> list[str]:
    """What :func:`publishable.example_leaks` finds in the two files of a report."""
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        bundle.write_reports(Path(folder), report.files(document))
        return example_leaks(Path(folder), texts)


def test_neither_file_of_a_report_holds_a_text_of_an_example_in_a_folder_that_got_it(sources, tmp_path):
    out = tmp_path / "site"

    sources.write(out, reports=report.files(success()))

    assert sorted(p.name for p in out.glob("report.*")) == ["report.json", "report.md"]
    assert example_leaks(out, SENTENCES) == []


def test_neither_file_of_a_refusal_holds_a_text_of_an_example(tmp_path):
    out = tmp_path / "site"

    bundle.write_reports(out, report.files(refusal()))

    assert example_leaks(out, SENTENCES) == []
    # The search can see what it looks for: put one sentence into the file and it finds it.
    (out / "report.md").write_text("see " + SENTENCES[3], encoding="utf-8")
    assert example_leaks(out, SENTENCES)


def test_the_report_names_the_files_of_the_task_and_not_the_folders_they_lie_in():
    document = success()

    assert document["settings"]["file"] == "cifarium.yaml" and document["examples"]["file"] == "examples.jsonl"
    assert "\\" not in json.dumps(document["settings"]) and "/" not in document["settings"]["file"]

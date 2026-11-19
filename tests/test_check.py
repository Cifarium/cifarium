"""The check command without a browser: what it reads from a folder and refuses, the four judgements
and their limits, what it serves and what it prints, and the exit codes. The folder is written from
the real manifest and the real report of a build (tests/test_report.py), the browser is a fake that
asks the real server for the page and the files, and the page itself runs in tests/test_check_e2e.py.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import shutil
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from test_build import good
from test_report import build as build_result
from test_report import manifest_for, of

from cifarium import bundle, check, cli, config, exitcodes, hosting, paths, report
from cifarium.bench import BenchError
from cifarium.check import FAIL, INFO, OK, WARN, Folder, Measures

LABELS = ["alpha", "beta"]
EXPECTED = LABELS * 10
LIMIT_MS = 50.0


def example_lines(expected=EXPECTED) -> str:
    return "".join(json.dumps({"input": f"text {k} of {label}", "expected": label}) + "\n"
                   for k, label in enumerate(expected))


@pytest.fixture()
def built(tmp_path) -> Path:
    """A folder with the files `check` reads: the manifest and the report of a build that chose a
    model (labels `alpha` and `beta`, pass rate 85 %, limit 50 ms) and the three hosting files."""
    result = build_result({"c": (good(), 20.0)}, limit_ms=LIMIT_MS)
    folder = tmp_path / "out"
    (folder / "hosting").mkdir(parents=True)
    (folder / bundle.MANIFEST).write_text(json.dumps(manifest_for(result), indent=2) + "\n", encoding="utf-8")
    (folder / bundle.REPORT_JSON).write_bytes(report.json_text(of(result, limit_ms=LIMIT_MS)))
    for name in hosting.FILES.values():
        shutil.copyfile(paths.web_dir() / "templates" / "hosting" / name, folder / "hosting" / name)
    (folder / "index.html").write_text("<!doctype html><title>the page of the site</title>", encoding="utf-8")
    (folder / "cifarium.js").write_text("export {};", encoding="utf-8")
    return folder


@pytest.fixture()
def examples(tmp_path) -> Path:
    path = tmp_path / "held-out.jsonl"
    path.write_text(example_lines(), encoding="utf-8")
    return path


def edit(folder: Path, name: str, change) -> None:
    path = folder / name
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read(folder: Path) -> Folder:
    return check.read_folder(folder)


def refused(folder: Path) -> check.CheckError:
    with pytest.raises(check.CheckError) as error:
        check.read_folder(folder)
    return error.value


# --- what is read from the folder --------------------------------------------------------------------


def test_a_folder_gives_the_labels_of_the_manifest_and_the_numbers_of_its_report(built):
    folder = read(built)
    document = json.loads((built / "report.json").read_text(encoding="utf-8"))

    assert [label.name for label in folder.labels] == LABELS
    assert folder.pass_rate == 0.85 and folder.max_latency_ms == LIMIT_MS
    assert folder.first_load_bytes == document["chosen"]["firstLoad"]["bytes"]
    assert folder.variant == "cpu" and folder.model == "org/c"


def test_a_build_without_a_limit_gives_none(tmp_path):
    result = build_result({"c": (good(), 20.0)})
    out = tmp_path / "out"
    out.mkdir()
    (out / bundle.MANIFEST).write_text(json.dumps(manifest_for(result)), encoding="utf-8")
    (out / bundle.REPORT_JSON).write_bytes(report.json_text(of(result)))

    assert read(out).max_latency_ms is None


def test_a_folder_that_is_not_there_is_refused():
    error = refused(Path("no-such-folder"))

    assert "no such folder" in str(error) and error.code == exitcodes.BAD_INPUT


@pytest.mark.parametrize("name", [bundle.MANIFEST, bundle.REPORT_JSON])
def test_a_folder_without_its_manifest_or_its_report_is_refused_with_the_name_of_the_file(built, name):
    (built / name).unlink()

    error = refused(built)

    assert f"{name}: file not found" in str(error) and error.code == exitcodes.BAD_INPUT


@pytest.mark.parametrize("name, change, expected", [
    ("cifarium.json", lambda m: m.update(format=2), "manifest format 2, newer than the 1"),
    ("cifarium.json", lambda m: m.pop("format"), "no `format` number"),
    ("cifarium.json", lambda m: m.update(format=True), "no `format` number"),
    ("cifarium.json", lambda m: m.update(task="search"), "task 'search'"),
    ("cifarium.json", lambda m: m.update(labels=[{"name": "alpha"}]), "`labels` is not a list of two or more"),
    ("cifarium.json", lambda m: m.update(labels="alpha"), "`labels` is not a list of two or more"),
    ("report.json", lambda r: r.update(format=2), "report format 2, newer than the 1"),
    ("report.json", lambda r: r.update(outcome="no-model"), "The last build chose no model"),
    ("report.json", lambda r: r["settings"].pop("passRate"), "`settings.passRate` is missing"),
    ("report.json", lambda r: r["settings"]["limits"].pop("maxLatencyMs"), "`settings.limits.maxLatencyMs` is missing"),
    ("report.json", lambda r: r["settings"].update(passRate="high"), "`settings.passRate` must be a number"),
    ("report.json", lambda r: r.update(chosen=None), "`chosen.firstLoad.bytes` is missing"),
])
def test_a_manifest_or_a_report_this_cifarium_cannot_judge_by_is_refused_and_says_why(
        built, name, change, expected):
    edit(built, name, change)

    error = refused(built)

    assert expected in str(error), str(error)
    assert error.code == exitcodes.BAD_INPUT


@pytest.mark.parametrize("name", ["cifarium.json", "report.json"])
def test_a_file_that_is_not_json_is_refused(built, name):
    (built / name).write_text("{not json", encoding="utf-8")

    assert "not valid JSON" in str(refused(built))


def test_a_report_that_says_the_last_build_chose_nothing_is_not_the_report_of_the_folder(built):
    edit(built, "report.json", lambda r: r.update(outcome="no-model", exitCode=1, chosen=None))

    message = str(refused(built))

    assert "outcome 'no-model'" in message and "Build again" in message


def test_a_manifest_with_a_bom_is_read(built):
    path = built / bundle.MANIFEST
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())

    assert [label.name for label in read(built).labels] == LABELS


# --- the four judgements -----------------------------------------------------------------------------


def folder_of(built: Path, **changes) -> Folder:
    return replace(read(built), **changes)


def measures(folder: Folder, **changes) -> Measures:
    made = Measures(load_ms=1500.0, first_ms=30.0, median_ms=10.0, p95_ms=15.0, calls=19, threads=4,
                    page_isolated=True, worker_isolated=True, cores=8, bytes=folder.first_load_bytes,
                    variant="cpu", device="wasm", dtype="q8")
    return replace(made, **changes)


def finding(folder: Folder, name: str, *, labels=None, items=None, hosting_file="_headers", **changes):
    items = items or config_items()
    labels = labels if labels is not None else [item.expected for item in items]
    found = check.judge(folder, items, labels, measures(folder, **changes), hosting_file=hosting_file)
    return next(item for item in found if item.name == name)


def config_items(expected=EXPECTED):
    from cifarium.examples import Example

    return [Example(k + 1, f"text {k}", label) for k, label in enumerate(expected)]


@pytest.mark.parametrize("wrong, status", [(0, OK), (3, OK), (4, FAIL)])
def test_the_share_of_right_answers_meets_the_pass_rate_of_the_report_exactly_or_fails(built, wrong, status):
    # 20 examples at 85 %: 17 right is exactly the rate, 16 is under it.
    folder = folder_of(built)
    items = config_items()
    answers = [item.expected for item in items]
    for k in range(wrong):
        answers[k] = "beta" if answers[k] == "alpha" else "alpha"

    result = finding(folder, "examples", items=items, labels=answers)

    assert result.status == status
    assert f"{20 - wrong} of 20 examples right" in result.text and "pass rate 85%" in result.text


def test_a_page_that_is_not_isolated_fails_and_names_the_headers_the_file_must_set(built):
    result = finding(folder_of(built), "isolation", page_isolated=False, worker_isolated=False, threads=1)

    assert result.status == FAIL
    for word in ("hosting/_headers", "not cross-origin isolated", "Cross-Origin-Opener-Policy",
                 "Cross-Origin-Embedder-Policy", "/cifarium-worker.js", "/vendor/*"):
        assert word in result.text, word


def test_a_worker_that_is_not_isolated_fails_on_its_own(built):
    result = finding(folder_of(built), "isolation", worker_isolated=False, threads=1)

    assert result.status == FAIL and "its worker, cifarium-worker.js, is not" in result.text


@pytest.mark.parametrize("cores, threads, status", [
    (12, 4, OK), (8, 4, OK), (4, 2, OK), (3, 2, OK), (2, 1, OK), (1, 1, OK),
    (8, 1, FAIL), (4, 1, FAIL), (3, 1, FAIL),
])
def test_an_isolated_page_must_get_the_threads_onnx_runtime_takes_for_its_cores(built, cores, threads, status):
    # Half the cores, at most four (ARCHITECTURE.md, section 1.6): a machine of two cores computes in
    # one thread even behind the headers, and that is not the hosting file's fault.
    result = finding(folder_of(built), "isolation", cores=cores, threads=threads)

    assert result.status == status, result.text


@pytest.mark.parametrize("median, threads, status", [
    (10.0, 4, OK), (50.0, 4, OK),
    (50.1, 4, WARN), (100.0, 4, WARN),
    (100.1, 4, FAIL), (400.0, 2, FAIL),
    (400.0, 1, WARN),
])
def test_the_usual_answer_against_the_limit_has_an_allowance_for_noise_and_one_thread_only_warns(
        built, median, threads, status):
    result = finding(folder_of(built), "speed", median_ms=median, threads=threads, cores=8)

    assert result.status == status, result.text


def test_without_a_limit_the_speed_is_only_told(built):
    result = finding(folder_of(built, max_latency_ms=None), "speed", median_ms=5000.0)

    assert result.status == INFO and "5000.0 ms" in result.text and "no `limits.max_latency_ms`" in result.text


def test_the_weight_of_the_first_load_is_the_number_of_the_report_or_the_check_fails(built):
    folder = folder_of(built)

    assert finding(folder, "first load").status == OK
    different = finding(folder, "first load", bytes=folder.first_load_bytes + 1)
    assert different.status == FAIL and "not of the same build" in different.text


def test_a_page_that_took_another_variant_than_the_report_counts_does_not_fail_the_weight(built):
    result = finding(folder_of(built), "first load", variant="gpu", bytes=1)

    assert result.status == INFO and "variant gpu" in result.text


def test_the_page_result_becomes_the_measures_the_judgement_reads():
    answers = [{"label": "a", "score": 0.5, "ms": ms} for ms in [30.0] + [float(k) for k in range(1, 21)]]
    result = {"info": {"threads": 2, "crossOriginIsolated": True, "bytes": 123, "variant": "cpu",
                       "device": "wasm", "dtype": "q8"},
              "loadMs": 2500.0, "answers": answers, "isolated": True, "cores": 4}

    made = check.measure(result)

    assert made.first_ms == 30.0 and made.calls == 20
    assert made.median_ms == 10.5 and made.p95_ms == 19.0, "the first call is the warm-up and is left out"
    assert (made.threads, made.cores, made.bytes, made.expected_threads) == (2, 4, 123, 2)
    assert made.page_isolated and made.worker_isolated


# --- the run: what is served, what is printed, the exit codes ----------------------------------------


class FakeChrome:
    headless = True
    device = {"userAgent": "Mozilla/5.0 Chrome/157.0.0.0 Safari/537.36", "webgpu": None}
    context = None


def page_result(labels=None, *, ms=10.0, threads=4, isolated=True, worker=True, cores=8, bytes_=None,
                folder: Folder | None = None) -> dict:
    labels = labels if labels is not None else EXPECTED
    answers = [{"label": label, "score": 0.9, "ms": 3 * ms if k == 0 else ms} for k, label in enumerate(labels)]
    info = {"threads": threads, "crossOriginIsolated": worker, "variant": "cpu", "device": "wasm", "dtype": "q8",
            "bytes": bytes_ if bytes_ is not None else (folder.first_load_bytes if folder else 0)}
    return {"result": {"info": info, "loadMs": 1500.0, "answers": answers, "isolated": isolated, "cores": cores},
            "events": []}


@pytest.fixture()
def chrome(monkeypatch, built):
    """Fake Chrome and a fake page. `drive` is given what the command would give the real one; the
    outcome it returns is `chrome.outcome`, and what it was asked is in `chrome.calls`."""
    fake = FakeChrome()
    fake.calls = []
    fake.outcome = page_result(folder=read(built))
    fake.opened = []

    @contextmanager
    def fake_open_bench(*, headed: bool = False):
        fake.opened.append(headed)
        yield fake

    def fake_drive(bench, url, texts, **kwargs):
        fake.calls.append((url, list(texts)))
        fake.fetched = {path: fetch(url, path) for path in ("/", "/index.html", "/cifarium.js", "/cifarium.json",
                                                              "/cifarium-worker.js", "/vendor/x/lib.mjs", "/missing")}
        if isinstance(fake.outcome, Exception):
            raise fake.outcome
        return fake.outcome

    monkeypatch.setattr(check, "open_bench", fake_open_bench)
    monkeypatch.setattr(check, "drive", fake_drive)
    return fake


def fetch(url: str, path: str) -> tuple[int, dict[str, str], bytes]:
    host = url.removeprefix("http://")
    connection = http.client.HTTPConnection(host, timeout=5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, {name.lower(): value for name, value in response.getheaders()}, response.read()
    finally:
        connection.close()


def tree_hash(folder: Path) -> dict[str, str]:
    return {file.relative_to(folder).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(folder.rglob("*")) if file.is_file()}


def run(built, examples, capsys, **options):
    code = check.run(built, examples=examples, **options)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_a_folder_that_answers_right_in_isolated_threads_passes_and_the_lines_say_so(built, examples, chrome, capsys):
    code, out, err = run(built, examples, capsys)

    assert code == exitcodes.DONE and err == ""
    assert chrome.calls[0][1][:2] == ["text 0 of alpha", "text 1 of beta"] and len(chrome.calls[0][1]) == 20
    for line in ("examples  20 in held-out.jsonl, 2 labels", "report.json: pass rate 85%, usual answer up to 50 ms",
                 "hosting/_headers, read the way netlify reads it", "Chrome 157 without a window",
                 "ok    examples    20 of 20 examples right (100%), pass rate 85%",
                 "ok    isolation   the page and the worker are cross-origin isolated, 4 threads on 8 cores",
                 "ok    speed       usual answer 10.0 ms, within `limits.max_latency_ms: 50`",
                 "ok    first load", "PASSED: the folder works as a site will see it."):
        assert line in out, line
    assert chrome.opened == [False]


def test_the_page_of_the_check_stands_at_the_root_and_the_folders_files_stay_where_they_are(built, examples, chrome, capsys):
    run(built, examples, capsys)

    status, headers, body = chrome.fetched["/"]
    assert status == 200 and b"window.cifariumCheck" in body and b"the page of the site" not in body
    assert chrome.fetched["/index.html"][2] == body
    assert chrome.fetched["/cifarium.js"][2] == b"export {};"
    assert chrome.fetched["/missing"][0] == 404


def test_the_headers_are_those_of_the_hosting_file_and_of_no_other_rule(built, examples, chrome, capsys):
    run(built, examples, capsys)

    for path in ("/", "/index.html", "/cifarium-worker.js", "/vendor/x/lib.mjs"):
        headers = chrome.fetched[path][1]
        assert headers["cross-origin-opener-policy"] == "same-origin", path
        assert headers["cross-origin-embedder-policy"] == "require-corp", path
    assert "cross-origin-opener-policy" not in chrome.fetched["/cifarium.json"][1]
    assert "no-store" not in chrome.fetched["/vendor/x/lib.mjs"][1].get("cache-control", "")


@pytest.mark.parametrize("name, file", sorted(hosting.FILES.items()))
def test_every_hosting_file_can_be_the_one_the_folder_is_served_by(built, examples, chrome, capsys, name, file):
    code, out, _ = run(built, examples, capsys, hosting_name=name)

    assert code == exitcodes.DONE
    assert f"hosting/{file}, read the way {name} reads it" in out
    assert chrome.fetched["/"][1]["cross-origin-opener-policy"] == "same-origin"


def test_a_hosting_file_with_the_rules_cut_out_fails_the_isolation_not_the_command(built, examples, chrome, capsys):
    # What the page reports when the file does not isolate it is the fake's to say; here the server
    # shows that the file's rules, and no others, decide the headers.
    (built / "hosting" / "_headers").write_text("/index.html\n  X-Only: this\n", encoding="utf-8")

    run(built, examples, capsys)

    assert "cross-origin-opener-policy" not in chrome.fetched["/"][1]
    assert chrome.fetched["/index.html"][1]["x-only"] == "this"


def test_check_writes_nothing_into_the_folder_nor_leaves_a_file_beside_it(built, examples, chrome, capsys):
    before = tree_hash(built)
    around = sorted(path.name for path in built.parent.iterdir())

    run(built, examples, capsys)

    assert tree_hash(built) == before
    assert sorted(path.name for path in built.parent.iterdir()) == around


def test_a_share_under_the_pass_rate_fails_the_check_and_lists_the_misses(built, examples, chrome, capsys):
    labels = list(EXPECTED)
    for k in range(5):
        labels[k] = "beta" if labels[k] == "alpha" else "alpha"
    chrome.outcome = page_result(labels, folder=read(built))

    code, out, err = run(built, examples, capsys)

    assert code == exitcodes.NOT_PASSED == 1
    assert "FAIL  examples    15 of 20 examples right (75%), pass rate 85%" in out
    assert "miss, line 1: beta instead of alpha: text 0 of alpha" in out
    assert "FAILED: the folder does not pass: examples." in err


def test_a_long_list_of_misses_is_cut_and_says_how_many_more(built, tmp_path, chrome, capsys):
    path = tmp_path / "many.jsonl"
    path.write_text(example_lines(EXPECTED * 3), encoding="utf-8")
    chrome.outcome = page_result(["beta" if label == "alpha" else "alpha" for label in EXPECTED * 3],
                                 folder=read(built))

    _, out, _ = run(built, path, capsys)

    assert out.count("miss, line") == check.MAX_MISSES_SHOWN
    assert f"... and {60 - check.MAX_MISSES_SHOWN} more" in out


def test_every_failing_check_is_named_in_the_last_line(built, examples, chrome, capsys):
    folder = read(built)
    chrome.outcome = page_result(["beta"] * 20, isolated=False, worker=False, threads=1, ms=500.0,
                                 bytes_=folder.first_load_bytes + 1)

    code, out, err = run(built, examples, capsys)

    assert code == 1
    assert "FAILED: the folder does not pass: examples, isolation, first load." in err
    assert "warn  speed" in out, "one thread only warns about the speed"


def test_a_speed_far_over_the_limit_with_threads_fails_but_a_noisy_one_only_warns(built, examples, chrome, capsys):
    folder = read(built)
    chrome.outcome = page_result(ms=70.0, folder=folder)
    code, out, _ = run(built, examples, capsys)
    assert code == 0 and "warn  speed" in out and "within 2x of it" in out

    chrome.outcome = page_result(ms=120.0, folder=folder)
    code, out, err = run(built, examples, capsys)
    assert code == 1 and "FAIL  speed" in out and "FAILED: the folder does not pass: speed." in err


def test_a_folder_whose_runtime_fails_in_the_page_is_not_passed_and_says_where(built, examples, chrome, capsys):
    chrome.outcome = {"failure": {"stage": "load", "index": None, "code": "load", "message": "no such file"},
                      "events": ["console error: Failed to load resource: 404"]}

    code, out, err = run(built, examples, capsys)

    assert code == exitcodes.NOT_PASSED
    assert "the page could not load the model [load]: no such file" in err
    assert "console error: Failed to load resource: 404" in err and "PASSED" not in out


@pytest.mark.parametrize("failure, words", [
    ({"stage": "import", "index": None, "code": None, "message": "x"}, "load cifarium.js as a module"),
    ({"stage": "create", "index": None, "code": "manifest", "message": "x"}, "read cifarium.json [manifest]"),
    ({"stage": "run", "index": 2, "code": "run", "message": "x"}, "answer the example on line 3"),
    ({"stage": "time", "index": None, "code": None, "message": "the page did not finish in 300 s"}, "finish"),
])
def test_the_stage_a_page_failed_at_is_in_words(built, examples, chrome, capsys, failure, words):
    chrome.outcome = {"failure": failure, "events": []}

    _, _, err = run(built, examples, capsys)

    assert words in err


def test_a_browser_that_does_not_start_is_the_environment_not_the_folder(built, examples, monkeypatch, capsys):
    @contextmanager
    def no_chrome(*, headed: bool = False):
        raise BenchError("Could not start Google Chrome: no such file")
        yield

    monkeypatch.setattr(check, "open_bench", no_chrome)

    code, _, err = run(built, examples, capsys)

    assert code == exitcodes.ENVIRONMENT and "FAILED: Could not start Google Chrome" in err


def test_a_page_the_browser_could_not_run_is_the_environment_too(built, examples, chrome, capsys):
    chrome.outcome = BenchError("The check page failed: net::ERR_CONNECTION_REFUSED")

    code, _, err = run(built, examples, capsys)

    assert code == exitcodes.ENVIRONMENT and "The check page failed" in err


def test_headed_opens_a_window(built, examples, chrome, capsys):
    run(built, examples, capsys, headed=True)

    assert chrome.opened == [True]


def test_nothing_is_started_when_the_input_is_wrong(built, examples, monkeypatch, capsys):
    def started(*args, **kwargs):
        raise AssertionError("a browser was started for input that is wrong")

    monkeypatch.setattr(check, "open_bench", started)
    (built / bundle.MANIFEST).unlink()

    code, _, err = run(built, examples, capsys)

    assert code == exitcodes.BAD_INPUT and "cifarium.json: file not found" in err


def test_examples_that_do_not_fit_the_labels_of_the_folder_are_refused_with_their_lines(built, tmp_path, chrome, capsys):
    path = tmp_path / "other.jsonl"
    path.write_text(example_lines(["alpha", "gamma"] * 5), encoding="utf-8")

    code, _, err = run(built, path, capsys)

    assert code == exitcodes.BAD_INPUT
    assert "label 'gamma' is not listed under labels in cifarium.json" in err.replace('"', "'")
    assert chrome.calls == []


def test_examples_that_are_not_there_or_have_a_label_of_the_folder_missing_are_refused(built, tmp_path, chrome, capsys):
    assert run(built, tmp_path / "nope.jsonl", capsys)[0] == exitcodes.BAD_INPUT
    only = tmp_path / "only.jsonl"
    only.write_text(example_lines(["alpha"] * 4), encoding="utf-8")

    code, _, err = run(built, only, capsys)

    assert code == exitcodes.BAD_INPUT and "label 'beta' from cifarium.json has no examples" in err.replace('"', "'")


def test_a_hosting_file_that_is_not_in_the_folder_is_refused_before_a_browser_starts(built, examples, chrome, capsys):
    (built / "hosting" / "vercel.json").unlink()

    code, _, err = run(built, examples, capsys, hosting_name="vercel")

    assert code == exitcodes.BAD_INPUT
    assert "hosting/vercel.json is not in" in err and "netlify, nginx, vercel" in err
    assert chrome.calls == []


def test_without_the_folder_and_the_examples_the_settings_name_them(built, tmp_path, chrome, capsys, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "examples.jsonl").write_text(example_lines(), encoding="utf-8")
    (project / "cifarium.yaml").write_text("task: classify\nexamples: examples.jsonl\noutput: ../out\n", encoding="utf-8")
    monkeypatch.chdir(project)

    code = check.run()
    out = capsys.readouterr().out

    assert code == 0
    assert "20 in examples.jsonl, 2 labels (the examples of the build: it has seen them" in out
    assert chrome.calls[0][1][0] == "text 0 of alpha"


def test_examples_of_another_file_are_not_called_the_examples_of_the_build(built, examples, chrome, capsys):
    _, out, _ = run(built, examples, capsys)

    assert "the examples of the build" not in out


def test_without_settings_the_folder_and_the_examples_must_be_named(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    code = check.run()
    err = capsys.readouterr().err

    assert code == exitcodes.BAD_INPUT
    assert "cifarium.yaml: file not found" in err and "`cifarium check <folder> --examples <file>`" in err


def test_a_folder_named_but_examples_not_still_reads_them_from_the_settings(built, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    code = check.run(built)

    assert code == exitcodes.BAD_INPUT and "cifarium.yaml: file not found" in capsys.readouterr().err


def test_the_command_does_not_fall_over_on_labels_in_any_script(built, tmp_path, chrome, capsys):
    # The examples of another script do not reach a narrow console as an error.
    edit(built, "cifarium.json", lambda m: m.update(labels=[{"name": "почта", "description": None},
                                                            {"name": "будильник", "description": None}]))
    path = tmp_path / "ru.jsonl"
    path.write_text("".join(json.dumps({"input": f"текст {k}", "expected": label}, ensure_ascii=False) + "\n"
                            for k, label in enumerate(["почта", "будильник"] * 3)), encoding="utf-8")
    chrome.outcome = page_result(["почта", "будильник"] * 3, folder=read(built))

    code, out, _ = run(built, path, capsys)

    assert code == 0 and "6 examples" in out


# --- the page and the command line -------------------------------------------------------------------


def test_the_page_of_the_check_talks_to_the_command_through_one_object_and_sends_nothing():
    page = check.PAGE.read_text(encoding="utf-8")

    assert "window.cifariumCheck = {" in page and "start(job)" in page and "outcome" in page
    assert "createCifarium('./cifarium.json')" in page and "import('./cifarium.js')" in page
    assert not re.search(r"https?://", re.sub(r"<!--.*?-->", "", page, flags=re.S))
    for word in ("sendBeacon", "XMLHttpRequest", "WebSocket", "EventSource", "fetch("):
        assert word not in page, word


def test_the_command_line_of_check_takes_a_folder_the_examples_the_hosting_and_a_window():
    parser = cli.build_parser()

    plain = parser.parse_args(["check"])
    assert (plain.folder, plain.examples, plain.hosting, plain.headed) == (None, None, "netlify", False)
    full = parser.parse_args(["check", "site/ml", "--examples", "held.jsonl", "--hosting", "nginx", "--headed"])
    assert (full.folder, full.examples, full.hosting, full.headed) == ("site/ml", "held.jsonl", "nginx", True)
    with pytest.raises(SystemExit) as refused_:
        parser.parse_args(["check", "--hosting", "apache"])
    assert refused_.value.code == exitcodes.BAD_INPUT


def test_the_command_hands_its_arguments_to_check(monkeypatch):
    seen = {}

    def fake_run(folder, **options):
        seen.update(folder=folder, **options)
        return 7

    monkeypatch.setattr(check, "run", fake_run)

    assert cli.main(["check", "site", "--examples", "e.jsonl", "--hosting", "vercel", "--headed"]) == 7
    assert seen == {"folder": "site", "examples": "e.jsonl", "hosting_name": "vercel", "headed": True}


def test_the_help_of_check_says_what_it_does_not_write_and_names_the_exit_codes(capsys):
    with pytest.raises(SystemExit):
        cli.main(["check", "--help"])
    text = " ".join(capsys.readouterr().out.split())

    assert "Writes nothing, the folder included" in text
    for code in ("0 the folder passed", "1 it did not pass", "2 the folder, its report", "3 the environment failed"):
        assert code in text, code


def test_an_unexpected_failure_of_check_is_the_environment_and_not_a_refusal(monkeypatch, capsys):
    def surprise(*args, **kwargs):
        raise RuntimeError("a bug in cifarium")

    monkeypatch.setattr(check, "run", surprise)

    assert cli.main(["check"]) == exitcodes.ENVIRONMENT
    assert "a bug in cifarium" in capsys.readouterr().err


def test_config_still_names_the_default_settings_file_check_looks_for():
    assert config.DEFAULT_FILE == "cifarium.yaml"

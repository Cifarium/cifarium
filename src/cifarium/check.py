"""The ``check`` command: does a build folder work the way a site will see it?

The folder is served with the headers one of its hosting files names (``server.SiteServer``,
``hosting``), a page stands where a site's page stands, imports ``cifarium.js``, loads the model and
answers the examples with ``run`` (``web/check/index.html``), and four things are judged against what
the build wrote in ``report.json``: the share of right answers against ``settings.passRate``, the
isolation and the threads, the usual answer against ``settings.limits.maxLatencyMs`` with an
allowance for timing noise, and ``info().bytes`` against ``chosen.firstLoad.bytes``. Nothing is
written: not into the folder, which is published, and not anywhere else. ARCHITECTURE.md, section 9
(``check``), has the choices; the exit codes are those of :mod:`exitcodes`.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from . import bundle, config, exitcodes, hosting, paths
from . import examples as example_files
from . import report as reporting
from .bench import Bench, BenchError, open_bench
from .build import describe_browser
from .server import SiteServer

PAGE = paths.web_dir() / "check" / "index.html"
"""The page of the check, served in place of the folder's page."""

TIME_LIMIT_S = 300.0
"""Time limit of the page: the first load of the model and every example."""

READY_TIMEOUT_MS = 30_000

LATENCY_TOLERANCE = 2.0
"""A usual answer up to this many times ``maxLatencyMs`` is a warning, not a failure: the time of a
laptop between two sessions varies about twofold (ARCHITECTURE.md, section 9)."""

MAX_MISSES_SHOWN = 10

OK, WARN, FAIL, INFO = "ok", "warn", "FAIL", "info"
"""What a line of the result says. Only :data:`FAIL` fails the check."""

MISS_TEXT_CHARS = 60


class CheckError(Exception):
    """Something the check cannot start from. ``code`` is the exit code it ends with."""

    def __init__(self, message: str, code: int = exitcodes.BAD_INPUT) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Folder:
    """What the check reads from a build folder: the labels of the manifest and the numbers of the
    report the folder was promised under."""

    path: Path
    model: str
    recipe: str
    built_by: str
    labels: tuple[config.Label, ...]
    pass_rate: float
    max_latency_ms: float | None
    first_load_bytes: int
    variant: str


@dataclass(frozen=True)
class Measures:
    """What the page measured, in the terms the judgement needs."""

    load_ms: float
    first_ms: float
    median_ms: float
    p95_ms: float
    calls: int
    threads: int
    page_isolated: bool
    worker_isolated: bool
    cores: int
    bytes: int
    variant: str
    device: str
    dtype: str

    @property
    def expected_threads(self) -> int:
        """What ONNX Runtime takes on an isolated page: half the cores, at most four."""
        return min(4, math.ceil(self.cores / 2))


@dataclass(frozen=True)
class Finding:
    name: str
    status: str
    text: str


# --- reading the folder, the report and the examples ------------------------------------------------


def read_folder(folder: Path) -> Folder:
    """Read ``cifarium.json`` and ``report.json`` of a build folder. Raises :class:`CheckError`."""
    if not folder.is_dir():
        raise CheckError(f"{folder}: no such folder. `cifarium build` writes it; "
                         "`cifarium check <folder>` checks another one.")
    manifest = _read_json(folder / bundle.MANIFEST)
    _check_format(folder / bundle.MANIFEST, manifest, bundle.FORMAT, "manifest")
    if manifest.get("task") != "classify":
        raise CheckError(f"{folder / bundle.MANIFEST}: task {manifest.get('task')!r}; this check handles "
                         "folders of the task `classify`")
    labels = _labels_of(folder / bundle.MANIFEST, manifest)
    report_path = folder / bundle.REPORT_JSON
    report = _read_json(report_path)
    _check_format(report_path, report, reporting.FORMAT, "report")
    if report.get("outcome") != reporting.BUILT:
        raise CheckError(
            f"{report_path}: outcome {report.get('outcome')!r}. The last build chose no model, so the "
            f"numbers here are not those of the folder, which is from an earlier build. Build again.")
    model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    return Folder(
        path=folder,
        model=str(model.get("repo", "?")),
        recipe=str(manifest.get("recipe", "?")),
        built_by=str(manifest.get("cifarium", "?")),
        labels=labels,
        pass_rate=float(_number(report_path, report, "settings", "passRate")),
        max_latency_ms=_number(report_path, report, "settings", "limits", "maxLatencyMs", null=True),
        first_load_bytes=int(_number(report_path, report, "chosen", "firstLoad", "bytes")),
        variant=str(_dig(report_path, report, "chosen", "variant")),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise CheckError(f"{path}: file not found. A folder that `cifarium build` wrote has it; "
                         "build, or name the folder with `cifarium check <folder>`.") from None
    except UnicodeDecodeError:
        raise CheckError(f"{path}: not UTF-8 text") from None
    except OSError as error:
        raise CheckError(f"{path}: cannot read the file: {error.strerror or error}",
                         exitcodes.ENVIRONMENT) from None
    try:
        data = json.loads(text)
    except ValueError as error:
        raise CheckError(f"{path}: not valid JSON ({error})") from None
    if not isinstance(data, dict):
        raise CheckError(f"{path}: must be a JSON object")
    return data


def _check_format(path: Path, data: Mapping[str, Any], known: int, what: str) -> None:
    number = data.get("format")
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise CheckError(f"{path}: no `format` number: not a {what} that cifarium wrote")
    if number > known:
        raise CheckError(f"{path}: {what} format {number}, newer than the {known} this cifarium reads. "
                         "The folder was built by a newer cifarium: install that one to check it.")


def _labels_of(path: Path, manifest: Mapping[str, Any]) -> tuple[config.Label, ...]:
    raw = manifest.get("labels")
    if (not isinstance(raw, list) or len(raw) < 2
            or not all(isinstance(item, dict) and isinstance(item.get("name"), str) for item in raw)):
        raise CheckError(f"{path}: `labels` is not a list of two or more `{{name, description}}`")
    return tuple(config.Label(item["name"], item.get("description")) for item in raw)


def _dig(path: Path, data: Mapping[str, Any], *keys: str, null: bool = False) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise CheckError(f"{path}: `{'.'.join(keys)}` is missing: not a report this cifarium wrote")
        value = value[key]
    if value is None and not null:
        raise CheckError(f"{path}: `{'.'.join(keys)}` is null")
    return value


def _number(path: Path, data: Mapping[str, Any], *keys: str, null: bool = False) -> float | None:
    value = _dig(path, data, *keys, null=null)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckError(f"{path}: `{'.'.join(keys)}` must be a number")
    return value


def _settings() -> config.Config:
    """The settings the folder and the examples come from when the command line names no others."""
    try:
        return config.load()
    except config.ConfigError as error:
        raise CheckError(
            f"{error}\ncheck takes the folder and the examples from {config.DEFAULT_FILE} in the "
            "current folder. Without it, name both: `cifarium check <folder> --examples <file>`.") from None


# --- measures and judgement -------------------------------------------------------------------------


def measure(result: Mapping[str, Any]) -> Measures:
    """Read the page's result (``CheckResult`` in ``web/check/index.html``)."""
    info = result["info"]
    times = [float(answer["ms"]) for answer in result["answers"]]
    usual = times[1:] or times
    ranked = sorted(usual)
    return Measures(
        load_ms=float(result["loadMs"]),
        first_ms=times[0],
        median_ms=float(statistics.median(usual)),
        p95_ms=ranked[math.ceil(0.95 * len(ranked)) - 1],
        calls=len(usual),
        threads=int(info.get("threads") or 1),
        page_isolated=bool(result["isolated"]),
        worker_isolated=bool(info.get("crossOriginIsolated")),
        cores=int(result.get("cores") or 1),
        bytes=int(info.get("bytes") or 0),
        variant=str(info.get("variant")),
        device=str(info.get("device")),
        dtype=str(info.get("dtype")),
    )


def judge(
    folder: Folder,
    items: Sequence[example_files.Example],
    labels: Sequence[str],
    measures: Measures,
    *,
    hosting_file: str,
) -> list[Finding]:
    """The four findings. ``labels`` are the page's answers, one for each of ``items``."""
    return [
        _judge_examples(folder, items, labels),
        _judge_isolation(measures, hosting_file),
        _judge_speed(folder, measures),
        _judge_first_load(folder, measures),
    ]


def hits(items: Sequence[example_files.Example], labels: Sequence[str]) -> int:
    return sum(1 for item, label in zip(items, labels) if item.expected == label)


def _judge_examples(folder: Folder, items: Sequence[example_files.Example], labels: Sequence[str]) -> Finding:
    correct, total = hits(items, labels), len(items)
    share = correct / total
    text = f"{correct} of {total} examples right ({_percent(share)}), pass rate {_percent(folder.pass_rate)}"
    if share + 1e-9 >= folder.pass_rate:
        return Finding("examples", OK, text)
    return Finding("examples", FAIL, text + ": the folder does not reach the pass rate it was built for")


def _judge_isolation(measures: Measures, hosting_file: str) -> Finding:
    where = f"{hosting.FOLDER}/{hosting_file}"
    rule = ("The file must set Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy: "
            "require-corp on `/`, `/index.html`, `/cifarium-worker.js` and `/vendor/*` "
            "(ARCHITECTURE.md, section 1.6).")
    if not measures.page_isolated:
        return Finding("isolation", FAIL, f"behind the headers of {where} the page is not cross-origin "
                       f"isolated, so the model computes in one thread. {rule}")
    if not measures.worker_isolated:
        return Finding("isolation", FAIL, f"the page is isolated but its worker, cifarium-worker.js, is "
                       f"not, so the model computes in one thread. {rule}")
    detail = f"{_plural(measures.threads, 'thread')} on {_plural(measures.cores, 'core')}"
    if measures.threads < measures.expected_threads:
        return Finding("isolation", FAIL, f"isolated, but {detail}: ONNX Runtime takes "
                       f"{measures.expected_threads} on an isolated page with that many cores")
    return Finding("isolation", OK, f"the page and the worker are cross-origin isolated, {detail}")


def _judge_speed(folder: Folder, measures: Measures) -> Finding:
    usual = _ms(measures.median_ms)
    limit = folder.max_latency_ms
    if limit is None:
        return Finding("speed", INFO, f"usual answer {usual}; the build set no `limits.max_latency_ms`")
    if measures.median_ms <= limit:
        return Finding("speed", OK, f"usual answer {usual}, within `limits.max_latency_ms: {limit:g}`")
    over = f"usual answer {usual} is over `limits.max_latency_ms: {limit:g}`"
    if measures.threads < 2:
        return Finding("speed", WARN, f"{over}, in one thread: this page or this machine cannot reproduce "
                       "the number the build measured in several")
    if measures.median_ms <= limit * LATENCY_TOLERANCE:
        return Finding("speed", WARN, f"{over}, within {LATENCY_TOLERANCE:g}x of it: the allowance for the "
                       "timing noise of one machine")
    return Finding("speed", FAIL, f"{over} by more than {LATENCY_TOLERANCE:g}x, with {_plural(measures.threads, 'thread')}")


def _judge_first_load(folder: Folder, measures: Measures) -> Finding:
    if measures.variant != folder.variant:
        return Finding("first load", INFO, f"the page took the variant {measures.variant}, report.json "
                       f"counts {folder.variant}: not compared")
    if measures.bytes == folder.first_load_bytes:
        return Finding("first load", OK, f"info().bytes is {measures.bytes} ({_mb(measures.bytes)}), "
                       "as report.json says")
    return Finding("first load", FAIL, f"info().bytes is {measures.bytes}, report.json says "
                   f"{folder.first_load_bytes}: the folder and its report are not of the same build. Build again.")


# --- the page ---------------------------------------------------------------------------------------


def drive(bench: Bench, url: str, texts: Sequence[str], *, timeout: float = TIME_LIMIT_S) -> dict[str, Any]:
    """Open the check page of the folder at ``url``, run the examples and return what it reports.

    The result is ``{"result": ...}`` when the page answered every text and ``{"failure": ...}``
    when the folder's runtime did not (``CheckResult`` and ``CheckFailure`` in the page), with the
    console and the errors of the page under ``events``. Raises :class:`BenchError` when the browser
    fails, which is not the folder's fault.
    """
    page = bench.context.new_page()
    events: list[str] = []
    page.on("console", lambda message: events.append(f"console {message.type}: {message.text}"))
    page.on("pageerror", lambda error: events.append(f"script error: {error}"))
    page.on("crash", lambda _: events.append("the tab crashed, probably out of memory"))
    try:
        page.goto(url + "/")
        page.wait_for_function("() => window.cifariumCheck !== undefined", timeout=READY_TIMEOUT_MS)
        page.evaluate("job => window.cifariumCheck.start(job)", {"texts": list(texts)})
        page.wait_for_function("() => window.cifariumCheck.outcome !== undefined", polling=100,
                               timeout=timeout * 1000)
        outcome = page.evaluate("() => window.cifariumCheck.outcome")
    except PlaywrightTimeoutError:
        return {"failure": {"stage": "time", "index": None, "code": None,
                            "message": f"the page did not finish in {timeout:g} s"}, "events": events}
    except PlaywrightError as error:
        lines = str(error).strip().splitlines()
        raise BenchError(f"The check page failed: {lines[0] if lines else type(error).__name__}") from None
    finally:
        try:
            page.close()
        except PlaywrightError:
            pass  # already closed, or Chrome is gone
    return {**outcome, "events": events}


# --- the command ------------------------------------------------------------------------------------


def run(
    folder: str | Path | None = None,
    *,
    examples: str | Path | None = None,
    hosting_name: str = hosting.DEFAULT,
    headed: bool = False,
) -> int:
    """Run the command and return the exit code (:mod:`exitcodes`): 0 when the folder passed every
    check; 1 when it ran and something failed; 2 when the folder, its report, the examples or the
    command line are wrong; 3 when the environment failed (Chrome, the disk)."""
    _tolerate_any_text()
    try:
        context = _prepare(folder, examples, hosting_name)
    except CheckError as error:
        print(error, file=sys.stderr)
        return error.code
    except config.TaskFileError as error:  # ExamplesError
        print(error, file=sys.stderr)
        return exitcodes.BAD_INPUT
    except hosting.HostingError as error:
        print(f"{error}. `cifarium check --hosting` takes {', '.join(sorted(hosting.FILES))}.", file=sys.stderr)
        return exitcodes.BAD_INPUT
    folder_info, items, headers, from_settings = context

    _say("cifarium check")
    _say(f"  folder    {_shown(folder_info.path)}: {folder_info.model}, built by cifarium {folder_info.built_by}")
    _say(f"  examples  {len(items.items)} in {items.path.name}, {len(items.labels)} labels"
         + (f" ({same_examples(folder_info.recipe)})" if from_settings else ""))
    limit = folder_info.max_latency_ms
    _say(f"  promised  report.json: pass rate {_percent(folder_info.pass_rate)}, usual answer "
         + ("without a limit" if limit is None else f"up to {limit:g} ms")
         + f", first load {_mb(folder_info.first_load_bytes)}")
    hosting_file = hosting.FILES[hosting_name]
    _say(f"  hosting   {hosting.FOLDER}/{hosting_file}, read the way {hosting_name} reads it")
    texts = [item.input for item in items.items]
    try:
        with open_bench(headed=headed) as bench:
            _say(f"  browser   {describe_browser(bench)}")
            _say(f"\nA page stands where the page of a site stands, loads the model and answers "
                 f"{len(texts)} examples:")
            with SiteServer(folder_info.path, headers, replace={"/": PAGE, "/index.html": PAGE}) as site:
                outcome = drive(bench, site.url, texts)
    except (BenchError, OSError) as error:
        print(f"\nFAILED: {error}", file=sys.stderr)
        return exitcodes.ENVIRONMENT

    failure = outcome.get("failure")
    if failure is not None:
        print(f"\nFAILED: {describe_failure(failure, items.items)}", file=sys.stderr)
        for event in outcome["events"][-5:]:
            print(f"  {event}", file=sys.stderr)
        return exitcodes.NOT_PASSED
    measures = measure(outcome["result"])
    labels = [answer["label"] for answer in outcome["result"]["answers"]]
    for line in describe_measures(measures):
        _say(f"  {line}")
    findings = judge(folder_info, items.items, labels, measures, hosting_file=hosting_file)
    _say("\nChecks:")
    for finding in findings:
        _say(f"  {finding.status:<5} {finding.name:<11} {finding.text}")
    for line in describe_misses(items.items, labels):
        _say(f"  {line}")
    failed = [finding.name for finding in findings if finding.status == FAIL]
    if failed:
        print(f"\nFAILED: the folder does not pass: {', '.join(failed)}.", file=sys.stderr)
        return exitcodes.NOT_PASSED
    _say("\nPASSED: the folder works as a site will see it.")
    return exitcodes.DONE


def _prepare(
    folder: str | Path | None, examples: str | Path | None, hosting_name: str
) -> tuple[Folder, example_files.Examples, hosting.HeadersOf, bool]:
    """Everything the command reads before it starts a browser: the folder, the examples, the
    headers of the hosting file, and whether the examples are those of the settings (the build has
    seen them). Nothing here needs Chrome."""
    settings = _settings() if folder is None or examples is None else None
    folder_path = Path(folder) if folder is not None else settings.output
    info = read_folder(folder_path)
    examples_file = Path(examples) if examples is not None else settings.examples
    items = example_files.read(examples_file, info.labels, settings_name=bundle.MANIFEST)
    headers = hosting.headers_of(info.path, hosting_name)
    return info, items, headers, examples is None


def same_examples(recipe: str) -> str:
    """What the examples of the build can say about the folder, which depends on the recipe.

    ``embed-centroid``: the centres were made from these examples, so the share says the runtime
    agrees with the build, not how the model does on new texts. ``zero-shot``: the model never saw
    them, so the share is the one the build counted on the same texts with the same sentences, and a
    different one means this page asks the model something else than the build did; for new texts
    name a file with ``--examples``."""
    if recipe == "zero-shot":
        return ("the examples of the build: the model has not learned from them, so the share is the one "
                "the build counted, and a different one means the page asks the model something else; "
                "it is not how the model does on new texts")
    return ("the examples of the build: it has seen them, so the share says the runtime agrees with it, "
            "not how the model does on new texts")


def describe_failure(failure: Mapping[str, Any], items: Sequence[example_files.Example]) -> str:
    """The sentence for a page that did not get to the end."""
    stage = failure.get("stage")
    doing = {"import": "load cifarium.js as a module", "create": "read cifarium.json",
             "load": "load the model"}.get(stage)
    if stage == "run":
        index = failure.get("index")
        doing = (f"answer the example on line {items[index].line}" if isinstance(index, int)
                 and 0 <= index < len(items) else "answer an example")
    elif stage == "time" or doing is None:
        doing = "finish"
    code = f" [{failure['code']}]" if failure.get("code") else ""
    return (f"the page could not {doing}{code}: {failure.get('message')}. A site with this folder would fail "
            "the same way.")


def describe_measures(measures: Measures) -> list[str]:
    """What the page measured, for the console; none of it is judged here."""
    return [
        f"loaded    {measures.load_ms / 1000:.2f} s from an empty browser, on {measures.device}, "
        f"{measures.dtype}, variant {measures.variant}",
        f"answered  usual answer {_ms(measures.median_ms)} (median of {measures.calls}), "
        f"95th percentile {_ms(measures.p95_ms)}, first call {_ms(measures.first_ms)}",
    ]


def describe_misses(items: Sequence[example_files.Example], labels: Sequence[str]) -> list[str]:
    """One line per wrong answer: the line of the examples file, the label given and the expected one."""
    missed = [(item, label) for item, label in zip(items, labels) if item.expected != label]
    lines = [f"miss, line {item.line}: {label} instead of {item.expected}: {item.input[:MISS_TEXT_CHARS]}"
             for item, label in missed[:MAX_MISSES_SHOWN]]
    if len(missed) > MAX_MISSES_SHOWN:
        lines.append(f"... and {len(missed) - MAX_MISSES_SHOWN} more")
    return lines


# --- small helpers ----------------------------------------------------------------------------------


def _shown(folder: Path) -> str:
    """The folder as the user can type it: relative to the current folder when it lies inside."""
    path = os.path.abspath(folder)
    try:
        inside = os.path.commonpath([path, os.getcwd()]) == os.getcwd()
    except ValueError:  # another drive on Windows
        inside = False
    return os.path.relpath(path) if inside and path != os.getcwd() else path


def _tolerate_any_text() -> None:
    """Labels and examples may hold any script; a console in a narrower code page must not end the
    check with UnicodeEncodeError. What it cannot show becomes ``?``."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def _say(message: str = "") -> None:
    print(message, flush=True)


def _mb(size: int | float) -> str:
    return f"{size / 1_000_000:.1f} MB"


def _ms(value: float) -> str:
    return f"{value:.1f} ms"


def _percent(share: float) -> str:
    return f"{share * 100:.1f}".removesuffix(".0") + "%"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"

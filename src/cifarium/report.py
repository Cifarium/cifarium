"""The report of a build: ``report.json`` for programs and ``report.md`` for people.

:func:`of_build` makes one dictionary that says everything; the JSON file is that dictionary and
the Markdown file is printed from it, so the two cannot disagree. The keys of the JSON are a public
contract (ARCHITECTURE.md, section 14, and ``tests/test_contracts.py``); the Markdown is not.

The report lies in the folder that is published, so it holds no text of an example: a miss is the
number of its line in the examples file and the two labels, never the words. What the machine tells
about itself is cut to what a reader needs: the version of Chrome and the kind of the adapter, no
paths, no names, no memory sizes. ARCHITECTURE.md, section 9 ("The report"), has the choices.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cifarium

from . import bundle, config, examples, exitcodes, paths

if TYPE_CHECKING:  # build imports this module: the types are only named here
    from .build import Result, Row

FORMAT = 1
"""Format of ``report.json``. Grows like the format of the manifest: only when a key is removed,
renamed or given a new meaning (ARCHITECTURE.md, section 14)."""

BUILT = "built"
"""``outcome`` of a build that chose a model and wrote the folder (exit code 0)."""

NO_MODEL = "no-model"
"""``outcome`` of a build that ran and chose nothing (exit code 1)."""

SHARE_DIGITS = 4
SCORE_DIGITS = 4
MS_DIGITS = 2


def device_of(info: Mapping[str, Any], headless: bool) -> dict[str, Any]:
    """What the report says of the browser and its adapter, out of the bench's report on the
    device (``bench.device``). Chrome by its major version and WebGPU by vendor and architecture;
    the user agent, the description of the card and the memory stay out of a published file."""
    version = re.search(r"Chrome/(\d+)", str(info.get("userAgent") or ""))
    gpu = info.get("webgpu")
    return {
        "browser": {"name": "Chrome", "version": int(version.group(1)) if version else None,
                    "headless": bool(headless)},
        "webgpu": None if not gpu else {
            "vendor": str(gpu.get("vendor") or ""),
            "architecture": str(gpu.get("architecture") or ""),
            "isFallbackAdapter": bool(gpu.get("isFallbackAdapter")),
        },
    }


def of_build(
    settings: config.Config,
    items: examples.Examples,
    result: Result,
    *,
    device: Mapping[str, Any],
    single: Any = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The report of a build that ended with a model chosen or with none.

    ``device`` is :func:`device_of`. ``single`` is what the one-thread run of the chosen model gave:
    a ``build.Speed``, the text of why it was not measured, or None when it did not run.
    ``manifest`` is the manifest the folder gets and is needed when a model was chosen: the weight
    of the first load by parts comes from it, by :func:`bundle.first_load`, the one definition.
    """
    chosen = result.chosen
    speeds = [row.speed for row in result.rows if row.speed is not None]
    limits = settings.limits
    return {
        "format": FORMAT,
        "cifarium": cifarium.__version__,
        "outcome": BUILT if chosen is not None else NO_MODEL,
        "exitCode": exitcodes.DONE if chosen is not None else exitcodes.NO_MODEL,
        "task": settings.task,
        "settings": {
            "file": settings.path.name,
            "passRate": settings.pass_rate,
            "limits": {"maxSizeMb": limits.max_size_mb, "maxLatencyMs": limits.max_latency_ms},
            "license": settings.license,
            "host": settings.host,
        },
        "examples": {
            "file": settings.examples.name,
            "count": len(items.items),
            "labels": [label.name for label in items.labels],
        },
        "device": {
            **device,
            # Every run of one build is on the same page, so the first measured one speaks for all.
            "threads": speeds[0].threads if speeds else None,
            "crossOriginIsolated": speeds[0].isolated if speeds else None,
        },
        "chosen": None if chosen is None else _chosen(chosen, single, manifest),
        "candidates": [_candidate(row, items.items) for row in result.rows],
    }


def files(report: Mapping[str, Any]) -> dict[str, bytes]:
    """Both files of the report, by name, as :func:`bundle.write` and :func:`bundle.write_reports`
    take them."""
    return {bundle.REPORT_JSON: json_text(report), bundle.REPORT_MARKDOWN: markdown_text(report)}


def json_text(report: Mapping[str, Any]) -> bytes:
    """``report.json``: UTF-8, two-space indent, one ``\\n`` at the end, keys in the order made."""
    return (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


# --- the dictionary ------------------------------------------------------------------------------------


def _chosen(row: Row, single: Any, manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    if manifest is None:
        raise ValueError("a report of a build that chose a model needs the manifest of its folder")
    candidate = row.outcome.candidate
    entry = candidate.entry
    variant = bundle.FIRST_LOAD_VARIANT
    total, _ = bundle.first_load(manifest, variant)
    return {
        "id": entry.id,
        "model": entry.repo,
        "revision": entry.revision,
        "recipe": candidate.recipe.kind,
        "license": entry.license,
        "variant": variant,
        "dtype": entry.variants[variant].dtype,
        "firstLoad": {
            "bytes": total,
            "engineBytes": sum(manifest["engine"][role]["bytes"] for role in bundle.ENGINE_FILES),
            "dataBytes": sum(ref["bytes"] for ref in manifest["data"].values()),
            "modelBytes": sum(manifest["variants"][variant]["files"].values()),
        },
        "oneThread": None if single is None or isinstance(single, str) else _one_thread(single),
        "oneThreadError": _public(single) if isinstance(single, str) else None,
    }


def _candidate(row: Row, items: Sequence[examples.Example]) -> dict[str, Any]:
    outcome = row.outcome
    entry = outcome.candidate.entry
    score = outcome.evaluation
    return {
        "id": entry.id,
        "model": entry.repo,
        "recipe": outcome.candidate.recipe.kind,
        "license": entry.license,
        "verdict": row.verdict,
        "reason": _public(row.message),
        "firstLoadBytes": outcome.size_bytes,
        "evaluation": None if score is None else {
            "correct": score.correct,
            "total": score.total,
            "accuracy": _round(score.accuracy, SHARE_DIGITS),
            # Negative when the candidate fell short of the pass rate.
            "margin": _round(score.accuracy - score.pass_rate, SHARE_DIGITS),
            "misses": [_miss(miss, items) for miss in score.misses],
        },
        "speed": None if row.speed is None else _speed(row.speed),
    }


def _miss(miss: Any, items: Sequence[examples.Example]) -> dict[str, Any]:
    """The line of the example in the examples file and the two labels: never its text."""
    return {
        "line": items[miss.index].line,
        "expected": miss.expected,
        "predicted": miss.predicted,
        "score": _round(miss.score, SCORE_DIGITS),
        "expectedScore": None if miss.expected_score is None else _round(miss.expected_score, SCORE_DIGITS),
    }


def _speed(speed: Any) -> dict[str, Any]:
    return {
        "loadMs": _round(speed.load_ms, MS_DIGITS),
        "firstMs": _round(speed.first_ms, MS_DIGITS),
        "medianMs": _round(speed.median_ms, MS_DIGITS),
        "p95Ms": _round(speed.p95_ms, MS_DIGITS),
        "calls": speed.calls,
        "threads": speed.threads,
        "crossOriginIsolated": speed.isolated,
    }


def _one_thread(speed: Any) -> dict[str, Any]:
    return {
        "firstMs": _round(speed.first_ms, MS_DIGITS),
        "medianMs": _round(speed.median_ms, MS_DIGITS),
        "p95Ms": _round(speed.p95_ms, MS_DIGITS),
        "calls": speed.calls,
    }


def _round(value: float, digits: int) -> float:
    # Adding 0.0 turns a rounded -0.0 into 0.0.
    return round(float(value), digits) + 0.0


def _public(text: str) -> str:
    """A text of the machine for a file that is published: the folder of the user, the working
    folder and the cache of cifarium become ``~``, ``<project>`` and ``<cache>``. Texts of the bench
    and of the browser can name them."""
    found: list[tuple[str, str]] = []
    for make, stand_for in ((paths.cache_dir, "<cache>"), (Path.cwd, "<project>"), (Path.home, "~")):
        try:
            folder = make()
        except (OSError, RuntimeError):
            continue
        found.extend((spelling, stand_for) for spelling in {str(folder), folder.as_posix()})
    # The longest first: the cache lies in the home folder.
    for spelling, stand_for in sorted(found, key=lambda pair: len(pair[0]), reverse=True):
        if len(spelling) > 1:
            text = re.sub(re.escape(spelling), lambda _: stand_for, text, flags=re.IGNORECASE)
    return text


# --- the Markdown --------------------------------------------------------------------------------------


def markdown_text(report: Mapping[str, Any]) -> bytes:
    """``report.md``, made from the dictionary of :func:`of_build` and from nothing else."""
    lines = [
        "# cifarium build report",
        "",
        f"Made by cifarium {report['cifarium']}. Task `{report['task']}`, exit code {report['exitCode']}.",
        "",
        "## Result",
        "",
    ]
    chosen = report["chosen"]
    row = next((row for row in report["candidates"] if row["verdict"] == "passed"), None)
    if chosen is None or row is None:
        lines += [
            "**No model passed.** The table below says why each one was not taken. More examples per "
            "label, a lower `pass_rate`, wider `limits` or `license: any` in the settings may let one "
            "through.",
        ]
    else:
        lines += _chosen_lines(report, chosen, row)
    lines += ["", "## Candidates, the lightest first", ""]
    lines += _table(report["candidates"])
    lines += _miss_lines(report)
    lines += _settings_lines(report)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _chosen_lines(report: Mapping[str, Any], chosen: Mapping[str, Any], row: Mapping[str, Any]) -> list[str]:
    score, speed, load = row["evaluation"], row["speed"], chosen["firstLoad"]
    limit = report["settings"]["limits"]["maxLatencyMs"]
    lines = [
        f"**Chosen: {chosen['id']}** ({chosen['model']}, {chosen['dtype']} on the processor, "
        f"license {chosen['license']})",
        "",
        f"- Examples right: {score['correct']} of {score['total']} ({_percent(score['accuracy'])}). "
        f"The pass rate is {_percent(report['settings']['passRate'])}, so the margin is "
        f"{score['margin'] * 100:+.1f} points (one example is worth {100 / score['total']:.1f} points).",
        f"- First load: {_size(load['bytes'])} = {_size(load['engineBytes'])} engine + "
        # The recipe zero-shot has no data of the labels: its hypotheses are in the manifest itself.
        + (f"{_size(load['dataBytes'])} data of the labels + " if load["dataBytes"] else "")
        + f"{_size(load['modelBytes'])} files of the model.",
        f"- Model load: {speed['loadMs'] / 1000:.2f} s.",
        f"- Per example: median {_ms(speed['medianMs'])}, 95th percentile {_ms(speed['p95Ms'])}, "
        f"first call {_ms(speed['firstMs'])}.",
        f"- Threads: {speed['threads']}, "
        + ("crossOriginIsolated." if speed["crossOriginIsolated"] else "not crossOriginIsolated."),
    ]
    one = chosen["oneThread"]
    if one is not None:
        lines.append(f"- One thread: median {_ms(one['medianMs'])} of {one['calls']} examples, as on a "
                     "site without the isolation headers.")
    elif chosen["oneThreadError"] is not None:
        lines.append(f"- One thread: not measured: {_cell(chosen['oneThreadError'])}")
    if limit is None:
        lines.append("- Speed limit: none, `limits.max_latency_ms` is not set.")
    else:
        lines.append(f"- Speed limit: {_ms(speed['medianMs'])} is within `limits.max_latency_ms: {limit:g}`.")
        if one is not None and one["medianMs"] > limit:
            lines.append(f"- Warning: in one thread the usual answer takes {_ms(one['medianMs'])}, over the "
                         "limit: a host without the isolation headers (GitHub Pages) misses it.")
    return lines


def _table(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["| model | recipe | first load | verdict | right | margin | reason |",
             "|---|---|---:|---|---:|---:|---|"]
    for row in candidates:
        score = row["evaluation"]
        right = "" if score is None else f"{score['correct']} of {score['total']}"
        margin = "" if score is None else f"{score['margin'] * 100:+.1f} points"
        lines.append(f"| {row['id']} | {row['recipe']} | {_size(row['firstLoadBytes'])} | {row['verdict']} | "
                     f"{right} | {margin} | {_cell(row['reason'])} |")
    return lines


def _miss_lines(report: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    file = report["examples"]["file"]
    for row in report["candidates"]:
        score = row["evaluation"]
        if score is None or not score["misses"]:
            continue
        lines += ["", f"## Misses of {row['id']}", "", f"By the line in `{file}`; the text of an example is not "
                  "in this file.", ""]
        for miss in score["misses"]:
            own = "no centre" if miss["expectedScore"] is None else f"{miss['expectedScore']:.3f}"
            lines.append(f"- line {miss['line']}: `{miss['predicted']}` {miss['score']:.3f} instead of "
                         f"`{miss['expected']}` {own}")
    return lines


def _settings_lines(report: Mapping[str, Any]) -> list[str]:
    settings, examples_ = report["settings"], report["examples"]
    limit = settings["limits"]["maxLatencyMs"]
    device, browser, gpu = report["device"], report["device"]["browser"], report["device"]["webgpu"]
    if gpu is None:
        adapter = "none"
    else:
        adapter = " ".join(part for part in (gpu["vendor"], gpu["architecture"]) if part) or "unnamed"
        if gpu["isFallbackAdapter"]:
            adapter += " (software)"
    version = "" if browser["version"] is None else f" {browser['version']}"
    window = "without a window" if browser["headless"] else "with a window"
    threads = "not measured" if device["threads"] is None else (
        f"{device['threads']}, " + ("crossOriginIsolated" if device["crossOriginIsolated"]
                                     else "not crossOriginIsolated"))
    return [
        "",
        "## Settings and device",
        "",
        f"- Settings: `{settings['file']}`, pass rate {_percent(settings['passRate'])}, first load up to "
        f"{settings['limits']['maxSizeMb']:g} MB, usual answer "
        + ("without a limit" if limit is None else f"up to {limit:g} ms")
        + f", license {settings['license']}, host {settings['host']}.",
        f"- Examples: {examples_['count']} in `{examples_['file']}`, labels "
        + ", ".join(f"`{name}`" for name in examples_["labels"]) + ".",
        f"- Browser: {browser['name']}{version} {window}. WebGPU adapter: {adapter}.",
        f"- Threads of the engine: {threads}.",
    ]


def _cell(text: str) -> str:
    """A text in a table cell or on one line: no pipe, no line break."""
    return " ".join(text.split()).replace("|", "\\|")


def _size(size: int) -> str:
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    if size >= 1_000:
        return f"{size / 1_000:.1f} KB"
    return f"{size} B"


def _ms(value: float) -> str:
    return f"{value:.1f} ms"


def _percent(share: float) -> str:
    return f"{share * 100:.1f}".removesuffix(".0") + "%"

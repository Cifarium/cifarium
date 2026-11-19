"""Snapshots of the public contracts (ARCHITECTURE.md, section 14), without a browser.

Each test holds the list of one contract as a literal. A change to the contract changes the
literal in the same diff, so it cannot pass unseen, and CHANGELOG.md gets a line for it: adding
a key, a field or a code is allowed in any release, removing, renaming or giving a new meaning is
not outside a minor version. The browser half of the snapshots (the real `info()`, the codes
a run produces, the reasons of `fallback`) is in tests/test_runtime_e2e.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from test_bundle import GRAPH, MODEL_FILES, Sources, fill, fill_engine
from test_report import success as sample_report

from cifarium import build, bundle, catalog, cli, config, exitcodes, paths, report, selection

RUNTIME = paths.web_dir() / "runtime"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sources(tmp_path: Path) -> Sources:
    engine, model = tmp_path / "cache" / "engine", tmp_path / "cache" / "model"
    fill_engine(engine)
    fill(model, {**MODEL_FILES, **GRAPH})
    return Sources(engine, model)


def key_paths(value, prefix: str = "") -> list[str]:
    """The keys of a JSON value as paths. The names of the model's files, which are the keys of
    `files`, are one `*`: they belong to the model, not to the format."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, inner in value.items():
            here = f"{prefix}.{key}" if prefix else key
            if key == "files" and isinstance(inner, dict):
                found.append(f"{here}.*")
            elif isinstance(inner, (dict, list)):
                found.extend(key_paths(inner, here))
            else:
                found.append(here)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                found.extend(key_paths(item, prefix + "[]"))
            else:
                found.append(prefix + "[]")  # a list of plain values: the list is the key
    return sorted(set(found))


REPORT_KEYS = [
    "candidates[].evaluation.accuracy",
    "candidates[].evaluation.correct",
    "candidates[].evaluation.margin",
    "candidates[].evaluation.misses[].expected",
    "candidates[].evaluation.misses[].expectedScore",
    "candidates[].evaluation.misses[].line",
    "candidates[].evaluation.misses[].predicted",
    "candidates[].evaluation.misses[].score",
    "candidates[].evaluation.total",
    "candidates[].firstLoadBytes",
    "candidates[].id",
    "candidates[].license",
    "candidates[].model",
    "candidates[].reason",
    "candidates[].recipe",
    "candidates[].speed.calls",
    "candidates[].speed.crossOriginIsolated",
    "candidates[].speed.firstMs",
    "candidates[].speed.loadMs",
    "candidates[].speed.medianMs",
    "candidates[].speed.p95Ms",
    "candidates[].speed.threads",
    "candidates[].verdict",
    "chosen.dtype",
    "chosen.firstLoad.bytes",
    "chosen.firstLoad.dataBytes",
    "chosen.firstLoad.engineBytes",
    "chosen.firstLoad.modelBytes",
    "chosen.id",
    "chosen.license",
    "chosen.model",
    "chosen.oneThread.calls",
    "chosen.oneThread.firstMs",
    "chosen.oneThread.medianMs",
    "chosen.oneThread.p95Ms",
    "chosen.oneThreadError",
    "chosen.recipe",
    "chosen.revision",
    "chosen.variant",
    "cifarium",
    "device.browser.headless",
    "device.browser.name",
    "device.browser.version",
    "device.crossOriginIsolated",
    "device.threads",
    "device.webgpu.architecture",
    "device.webgpu.isFallbackAdapter",
    "device.webgpu.vendor",
    "examples.count",
    "examples.file",
    "examples.labels[]",
    "exitCode",
    "format",
    "outcome",
    "settings.file",
    "settings.host",
    "settings.license",
    "settings.limits.maxLatencyMs",
    "settings.limits.maxSizeMb",
    "settings.passRate",
    "task",
]
"""The keys of `report.json` as paths (ARCHITECTURE.md, section 14); the sample has every key a
report can have. A change here is a change of the contract and goes with a line in CHANGELOG.md."""


def report_key_paths(document) -> list[str]:
    """The keys of a report as paths. A key that is null in one row and an object in another is the
    object's: the null leaf is dropped when a longer path starts with it."""
    found = key_paths(document)
    return [path for path in found if not any(other.startswith((path + ".", path + "[]")) for other in found)]


def unknown_report_keys(document) -> list[str]:
    """The keys of a real report that the contract does not have. A key that is null in this report
    (no adapter, no model chosen, a candidate that never ran) is known when the contract has keys
    below it."""
    return [path for path in report_key_paths(document)
            if path not in REPORT_KEYS
            and not any(known.startswith((path + ".", path + "[]")) for known in REPORT_KEYS)]


def source(name: str) -> str:
    return (RUNTIME / name).read_text(encoding="utf-8")


def frozen_list(name: str) -> list[str]:
    match = re.search(rf"^export const {name} = Object\.freeze\(\[(.*?)\]\);$", source("cifarium.js"),
                      re.MULTILINE | re.DOTALL)
    assert match, f"cifarium.js does not export {name}"
    return re.findall(r"'([^']+)'", match.group(1))


def typedef(name: str) -> list[str]:
    """The `@property` names of a JSDoc typedef of cifarium.js, in the order written."""
    text = source("cifarium.js")
    block = re.search(rf"/\*\*\s*\n\s*\* @typedef \{{Object\}} {name}\b.*?\*/", text, re.DOTALL)
    assert block, f"cifarium.js has no typedef {name}"
    return re.findall(r"^ \* @property \{.*?\} \[?(\w+)\]?", block.group(0), re.MULTILINE | re.DOTALL)


# --- the manifest and the folder --------------------------------------------------------------------


def test_the_keys_of_the_manifest_are_the_contract(sources, tmp_path):
    manifest = sources.write(tmp_path / "out").manifest

    assert key_paths(manifest) == [
        "cifarium",
        "data.centres.bytes",
        "data.centres.path",
        "engine.executor.bytes",
        "engine.executor.path",
        "engine.library.bytes",
        "engine.library.path",
        "engine.loader.bytes",
        "engine.loader.path",
        "engine.onnxruntimeWeb",
        "engine.transformers",
        "format",
        "labels[].description",
        "labels[].name",
        "model.license",
        "model.path",
        "model.repo",
        "model.revision",
        "params.pooling",
        "params.prefix",
        "recipe",
        "task",
        "variants.cpu.device",
        "variants.cpu.dtype",
        "variants.cpu.externalData",
        "variants.cpu.files.*",
        "variants.cpu.graph",
    ]


def test_every_key_of_the_manifest_is_written_in_camel_case(sources, tmp_path):
    manifest = sources.write(tmp_path / "out").manifest

    for path in key_paths(manifest):
        for key in re.split(r"\.|\[\]", path):
            assert key in ("", "*") or re.fullmatch(r"[a-z][a-zA-Z0-9]*", key), path


def test_the_manifest_has_the_format_and_the_version_first_and_is_json_with_utf8_text(sources, tmp_path):
    out = tmp_path / "out"
    sources.write(out)

    raw = (out / "cifarium.json").read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    assert list(json.loads(raw))[:2] == ["format", "cifarium"]


def test_the_keys_of_the_report_are_the_contract():
    # The sample has a model chosen, one that ran too slow, one that fell short with its misses, a
    # one-thread run and an adapter. `chosen` and `evaluation` are null for a refusal and for a
    # candidate that never ran; `oneThread` is null when it was not measured.
    assert report_key_paths(sample_report()) == REPORT_KEYS


def test_every_key_of_the_report_is_written_in_camel_case():
    for path in report_key_paths(sample_report()):
        for key in re.split(r"\.|\[\]", path):
            assert key == "" or re.fullmatch(r"[a-z][a-zA-Z0-9]*", key), path


def test_the_report_has_the_format_and_the_version_first_and_is_json_with_utf8_text():
    raw = report.json_text(sample_report())

    assert raw.endswith(b"\n") and b"\r" not in raw
    assert list(json.loads(raw))[:2] == ["format", "cifarium"]
    assert json.loads(raw)["format"] == report.FORMAT == 1


def test_the_words_of_the_verdicts_and_of_the_outcome_are_the_contract():
    # `candidates[].verdict` is one of these; `outcome` is one of the last two; `latency` lives in `build`.
    assert [verdict.value for verdict in selection.Verdict] == [
        "passed", "license", "size", "language", "recipe", "accuracy", "vectors", "bench", "not-tried"]
    assert build.LATENCY == "latency"
    assert (report.BUILT, report.NO_MODEL) == ("built", "no-model")
    assert (bundle.REPORT_JSON, bundle.REPORT_MARKDOWN) == ("report.json", "report.md")


def test_the_runtime_describes_the_manifest_it_reads_with_the_keys_the_build_writes():
    # The Manifest typedef of cifarium.js and the keys of bundle.write are one list.
    assert typedef("Manifest") == ["format", "cifarium", "task", "recipe", "labels", "model", "variants",
                                   "params", "engine", "data"]
    assert bundle.FORMAT == 1


# --- the calls of the runtime -----------------------------------------------------------------------


def test_the_runtime_exports_these_names_and_no_others():
    exported = re.findall(r"^export (?:async )?(?:function|class|const) (\w+)", source("cifarium.js"),
                          re.MULTILINE)

    assert sorted(exported) == ["CifariumError", "ERROR_CODES", "FALLBACK_REASONS", "MANIFEST_FORMAT",
                                "PROGRESS_PHASES", "VERSION", "createCifarium"]


def test_the_error_codes_are_the_contract():
    assert frozen_list("ERROR_CODES") == ["manifest", "unsupported", "input", "not-loaded", "network", "load",
                                          "run", "disposed"]


def test_the_reasons_of_a_fallback_are_the_contract():
    assert frozen_list("FALLBACK_REASONS") == ["no-webgpu", "no-adapter", "software-adapter", "no-shader-f16",
                                               "requested", "failed"]


def test_the_phases_of_a_progress_event_are_the_contract():
    assert frozen_list("PROGRESS_PHASES") == ["download", "prepare"]


def test_the_fields_of_info_are_the_contract():
    assert typedef("Info") == [
        "state", "runtime", "cifarium", "format", "task", "recipe", "model", "revision", "license",
        "transformers", "onnxruntimeWeb", "labels", "variant", "device", "dtype", "threads",
        "crossOriginIsolated", "bytes", "fallback",
    ]


def test_the_fields_of_an_answer_a_label_a_fallback_and_a_progress_event_are_the_contract():
    assert typedef("Result") == ["label", "score", "scores"]
    assert typedef("Score") == ["label", "score"]
    assert typedef("Label") == ["name", "description"]
    assert typedef("Fallback") == ["reason", "message"]
    assert typedef("Progress") == ["phase", "file", "loaded", "total", "allLoaded", "allTotal", "percent"]


def test_the_runtime_uses_only_codes_and_reasons_it_lists():
    codes = set(frozen_list("ERROR_CODES"))
    used = set(re.findall(r"CifariumError\(\s*'([a-z-]+)'", source("cifarium.js")))
    used |= set(re.findall(r"Failure\(\s*'([a-z-]+)'", source("cifarium-worker.js")))
    for name in ("cifarium.js", "cifarium-worker.js"):  # a code chosen by a condition: `? 'network' : 'load'`
        used |= {code for pair in re.findall(r"\? '([a-z-]+)' : '([a-z-]+)'", source(name)) for code in pair}
    assert used <= codes, used - codes
    # Every listed code is raised somewhere: a code nobody raises is a promise nobody keeps.
    text = source("cifarium.js") + source("cifarium-worker.js")
    for code in codes:
        assert f"'{code}'" in text, code

    reasons = set(frozen_list("FALLBACK_REASONS"))
    made = set(re.findall(r"reason: '([a-z0-9-]+)'", source("cifarium-worker.js")))
    assert made == reasons


def test_the_runtime_sends_nothing_anywhere_by_its_source():
    # The browser half is in tests/test_page_e2e.py; this is the cheap part: no address that is
    # not the page's own, no way to send data out except fetch of a file of the folder.
    for name in ("cifarium.js", "cifarium-worker.js"):
        code = re.sub(r"/\*.*?\*/|//[^\n]*", "", source(name), flags=re.DOTALL)
        assert not re.search(r"https?://", code), name
        for forbidden in ("sendBeacon", "XMLHttpRequest", "WebSocket", "EventSource", "importScripts"):
            assert forbidden not in code, f"{name}: {forbidden}"
    page = (paths.web_dir() / "templates" / "index.html").read_text(encoding="utf-8")
    page_code = re.sub(r"<!--.*?-->|/\*.*?\*/", "", page, flags=re.DOTALL)
    assert not re.search(r"(?:src|href|action)\s*=\s*[\"']?(?:https?:)?//", page_code)
    assert not re.search(r"https?://", page_code)


# --- the settings, the exit codes, the license file -------------------------------------------------


def test_the_keys_and_the_values_of_the_settings_are_the_contract():
    assert config.KEYS == ("task", "examples", "labels", "pass_rate", "limits", "license", "host", "output")
    assert config.LIMIT_KEYS == ("max_size_mb", "max_latency_ms")
    assert config.TASKS == ("classify", "search", "summarize")
    assert config.SUPPORTED_TASKS == ("classify",)
    assert config.LICENSES == ("permissive", "any")
    assert config.HOSTS == ("bundle", "hub")
    # The licenses `license: permissive` lets through.
    assert catalog.PERMISSIVE_LICENSES == {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause"}


def test_the_keys_of_tasks_that_are_not_released_are_reserved_and_the_old_license_value_is_renamed():
    assert config.RESERVED == {"documents": "search", "similarity": "summarize",
                               "limits.min_tokens_per_sec": "summarize"}
    assert config.RENAMED == {"license": {"commercial": "permissive"}}


def test_the_defaults_of_the_settings_are_the_contract(tmp_path):
    # Changing one does not break a settings file but changes what a build does: CHANGELOG.md says so.
    (tmp_path / "examples.jsonl").write_text('{"input": "a", "expected": "x"}\n', encoding="utf-8")
    (tmp_path / "cifarium.yaml").write_text("task: classify\nexamples: examples.jsonl\n", encoding="utf-8")

    settings = config.load(tmp_path / "cifarium.yaml")

    assert (settings.pass_rate, settings.license, settings.host) == (0.85, "permissive", "bundle")
    assert settings.limits == config.Limits(max_size_mb=300.0, max_latency_ms=None)
    assert settings.labels is None
    assert settings.output == (tmp_path / "cifarium-out").resolve()
    assert config.DEFAULT_OUTPUT == "./cifarium-out" and config.DEFAULT_FILE == "cifarium.yaml"


def test_the_starter_settings_hold_the_defaults_they_show():
    text = (Path(cli.STARTER_DIR) / "cifarium.yaml").read_text(encoding="utf-8")

    for line in ("pass_rate: 0.85", "max_size_mb: 300", "license: permissive", "host: bundle",
                 "output: ./cifarium-out"):
        assert line in text, line


def test_the_exit_codes_are_the_contract():
    assert exitcodes.CODES == {"DONE": 0, "NO_MODEL": 1, "BAD_INPUT": 2, "ENVIRONMENT": 3}


def test_the_help_of_the_commands_names_the_exit_codes(capsys):
    for argv in (["--help"], ["build", "--help"], ["init", "--help"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
        text = " ".join(capsys.readouterr().out.split())
        assert "0 " in text and "2 " in text and "3 " in text, argv


def test_the_license_file_has_these_parts_in_this_order(sources, tmp_path):
    # The names of the parts and the full texts of the two licenses are the contract; the words
    # between them are not.
    out = tmp_path / "out"
    sources.write(out)

    text = (out / "LICENSES.txt").read_text(encoding="utf-8")
    headings = [line for line in text.splitlines() if line and not line.startswith(" ")
                and (line.startswith(("cifarium runtime", "Transformers.js ", "ONNX Runtime Web ", "The model: "))
                     or line.endswith(("Transformers.js", "ONNX Runtime")))]
    assert [heading.split(":")[0] for heading in headings] == [
        "cifarium runtime", "Transformers.js 4.3.2", "ONNX Runtime Web 1.31.0-dev.20260914-8d85527a0",
        "The model", "Apache License 2.0", "MIT License",
    ]
    assert text.index("Apache License\nVersion 2.0") < text.index("MIT License\n\nCopyright (c)")
    assert text.startswith("Licenses of the files in this folder\n")
    assert "\r" not in text


def test_the_catalog_fields_a_license_file_is_made_from_are_the_contract():
    assert catalog.KEYS == ("id", "repo", "revision", "license", "license_source", "notice", "languages",
                            "common_files", "variants", "recipes")
    assert catalog.RECIPES["embed-centroid"].keys() == {"pooling", "prefix", "dimensions"}


# --- the files around ------------------------------------------------------------------------------


def test_the_changelog_exists_and_names_the_rule_of_compatibility():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert text.startswith("# Changelog")
    assert "section 14" in text or "Section 14" in text
    assert "report.json" in text and "report.md" in text

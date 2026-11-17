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
from test_bundle import ENGINE_SIZES, GRAPH, MODEL_FILES, Sources, fill

from cifarium import bundle, paths

RUNTIME = paths.web_dir() / "runtime"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sources(tmp_path: Path) -> Sources:
    engine, model = tmp_path / "cache" / "engine", tmp_path / "cache" / "model"
    fill(engine, ENGINE_SIZES)
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
            found.extend(key_paths(item, prefix + "[]"))
    return sorted(set(found))


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


# --- the files around ------------------------------------------------------------------------------


def test_the_changelog_exists_and_names_the_rule_of_compatibility():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert text.startswith("# Changelog")
    assert "section 14" in text or "Section 14" in text

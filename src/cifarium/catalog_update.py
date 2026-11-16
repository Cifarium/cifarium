"""Service command: compare the catalog with Hugging Face and refresh the pinned facts.

For every entry it asks the Hub for the newest revision, applies the file-picking rule of
ARCHITECTURE.md, section 4, and reports what differs from the entry. By default it writes
nothing. With ``write`` it rewrites only ``revision``, ``common_files`` and ``variants``;
the fields a person checks by hand (``license``, ``license_source``, ``languages``,
``recipes``) are never touched, a license mismatch is only shown.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from huggingface_hub import HfApi

from . import catalog, hub
from .config import did_you_mean

# Files every model needs next to its graph.
COMMON_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
# The only dtypes the catalog takes for the cpu variant, and the graph file of each. The order
# is the tie-break: at equal weight q8 wins.
DTYPE_GRAPHS = {"q8": "onnx/model_quantized.onnx", "q4": "onnx/model_q4.onnx"}
VARIANT = "cpu"

# Top-level key of a YAML file: it starts in the first column.
_TOP_KEY = re.compile(r"([A-Za-z_][\w-]*):")


class Problem(Exception):
    """One entry cannot be updated; the message says why."""


@dataclass(frozen=True)
class Fresh:
    """What the catalog should say about one model, according to the Hub."""

    revision: str
    common_files: dict[str, int]
    dtype: str
    files: dict[str, int]
    """Files of the cpu variant: the graph and its ``.onnx_data`` parts."""


@dataclass
class Result:
    entry: catalog.Entry
    fresh: Fresh | None = None
    changes: list[str] = field(default_factory=list)
    """Differences ``write`` can fix."""
    notes: list[str] = field(default_factory=list)
    """Differences in fields that a person checks by hand."""
    errors: list[str] = field(default_factory=list)
    written: bool = False

    @property
    def status(self) -> str:
        parts = []
        if self.changes:
            parts.append(("changed (rewritten): " if self.written else "changed: ") + "; ".join(self.changes))
        if self.notes:
            parts.append("check by hand: " + "; ".join(self.notes))
        if self.errors:
            parts.append("ERROR: " + "; ".join(self.errors))
        return " | ".join(parts) or "unchanged"


def fresh_of(entry: catalog.Entry) -> Fresh:
    """The facts an entry holds now, in the shape of :func:`pick`."""
    cpu = entry.variants[VARIANT]
    return Fresh(entry.revision, dict(entry.common_files), cpu.dtype, dict(cpu.files))


def pick(info: hub.RepoInfo) -> Fresh:
    """Choose the files of one model by the rule of section 4. Raises :class:`Problem`.

    Common files are fixed. The cpu variant is q8 or q4, whichever weighs less counting the
    graph and its ``.onnx_data`` parts; at equal weight q8. Names are matched exactly:
    ``model_no_gather_q4.onnx`` is not q4.
    """
    missing = [name for name in COMMON_FILES if name not in info.files]
    if missing:
        raise Problem(f"common file missing on the Hub: {', '.join(missing)}")
    options: dict[str, dict[str, int]] = {}
    for dtype, graph in DTYPE_GRAPHS.items():
        if graph in info.files:
            data = re.compile(re.escape(graph) + r"_data(_\d+)?")
            options[dtype] = {name: size for name, size in sorted(info.files.items())
                              if name == graph or data.fullmatch(name)}
    if not options:
        raise Problem(f"no q8 ({DTYPE_GRAPHS['q8']}) or q4 ({DTYPE_GRAPHS['q4']}) file on the Hub")
    dtype = min(options, key=lambda name: (sum(options[name].values()), list(DTYPE_GRAPHS).index(name)))
    return Fresh(info.revision, {name: info.files[name] for name in COMMON_FILES}, dtype, options[dtype])


def _file_changes(old: dict[str, int], new: dict[str, int]) -> list[str]:
    changes = []
    for name in sorted(old.keys() | new.keys()):
        if name not in new:
            changes.append(f"-{name}")
        elif name not in old:
            changes.append(f"+{name} ({new[name]} bytes)")
        elif old[name] != new[name]:
            changes.append(f"{name} {old[name]} -> {new[name]} bytes")
    return changes


def diff(entry: catalog.Entry, fresh: Fresh) -> list[str]:
    """What differs between the entry and the fresh facts, one short phrase each."""
    old = fresh_of(entry)
    changes = []
    if old.revision != fresh.revision:
        changes.append(f"revision {old.revision[:7]} -> {fresh.revision[:7]}")
    changes += _file_changes(old.common_files, fresh.common_files)
    if old.dtype != fresh.dtype:
        changes.append(f"dtype {old.dtype} -> {fresh.dtype}")
    changes += _file_changes(old.files, fresh.files)
    return changes


def _license_notes(entry: catalog.Entry, info: hub.RepoInfo, api: HfApi) -> list[str]:
    """Differences in the fields checked by hand. The license is the source model's own."""
    notes = []
    if info.base_model and info.base_model != entry.license_source:
        notes.append(f"license_source is {entry.license_source}, the card now names "
                     f"{info.base_model} as base_model")
    if entry.license_source == entry.repo:
        source = info
    else:
        try:
            source = hub.repo_info(entry.license_source, api=api)
        except hub.HubError as error:
            return notes + [f"license not checked: {error}"]
    # Same wording as the entries: "unknown" when the card names no license.
    said = (source.license or "unknown").lower()
    if said != entry.license:
        notes.append(f"license is {entry.license}, the card of {entry.license_source} says {said}")
    return notes


def check(entry: catalog.Entry, api: HfApi) -> Result:
    """Compare one entry with the Hub. Reads only; never raises for a Hub or entry problem."""
    result = Result(entry)
    try:
        if set(entry.variants) != {VARIANT}:
            raise Problem(f"entry has variants other than {VARIANT}; this command updates only {VARIANT}")
        info = hub.repo_info(entry.repo, api=api)
        # Gating is checked on the copy visitors download; the source model of Gemma is gated
        # and that is fine.
        if info.gated:
            raise Problem("the repository now needs a signed-in account (gated)")
        result.fresh = pick(info)
        result.changes = diff(entry, result.fresh)
        result.notes = _license_notes(entry, info, api)
    except (Problem, hub.HubError) as error:
        result.errors.append(str(error))
    return result


def rewrite(text: str, fresh: Fresh) -> str:
    """The entry file with ``revision``, ``common_files`` and ``variants`` replaced.

    Every other line stays as it is, so the diff in git shows only the facts that changed.
    """
    eol = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        match = _TOP_KEY.match(lines[i])
        key = match.group(1) if match else None
        if key == "revision":
            out.append(f"revision: {fresh.revision}{eol}")
            i += 1
        elif key in ("common_files", "variants"):
            i += 1
            while i < len(lines) and (lines[i][:1] in (" ", "\t") or not lines[i].strip()):
                i += 1
            while not lines[i - 1].strip():  # blank lines after the block are not part of it
                i -= 1
            out.append(_block(key, fresh, eol))
        else:
            out.append(lines[i])
            i += 1
    return "".join(out)


def _block(key: str, fresh: Fresh, eol: str) -> str:
    if key == "common_files":
        rows = ["common_files:"] + [f"  {name}: {size}" for name, size in sorted(fresh.common_files.items())]
    else:
        rows = ["variants:", f"  {VARIANT}:", f"    device: {catalog.DEVICES[VARIANT]}",
                f"    dtype: {fresh.dtype}", "    files:"]
        rows += [f"      {name}: {size}" for name, size in sorted(fresh.files.items())]
    return "".join(row + eol for row in rows)


def _write(result: Result) -> None:
    """Rewrite the entry file, then read it back with the catalog's own checks."""
    entry, fresh = result.entry, result.fresh
    try:
        original = entry.path.read_bytes()
        text = original.decode("utf-8")
        new = rewrite(text, fresh)
        expected = {**yaml.safe_load(text), "revision": fresh.revision,
                    "common_files": dict(fresh.common_files),
                    "variants": {VARIANT: {"device": catalog.DEVICES[VARIANT], "dtype": fresh.dtype,
                                           "files": dict(fresh.files)}}}
        if yaml.safe_load(new) != expected:
            raise Problem("the rewritten text does not read back as the fresh facts")
        entry.path.write_bytes(new.encode("utf-8"))
        try:
            catalog.read_entry(entry.path)
        except catalog.CatalogError:
            entry.path.write_bytes(original)
            raise
    except (OSError, ValueError, yaml.YAMLError, Problem, catalog.CatalogError) as error:
        lines = str(error).strip().splitlines()
        result.errors.append(f"not rewritten: {lines[0] if lines else type(error).__name__}")
        return
    result.written = True


def run(
    *,
    write: bool = False,
    only: Iterable[str] = (),
    folder: Path | None = None,
    api: HfApi | None = None,
    out: Callable[[str], object] = print,
) -> int:
    """Check the catalog against the Hub and report. Returns the exit code.

    0 when the catalog says what Hugging Face says. 1 when something differs and was not
    fixed: a change without ``write``, a field to check by hand, or an error. With ``write``
    the changes it rewrote count as fixed.
    """
    try:
        entries = catalog.load(folder)
    except catalog.CatalogError as error:
        print(error, file=sys.stderr)
        return 1
    only = list(only)
    ids = [entry.id for entry in entries]
    for name in only:
        if name not in ids:
            print(f"--only {name}: no such catalog entry{did_you_mean(name, ids)}", file=sys.stderr)
    if any(name not in ids for name in only):
        return 1
    if only:
        entries = tuple(entry for entry in entries if entry.id in only)

    api = api or HfApi()
    results = [check(entry, api) for entry in entries]
    if write:
        for result in results:
            if result.changes and not result.errors:
                _write(result)

    width = max(len(result.entry.id) for result in results)
    for result in results:
        out(f"{result.entry.id:<{width}}  {result.status}")
    errors = [r for r in results if r.errors]
    changed = [r for r in results if r.changes and not r.errors]
    by_hand = [r for r in results if r.notes and not r.changes and not r.errors]
    unchanged = len(results) - len(errors) - len(changed) - len(by_hand)
    summary = (f"{len(results)} entries: {unchanged} unchanged, {len(changed)} changed, "
               f"{len(by_hand)} to check by hand, {len(errors)} with errors")
    rewritten = sum(result.written for result in results)
    if write:
        summary += f"; {rewritten} files rewritten"
    out(summary)
    if changed and not write:
        out("Nothing written. Run with --write to rewrite revision, common_files and variants.")

    pending = any(r.errors or r.notes or (r.changes and not r.written) for r in results)
    return 1 if pending else 0

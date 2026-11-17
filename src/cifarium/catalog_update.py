"""Service command: compare the catalog with Hugging Face and refresh the pinned facts.

For every entry it asks the Hub for the newest revision, applies the file-picking rule of
ARCHITECTURE.md, section 4, and reports what differs from the entry. When the newest revision
is not the pinned one it also asks for the pinned revision: if the Hub no longer has it that is
an error (downloads of the model are broken), and if both hold the same files a newer hash
alone is not a change. By default it writes nothing. With ``write`` it rewrites only
``revision``, ``common_files`` and ``variants``; the fields a person checks by hand
(``license``, ``license_source``, ``languages``, ``recipes``) are never touched, a license
mismatch is only shown.
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
# The only dtypes the catalog takes for the cpu variant, and the graph files of each, the
# preferred one first. The order of the dtypes is the tie-break: at equal weight q8 wins.
# q4 prefers the graph without GatherBlockQuantized: the build of onnxruntime-web in the
# engine has no such operation, so ``model_q4.onnx`` of Gemma does not load on the processor.
DTYPE_GRAPHS = {
    "q8": ("onnx/model_quantized.onnx",),
    "q4": ("onnx/model_no_gather_q4.onnx", "onnx/model_q4.onnx"),
}
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
    newer: str | None = None
    """Hash of a newer revision that holds exactly the files of the pinned one. Not a change:
    ``write`` leaves the entry alone and the exit code stays 0."""
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
        text = " | ".join(parts)
        if self.newer:
            remark = f"newer revision {self.newer[:7]} has the same files"
            return f"{text} | {remark}" if text else f"unchanged ({remark})"
        return text or "unchanged"


def fresh_of(entry: catalog.Entry) -> Fresh:
    """The facts an entry holds now, in the shape of :func:`pick`."""
    cpu = entry.variants[VARIANT]
    return Fresh(entry.revision, dict(entry.common_files), cpu.dtype, dict(cpu.files))


def pick(info: hub.RepoInfo) -> Fresh:
    """Choose the files of one model by the rule of section 4. Raises :class:`Problem`.

    Common files are fixed. The cpu variant is q8 or q4, whichever weighs less counting the
    graph and its ``.onnx_data`` parts; at equal weight q8. The graph of a dtype is the first
    of its names in ``DTYPE_GRAPHS`` that the Hub has, and only its own data parts come with
    it. Names are matched exactly, never as substrings.
    """
    missing = [name for name in COMMON_FILES if name not in info.files]
    if missing:
        raise Problem(f"common file missing on the Hub: {', '.join(missing)}")
    options: dict[str, dict[str, int]] = {}
    for dtype, graphs in DTYPE_GRAPHS.items():
        graph = next((name for name in graphs if name in info.files), None)
        if graph is not None:
            data = re.compile(re.escape(graph) + r"_data(_\d+)?")
            options[dtype] = {name: size for name, size in sorted(info.files.items())
                              if name == graph or data.fullmatch(name)}
    if not options:
        wanted = " or ".join(f"{dtype} ({' or '.join(graphs)})" for dtype, graphs in DTYPE_GRAPHS.items())
        raise Problem(f"no {wanted} file on the Hub")
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
    """What differs in the files and the dtype between the entry and the fresh facts.

    The revision is not part of it: a new hash alone is not a change, see :func:`check`.
    """
    old = fresh_of(entry)
    changes = _file_changes(old.common_files, fresh.common_files)
    if old.dtype != fresh.dtype:
        changes.append(f"dtype {old.dtype} -> {fresh.dtype}")
    return changes + _file_changes(old.files, fresh.files)


def _content_changes(entry: catalog.Entry, pinned: hub.RepoInfo, latest: hub.RepoInfo,
                     fresh: Fresh) -> list[str]:
    """Files of the same name and size whose content still differs, by ``blob_id``.

    A file with another size is already named by :func:`diff`. A missing id on either side
    cannot prove the files are the same, so it counts as a difference.
    """
    old = fresh_of(entry)
    before = {**old.common_files, **old.files}
    changes = []
    for name, size in sorted({**fresh.common_files, **fresh.files}.items()):
        if before.get(name) != size:
            continue
        was, now = pinned.blobs.get(name), latest.blobs.get(name)
        if not was or not now:
            changes.append(f"{name} cannot be compared, the Hub gave no blob id")
        elif was != now:
            changes.append(f"{name} content changed, same size ({size} bytes)")
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
        if info.revision != entry.revision:
            _compare_with_pinned(entry, info, result, api)
        result.notes = _license_notes(entry, info, api)
    except (Problem, hub.HubError) as error:
        result.errors.append(str(error))
    return result


def _compare_with_pinned(entry: catalog.Entry, latest: hub.RepoInfo, result: Result, api: HfApi) -> None:
    """Decide what the newer hash means: a change, or only a newer hash over the same files.

    A new hash is also what an edit of the model card makes. Rewriting the entry for it would
    rename the model's folder in every build and clear every cache for nothing, so the entry
    changes only when the files differ: names, dtype, sizes or ``blob_id``.
    """
    short = entry.revision[:7]
    try:
        pinned = hub.repo_info(entry.repo, revision=entry.revision, api=api)
    except hub.RevisionNotFound:
        # What the newest revision holds is still shown, to help re-pin by hand; the error
        # stops ``write`` from touching the entry.
        result.changes.insert(0, f"revision {short} -> {latest.revision[:7]}")
        raise Problem(f"the pinned revision {short} is gone from Hugging Face, so downloading "
                      "this model fails for everyone who installed cifarium") from None
    except hub.HubError as error:
        raise Problem(f"could not check the pinned revision {short}: {error}") from None
    result.changes += _content_changes(entry, pinned, latest, result.fresh)
    if result.changes:
        result.changes.insert(0, f"revision {short} -> {latest.revision[:7]}")
    else:
        result.newer = latest.revision


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

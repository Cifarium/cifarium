"""The folder of a build: what the site gets (ARCHITECTURE.md, section 7).

:func:`write` lays out the runtime, the engine, the model, the manifest, the label data, the
example page, the hosting settings and the license file. It needs no browser: the vectors of
the examples come in from the build as the centres of the labels. A folder it wrote earlier is
brought up to date; files it did not write are never touched.
"""

from __future__ import annotations

import json
import os
import shutil
import string
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

import cifarium

from . import catalog, hub, paths, vendor
from .bench import external_data_files, graph_name

FORMAT = 1
"""Manifest format. The runtime reads only this one (``MANIFEST_FORMAT`` in cifarium.js)."""

MANIFEST = "cifarium.json"
RUNTIME_FILES = ("cifarium.js", "cifarium-worker.js")
PAGE = "index.html"
LICENSES = "LICENSES.txt"
HOSTING_FILES = ("_headers", "vercel.json", "nginx.conf")
ENGINE_FILES = {
    "library": vendor.LIBRARY_FILE,
    "loader": vendor.WASM_LOADER_FILE,
    "executor": vendor.WASM_BINARY_FILE,
}
LABELS_FILE = "data/labels.json"
CENTRES_FILE = "data/labels.bin"

TEMPLATES = "templates"
"""Folder of the example page, the hosting settings and the license file under ``web/``."""


class BundleError(RuntimeError):
    """The folder could not be written: a file is missing or wrong, or the folder cannot be used."""


@dataclass(frozen=True)
class Built:
    folder: Path
    manifest: dict[str, Any]
    files: tuple[str, ...]
    """Every file of this build, relative to the folder, with forward slashes."""
    removed: tuple[str, ...]
    """Files of the previous build that this one does not have any more, and that were deleted."""
    first_load_bytes: int
    """What a first visit downloads for the processor's variant: engine, model and data."""
    size_bytes: int
    """Everything the build holds on disk, which is what an upload to the site sends."""


def write(
    folder: Path,
    candidate: catalog.Candidate,
    names: Sequence[str],
    centres: np.ndarray,
    *,
    variants: Sequence[str] = ("cpu",),
    engine_dir: Path | None = None,
    model_dir: Path | None = None,
    engine_sizes: Mapping[str, int] = vendor.ENGINE_BYTES,
) -> Built:
    """Write the build of ``candidate`` into ``folder`` and return what is there now.

    ``names`` are the labels in the order of the settings and ``centres`` has one row for each,
    the way :func:`evaluate.centroids` gives them. ``variants`` are the variants of the entry that
    the bench has checked. The engine and the model are copied from the cache (``engine_dir`` and
    ``model_dir`` replace it); a file whose size is not the one the catalog or the pinned engine
    names is an error, because the manifest promises these sizes to the runtime.

    The manifest is written after everything it names, so a page that reads it while the build
    runs sees the old build or the new one. Files of the previous build that the new one does
    not name are deleted at the end; see :func:`stale`.
    """
    entry, recipe = candidate.entry, candidate.recipe
    names = list(names)
    rows = _centres(names, centres)
    if recipe.kind != "embed-centroid":
        raise BundleError(f"{entry.id}: the folder of the recipe {recipe.kind} is not written yet")
    if not variants:
        raise BundleError(f"{entry.id}: a build needs at least one variant")
    for variant in variants:
        if variant not in entry.variants:
            raise BundleError(f"{entry.id} has no {variant} variant")
    folder = Path(folder)
    if folder.exists() and not folder.is_dir():
        raise BundleError(f"{folder} is a file, not a folder: change `output` in the settings")
    engine_dir = vendor.ensure_engine() if engine_dir is None else engine_dir
    model_dir = hub.model_dir(entry.repo) if model_dir is None else model_dir

    before = _previous(folder / MANIFEST)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        engine_path = f"vendor/transformers-{vendor.TRANSFORMERS_VERSION}/"
        engine: dict[str, Any] = {
            "transformers": vendor.TRANSFORMERS_VERSION,
            "onnxruntime_web": vendor.ORT_WEB_VERSION,
        }
        for role, name in ENGINE_FILES.items():
            _copy(engine_dir / name, folder / engine_path / name, engine_sizes[name])
            engine[role] = {"path": engine_path + name, "bytes": engine_sizes[name]}

        model_path = f"models/{entry.revision}/"
        for variant in variants:
            for name, size in entry.files(variant).items():
                _copy(model_dir / name, folder / model_path / entry.repo / name, size)

        labels = json.dumps(names, ensure_ascii=False).encode("utf-8")
        _write_bytes(folder / LABELS_FILE, labels)
        _write_bytes(folder / CENTRES_FILE, rows.tobytes())

        manifest = {
            "format": FORMAT,
            "cifarium": cifarium.__version__,
            "task": candidate.recipe.task,
            "recipe": recipe.kind,
            "model": {"repo": entry.repo, "revision": entry.revision, "license": entry.license,
                      "path": model_path},
            "variants": {variant: variant_record(entry, variant) for variant in variants},
            "params": {"pooling": recipe.params["pooling"], "prefix": recipe.params["prefix"]},
            "engine": engine,
            "data": {
                "labels": {"path": LABELS_FILE, "bytes": len(labels)},
                "centres": {"path": CENTRES_FILE, "bytes": rows.nbytes},
            },
        }

        runtime = paths.web_dir() / "runtime"
        for name in RUNTIME_FILES:
            _write_bytes(folder / name, (runtime / name).read_bytes())
        templates = paths.web_dir() / TEMPLATES
        _write_bytes(folder / PAGE, (templates / PAGE).read_bytes())
        for name in HOSTING_FILES:
            _write_bytes(folder / "hosting" / name, (templates / "hosting" / name).read_bytes())
        _write_bytes(folder / LICENSES, licenses_text(manifest, entry))
        _write_bytes(folder / MANIFEST, _manifest_text(manifest))

        now = owned(manifest)
        removed = _remove(folder, stale(before, now))
    except OSError as error:
        raise BundleError(f"cannot write {folder}: {error.strerror or error}") from error

    ordered = tuple(sorted(now))
    total, _ = first_load(manifest, variants[0])
    return Built(folder, manifest, ordered, removed, total, sum((folder / name).stat().st_size for name in ordered))


def variant_record(entry: catalog.Entry, name: str) -> dict[str, Any]:
    """The manifest's record of one variant: how the library loads it and which files it reads."""
    variant = entry.variants[name]
    return {
        "device": variant.device,
        "dtype": variant.dtype,
        "graph": graph_name(variant.files, variant.dtype),
        "externalData": external_data_files(variant.files),
        "files": entry.files(name),
    }


def first_load(manifest: Mapping[str, Any], variant: str) -> tuple[int, set[str]]:
    """Bytes and paths, relative to the manifest, of everything a first visit downloads for
    ``variant``: the engine, the label data and the files of the variant."""
    refs = [manifest["engine"][role] for role in ENGINE_FILES] + list(manifest["data"].values())
    folder = manifest["model"]["path"] + manifest["model"]["repo"] + "/"
    files = {ref["path"]: ref["bytes"] for ref in refs}
    files.update({folder + name: size for name, size in manifest["variants"][variant]["files"].items()})
    return sum(files.values()), set(files)


def owned(manifest: Mapping[str, Any]) -> set[str]:
    """Every file a build with this manifest writes: the ones it names and the fixed ones."""
    files = {MANIFEST, PAGE, LICENSES, *RUNTIME_FILES, *(f"hosting/{name}" for name in HOSTING_FILES)}
    files.update(manifest["engine"][role]["path"] for role in ENGINE_FILES)
    files.update(ref["path"] for ref in manifest["data"].values())
    folder = manifest["model"]["path"] + manifest["model"]["repo"] + "/"
    for variant in manifest["variants"].values():
        files.update(folder + name for name in variant["files"])
    return files


def stale(before: Iterable[str], now: Iterable[str]) -> list[str]:
    """What the previous build wrote and this one does not: the files that are safe to delete.

    Only files the previous manifest names, so a file the developer put into the folder stays.
    A model of another revision, the files of another variant and the engine of another
    version are the usual cases.
    """
    keep = set(now)
    return sorted(name for name in set(before) if name not in keep)


def licenses_text(manifest: Mapping[str, Any], entry: catalog.Entry) -> bytes:
    template = string.Template((paths.web_dir() / TEMPLATES / LICENSES).read_text(encoding="utf-8"))
    text = template.substitute(
        cifarium=manifest["cifarium"],
        transformers=manifest["engine"]["transformers"],
        onnxruntime_web=manifest["engine"]["onnxruntime_web"],
        engine_path=PurePosixPath(manifest["engine"]["library"]["path"]).parent.as_posix(),
        repo=entry.repo,
        revision=entry.revision,
        license=entry.license,
        license_source=entry.license_source,
        model_path=manifest["model"]["path"],
    )
    return text.replace("\r\n", "\n").encode("utf-8")


def _centres(names: list[str], centres: np.ndarray) -> np.ndarray:
    rows = np.asarray(centres, dtype="<f4")
    if len(names) < 2 or len(set(names)) != len(names):
        raise BundleError(f"the labels must be at least two and all different, got {names}")
    if rows.ndim != 2 or rows.shape[0] != len(names) or rows.shape[1] < 1:
        raise BundleError(f"{len(names)} labels need one centre each, got an array of shape {rows.shape}")
    if not np.isfinite(rows).all():
        raise BundleError("a centre of a label is not a finite number")
    return rows


def _previous(manifest_path: Path) -> set[str]:
    """Files the build before this one wrote, from its manifest; none if there is no readable one."""
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            return set()
        return {name for name in owned(data) if isinstance(name, str)}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return set()


def _copy(source: Path, target: Path, size: int) -> None:
    """Copy a file whose name carries its version. Skipped when the target is already whole."""
    try:
        actual = source.stat().st_size
    except OSError:
        raise BundleError(f"{source} is missing: run `cifarium build` again to download it") from None
    if actual != size:
        raise BundleError(f"{source} is {actual} bytes, but {size} are expected: delete the file "
                          "so that `cifarium build` downloads it again")
    try:
        if target.stat().st_size == size:
            return
    except OSError:
        pass
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    shutil.copyfile(source, part)
    os.replace(part, target)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)


def _manifest_text(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _remove(folder: Path, names: Sequence[str]) -> tuple[str, ...]:
    """Delete the listed files of the folder, then the folders they leave empty."""
    root = folder.resolve()
    removed = []
    for name in names:
        parts = PurePosixPath(name).parts
        if not parts or PurePosixPath(name).is_absolute() or ".." in parts or "\\" in name:
            continue
        file = root.joinpath(*parts)
        # A drive letter or a link in the old manifest's path must not lead out of the folder.
        if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(root):
            continue
        file.unlink()
        removed.append(name)
        parent = file.parent
        while parent != root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return tuple(removed)

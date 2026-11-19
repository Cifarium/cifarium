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
from .config import Label

FORMAT = 1
"""Manifest format. The runtime reads this one and every older one (``MANIFEST_FORMAT`` in
cifarium.js); ARCHITECTURE.md, section 14, says what changes the number."""

MANIFEST = "cifarium.json"
RUNTIME_FILES = ("cifarium.js", "cifarium-worker.js")
PAGE = "index.html"
LICENSES = "LICENSES.txt"
REPORT_JSON = "report.json"
REPORT_MARKDOWN = "report.md"
REPORT_FILES = (REPORT_JSON, REPORT_MARKDOWN)
"""The report of the build, for programs and for people (:mod:`report`). The folder owns both names,
so a rebuild writes them again, but only the files :func:`write` was given are written."""
HOSTING_FILES = ("_headers", "vercel.json", "nginx.conf")
ENGINE_FILES = {
    "library": vendor.LIBRARY_FILE,
    "loader": vendor.WASM_LOADER_FILE,
    "executor": vendor.WASM_BINARY_FILE,
}
CENTRES_FILE = "data/labels.bin"

FIRST_LOAD_VARIANT = "cpu"
"""The variant whose weight is "the first load" while the device is not known: the processor's.
Every visitor can run it and stage 1 builds nothing else (ARCHITECTURE.md, section 7)."""

PARAMS = {"embed-centroid": ("pooling", "prefix")}
"""The parameters of a recipe that go into the manifest. ``dimensions`` stays in the catalog: the
runtime reads the length of a vector from the size of the centres file."""

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
    """What a first visit downloads, by :func:`first_load`, for the first of the variants written."""
    size_bytes: int
    """Everything the build holds on disk, which is what an upload to the site sends."""


def write(
    folder: Path,
    candidate: catalog.Candidate,
    labels: Sequence[Label],
    centres: np.ndarray,
    *,
    variants: Sequence[str] = ("cpu",),
    engine_dir: Path | None = None,
    model_dir: Path | None = None,
    engine_sizes: Mapping[str, int] = vendor.ENGINE_BYTES,
    reports: Mapping[str, bytes] | None = None,
) -> Built:
    """Write the build of ``candidate`` into ``folder`` and return what is there now.

    ``labels`` are the labels of the settings, name and description, in their order, and
    ``centres`` has one row for each, the way :func:`evaluate.centroids` gives them. They go into
    the manifest; nothing of the examples does, and everything in the folder may be published
    (ARCHITECTURE.md, section 7). ``variants`` are the variants of the entry that the bench has
    checked. The engine and the model are copied from the cache (``engine_dir`` and ``model_dir``
    replace it); a file whose size is not the one the catalog or the pinned engine names is an
    error, because the manifest promises these sizes to the runtime.

    ``reports`` are the files of the report by name (:data:`REPORT_FILES`), made by :mod:`report`
    before the folder is written; they are written with the other files, before the manifest. A
    report that is not given is not written, and an older one in the folder is left as it is.

    The same input gives the same bytes in every file, with no time or path in them. The
    manifest is written after everything it names, so a page that reads it while the build
    runs sees the old build or the new one. Files of the previous build that the new one does
    not name are deleted at the end; see :func:`stale`.
    """
    reports = _checked_reports(reports)
    entry, recipe = candidate.entry, candidate.recipe
    labels = list(labels)
    names = [label.name for label in labels]
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

    manifest = manifest_of(candidate, labels, rows.nbytes, variants=variants, engine_sizes=engine_sizes)
    before = _previous(folder / MANIFEST)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for role, name in ENGINE_FILES.items():
            ref = manifest["engine"][role]
            _copy(engine_dir / name, folder / ref["path"], ref["bytes"])

        model_path = manifest["model"]["path"]
        for variant in variants:
            for name, size in entry.files(variant).items():
                _copy(model_dir / name, folder / model_path / entry.repo / name, size)

        _write_bytes(folder / CENTRES_FILE, rows.tobytes())

        runtime = paths.web_dir() / "runtime"
        for name in RUNTIME_FILES:
            _write_bytes(folder / name, (runtime / name).read_bytes())
        templates = paths.web_dir() / TEMPLATES
        _write_bytes(folder / PAGE, (templates / PAGE).read_bytes())
        for name in HOSTING_FILES:
            _write_bytes(folder / "hosting" / name, (templates / "hosting" / name).read_bytes())
        _write_bytes(folder / LICENSES, licenses_text(manifest, entry, engine_dir))
        for name, data in reports.items():
            _write_bytes(folder / name, data)
        _write_bytes(folder / MANIFEST, _manifest_text(manifest))

        now = owned(manifest)
        removed = _remove(folder, stale(before, now))
    except OSError as error:
        raise BundleError(f"cannot write {folder}: {error.strerror or error}") from error

    # The folder owns both report names, but a file of an earlier build is not a file of this one.
    ordered = tuple(sorted(name for name in now if name not in REPORT_FILES or name in reports))
    total, _ = first_load(manifest, variants[0])
    return Built(folder, manifest, ordered, removed, total, sum((folder / name).stat().st_size for name in ordered))


def write_reports(folder: Path, reports: Mapping[str, bytes]) -> tuple[str, ...]:
    """Write only the files of the report into ``folder``, which is made if it is not there. This is
    the end of a build that chose no model: there is no folder to write, but the table of reasons
    must reach the developer and a script. Nothing else in the folder is touched, so a build of an
    earlier run stays as it was. Returns the names written; :class:`BundleError` when the folder
    cannot be used."""
    reports = _checked_reports(reports)
    folder = Path(folder)
    if folder.exists() and not folder.is_dir():
        raise BundleError(f"{folder} is a file, not a folder: change `output` in the settings")
    try:
        for name, data in reports.items():
            _write_bytes(folder / name, data)
    except OSError as error:
        raise BundleError(f"cannot write {folder}: {error.strerror or error}") from error
    return tuple(reports)


def _checked_reports(reports: Mapping[str, bytes] | None) -> dict[str, bytes]:
    found = dict(reports or {})
    unknown = sorted(set(found) - set(REPORT_FILES))
    if unknown:
        raise ValueError(f"{unknown} are not files of the report, which are {list(REPORT_FILES)}")
    return found


def manifest_of(
    candidate: catalog.Candidate,
    labels: Sequence[Label],
    centres_bytes: int,
    *,
    variants: Sequence[str] = (FIRST_LOAD_VARIANT,),
    engine_sizes: Mapping[str, int] = vendor.ENGINE_BYTES,
) -> dict[str, Any]:
    """The manifest of a build of ``candidate``. Touches no file: the build and the selection both
    ask it, so that what the selection weighs is what the build writes.

    ``centres_bytes`` is the size of the centres file: four bytes for each number of each label.
    Only the recipe ``embed-centroid`` has such a file; a recipe the folder is not written for yet
    has no data (:func:`write` refuses it) and its weight is that of the engine and the model.
    """
    entry, recipe = candidate.entry, candidate.recipe
    engine_path = f"vendor/transformers-{vendor.TRANSFORMERS_VERSION}/"
    engine: dict[str, Any] = {
        "transformers": vendor.TRANSFORMERS_VERSION,
        "onnxruntimeWeb": vendor.ORT_WEB_VERSION,
    }
    for role, name in ENGINE_FILES.items():
        engine[role] = {"path": engine_path + name, "bytes": engine_sizes[name]}
    data = {"centres": {"path": CENTRES_FILE, "bytes": centres_bytes}} if recipe.kind == "embed-centroid" else {}
    return {
        "format": FORMAT,
        "cifarium": cifarium.__version__,
        "task": recipe.task,
        "recipe": recipe.kind,
        "labels": [{"name": label.name, "description": label.description} for label in labels],
        "model": {"repo": entry.repo, "revision": entry.revision, "license": entry.license,
                  "path": f"models/{entry.revision}/"},
        "variants": {variant: variant_record(entry, variant) for variant in variants},
        "params": {key: recipe.params[key] for key in PARAMS.get(recipe.kind, ())},
        "engine": engine,
        # The files of `data` depend on the recipe; embed-centroid has the centres only.
        "data": data,
    }


def centres_bytes(candidate: catalog.Candidate, labels: int) -> int:
    """The size the centres file of a build of ``candidate`` will have, from the number of labels
    and the length of the vector the catalog names (``dimensions``); 0 for a recipe without one.
    The selection needs it before there is a vector to measure."""
    if candidate.recipe.kind != "embed-centroid":
        return 0
    return 4 * labels * candidate.recipe.params["dimensions"]


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
    """THE definition of the weight of a first visit: bytes and paths, relative to the manifest, of
    everything the first visit of a page with this build downloads for ``variant``: the three files
    of the engine, the data of the recipe, and the files of the variant (the model's own and the
    ones every variant shares). Nothing else: not the page, not the runtime, not the manifest, not
    the files of the other variants. Uncompressed sizes, as the manifest holds them.

    The selection (through :func:`manifest_of`), ``limits.max_size_mb``, :attr:`Built.first_load_bytes`
    and the runtime's ``info().bytes`` are this number; ARCHITECTURE.md, section 7.
    """
    refs = [manifest["engine"][role] for role in ENGINE_FILES] + list(manifest["data"].values())
    folder = manifest["model"]["path"] + manifest["model"]["repo"] + "/"
    files = {ref["path"]: ref["bytes"] for ref in refs}
    files.update({folder + name: size for name, size in manifest["variants"][variant]["files"].items()})
    return sum(files.values()), set(files)


def owned(manifest: Mapping[str, Any]) -> set[str]:
    """Every file a build with this manifest writes: the ones it names and the fixed ones."""
    files = {MANIFEST, PAGE, LICENSES, *REPORT_FILES, *RUNTIME_FILES,
             *(f"hosting/{name}" for name in HOSTING_FILES)}
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


def licenses_text(manifest: Mapping[str, Any], entry: catalog.Entry, engine_dir: Path) -> bytes:
    """``LICENSES.txt``: the parts of the folder and their licenses, the full texts of the licenses of
    the engine (read from ``engine_dir``, where :mod:`vendor` stored them) and the notice of the
    model's license when the catalog has one."""
    template = string.Template((paths.web_dir() / TEMPLATES / LICENSES).read_text(encoding="utf-8"))
    notice = ""
    if entry.notice:
        lines = "\n".join(f"    {line}".rstrip() for line in entry.notice.splitlines())
        notice = f"\n\n  The notice that goes with the model, from its terms or its card:\n{lines}"
    text = template.substitute(
        cifarium=manifest["cifarium"],
        transformers=manifest["engine"]["transformers"],
        onnxruntime_web=manifest["engine"]["onnxruntimeWeb"],
        engine_path=PurePosixPath(manifest["engine"]["library"]["path"]).parent.as_posix(),
        repo=entry.repo,
        revision=entry.revision,
        license=entry.license,
        license_source=entry.license_source,
        model_path=manifest["model"]["path"],
        model_notice=notice,
        transformers_license=_license_of(engine_dir, vendor.TRANSFORMERS_LICENSE_FILE),
        onnxruntime_license=_license_of(engine_dir, vendor.ORT_LICENSE_FILE),
    )
    return text.replace("\r\n", "\n").encode("utf-8")


def _license_of(engine_dir: Path, name: str) -> str:
    """The text of a license stored with the engine, with the line ends of the file and no blank
    lines around it."""
    try:
        text = (engine_dir / name).read_text(encoding="utf-8")
    except OSError:
        raise BundleError(f"{engine_dir / name} is missing: delete the folder {engine_dir} and run "
                          "`cifarium build` again to download the engine with its licenses") from None
    return text.replace("\r\n", "\n").strip("\n")


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

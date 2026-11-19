"""The cache of results: the vectors the bench computed, kept between builds.

An entry is the vector of one text. A model gives the same vector for the same text every
time, so a build that changes one example computes one vector, and a build that changes
nothing computes none. A file holds the vectors of one *key*: everything the vector depends
on except the text (:func:`components`). A vector is only ever found under the key it was
computed with, and a file that cannot be read, or does not hold what its key says, counts as
empty: a wrong vector from the cache is worse than no cache.

What the cache does not hold: the times of the bench (a build measures them on the machine
that runs it, now), a failure of the bench, and the verdicts. ARCHITECTURE.md, sections 6 and
9, has the choices.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

import cifarium

from . import catalog, paths, vendor

FORMAT = 1
"""Layout of the files and the list of what makes a key. A change of either raises it, and the
files of the old format are never found again (they go away with their age)."""

SUFFIX = ".npz"
DIGEST_BYTES = hashlib.sha256().digest_size
MAX_VECTORS = 4096
"""Vectors in one file. The ones of the build that wrote it come first, older ones fill what is
left, so a developer who goes between two sets of examples keeps both until the file is full."""
MAX_FILE_BYTES = 64 * 1024 * 1024
"""A file bigger than this is not read: the cache never writes one (4096 vectors of 4096 numbers)."""
MAX_AGE_DAYS = 60
"""A file nobody read or wrote for this long goes: a model, a revision or a cifarium version that
changed leaves its files behind, and nothing else would ever remove them."""
MAX_BYTES = 256_000_000
"""All the files together. The oldest go first when it is passed."""
UNIT_TOLERANCE = 1e-3
"""How far from 1 the length of a stored vector may be. The bench scales every vector to unit
length; a file with others was not written by this code or was damaged."""
STALE_TEMPORARY_S = 24 * 3600


def text_hash(text: str) -> bytes:
    """What stands for a text in a file: the text itself never reaches the disk."""
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).digest()


@functools.cache
def page_digest() -> str:
    """A hash of the files of the bench page: it pools, prefixes and scales the vectors, so
    a change of it changes the vector without any version moving (the package is 0.0.x
    for as long as it is developed)."""
    folder = paths.web_dir() / "bench"
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(folder).as_posix().encode("utf-8") + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def components(candidate: catalog.Candidate, variant: str = "cpu") -> dict[str, Any]:
    """Everything a vector depends on, but the text.

    The variant says where the job runs (``cpu``, later ``gpu``) and the dtype it runs in, so
    a vector of one never stands in for the other; the files are the ones the library loads,
    the tokenizer and the configuration among them. The number of threads is not here: the page
    picks it, and it is known only after the run (ARCHITECTURE.md, section 9).
    """
    entry, recipe = candidate.entry, candidate.recipe
    chosen = entry.variants[variant]
    return {
        "format": FORMAT,
        "model": entry.repo,
        "revision": entry.revision,
        "variant": {"name": chosen.name, "device": chosen.device, "dtype": chosen.dtype},
        "files": dict(sorted(entry.files(variant).items())),
        "recipe": {"kind": recipe.kind, "params": recipe.params},
        "cifarium": cifarium.__version__,
        "engine": {"transformers.js": vendor.TRANSFORMERS_VERSION, "onnxruntime-web": vendor.ORT_WEB_VERSION},
        "page": page_digest(),
    }


def key_text(candidate: catalog.Candidate, variant: str = "cpu") -> str:
    """The key: :func:`components` as one written text. A file holds its own, and is used only
    when it is the one asked for, whatever its name says."""
    return json.dumps(components(candidate, variant), sort_keys=True, separators=(",", ":"))


class Cache:
    """The folder of the cache. Nothing is read or written until it is asked to."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def path_of(self, candidate: catalog.Candidate, variant: str = "cpu") -> Path:
        """The file of a key: the id of the model, so a person can tell what is whose, and a
        piece of the key's hash. The hash of a key that is read from the file settles the rest."""
        name = re.sub(r"[^A-Za-z0-9._-]", "-", candidate.entry.id)
        digest = hashlib.sha256(key_text(candidate, variant).encode("utf-8")).hexdigest()
        return self.folder / f"{name}-{digest[:20]}{SUFFIX}"

    def load(self, candidate: catalog.Candidate, variant: str = "cpu") -> dict[bytes, np.ndarray]:
        """The vectors kept under the key, by the hash of their text. Empty when there is no file
        and when there is one that is not usable; this never raises, because the cache is an
        optimisation and a build must not fail on it. A file that was used is marked as such."""
        path = self.path_of(candidate, variant)
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                return {}
            with np.load(path, allow_pickle=False) as data:
                if str(data["meta"]) != key_text(candidate, variant):
                    return {}
                texts, vectors = data["texts"], data["vectors"]
            if not _usable(texts, vectors, candidate.recipe.params.get("dimensions")):
                return {}
            entries = {bytes(row): vector for row, vector in zip(texts, vectors)}
        except Exception:  # noqa: BLE001
            # Whatever numpy, zipfile or the disk raise for a missing, cut, foreign or garbled file:
            # the file is not used, and the next write replaces it.
            return {}
        _touch(path)
        return entries

    def store(self, candidate: catalog.Candidate, current: Mapping[bytes, np.ndarray],
              variant: str = "cpu") -> None:
        """Keep ``current`` (every text of this build, in the order of the examples) and, after it,
        what the file held before, up to :data:`MAX_VECTORS`. Written to a temporary file and
        renamed, so a reader never sees half a file and two builds at once leave one whole file.

        Nothing is written when one of the vectors is not usable: it would be thrown away at the
        next read, and the rest would go with it. Raises ``OSError`` when the disk refuses.
        """
        entries = dict(current)
        for digest, vector in self.load(candidate, variant).items():
            entries.setdefault(digest, vector)
        kept = list(entries.items())[:MAX_VECTORS]
        if not kept:
            return
        texts =np.frombuffer(b"".join(digest for digest, _ in kept), dtype=np.uint8).reshape(-1, DIGEST_BYTES)
        vectors = np.stack([np.asarray(vector, dtype="<f4") for _, vector in kept])
        if not _usable(texts, vectors, candidate.recipe.params.get("dimensions")):
            return
        path = self.path_of(candidate, variant)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            with open(temporary, "wb") as handle:
                np.savez(handle, meta=np.array(key_text(candidate, variant)), texts=texts, vectors=vectors)
            os.replace(temporary, path)
        except OSError:
            _remove(temporary)
            raise

    def prune(self, now: float | None = None) -> int:
        """Remove the files nobody used for :data:`MAX_AGE_DAYS`, then the oldest until the rest fits
        in :data:`MAX_BYTES`, and the leftovers of a write that was cut. Only files of the cache's
        own names are looked at. Returns how many went; a folder that cannot be read is left alone."""
        now = time.time() if now is None else now
        removed = 0
        kept: list[tuple[float, int, Path]] = []
        try:
            files = [(path, path.stat()) for path in self.folder.iterdir() if path.is_file()]
        except OSError:
            return 0
        for path, stat in files:
            age = now - stat.st_mtime
            if path.name.endswith(".tmp") and f"{SUFFIX}." in path.name:
                if age > STALE_TEMPORARY_S:
                    removed += _remove(path)
            elif path.suffix == SUFFIX:
                if age > MAX_AGE_DAYS * 24 * 3600:
                    removed += _remove(path)
                else:
                    kept.append((stat.st_mtime, stat.st_size, path))
        total = sum(size for _, size, _ in kept)
        for _, size, path in sorted(kept):
            if total <= MAX_BYTES:
                break
            if _remove(path):
                removed += 1
                total -= size
        return removed


def _usable(texts: np.ndarray, vectors: np.ndarray, dimensions: int | None) -> bool:
    """Whether the arrays are what :meth:`Cache.store` writes: one row of each for a text, floats
    that are finite and of unit length, and the number of numbers the catalog promises."""
    if not (texts.dtype == np.uint8 and texts.ndim == 2 and texts.shape[1] == DIGEST_BYTES):
        return False
    if not (vectors.dtype == np.dtype("<f4") and vectors.ndim == 2 and len(vectors) == len(texts)):
        return False
    if not 0 < len(vectors) <= MAX_VECTORS or (dimensions is not None and vectors.shape[1] != dimensions):
        return False
    if not np.isfinite(vectors).all():
        return False
    return bool(np.all(np.abs(np.linalg.norm(vectors, axis=1) - 1.0) <= UNIT_TOLERANCE))


def _touch(path: Path) -> None:
    try:
        os.utime(path)
    except OSError:
        pass


def _remove(path: Path) -> int:
    """1 if the file went, 0 if it could not be removed."""
    try:
        path.unlink()
    except OSError:
        return 0
    return 1

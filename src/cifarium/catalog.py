"""The model catalog: candidate models with pinned versions, files and sizes.

One YAML file per model in the ``catalog/`` folder of the package, named after the
entry's ``id``. ARCHITECTURE.md, section 4, describes the format and the choices
behind it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import TASKS, TaskFileError, describe, did_you_mean, read_yaml

CATALOG_DIR = Path(__file__).resolve().parent / "catalog"

# `license: permissive` in the settings lets through only these licenses: the ones that allow
# commercial use with no terms to accept beyond keeping the notice. A model that allows commercial
# use under terms of its own (Gemma) is not among them; `license: any` takes it.
PERMISSIVE_LICENSES = frozenset({"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause"})

KEYS = ("id", "repo", "revision", "license", "license_source", "languages",
        "common_files", "variants", "recipes")
VARIANT_KEYS = ("device", "dtype", "files")
LANGUAGES = ("en", "multi")
DEVICES = {"cpu": "wasm", "gpu": "webgpu"}  # variant name -> Transformers.js device
DTYPES = ("fp32", "fp16", "q8", "int8", "uint8", "q4", "q4f16", "bnb4")
# Recipe kind -> its parameters: allowed values, or the type of a free value, and the default.
RECIPES: dict[str, dict[str, tuple[Any, Any]]] = {
    "embed-centroid": {"pooling": (("mean", "cls", "none"), None), "prefix": (str, "")},
    "zero-shot": {},
}

_ID = re.compile(r"[a-z0-9][a-z0-9.-]*")
_REPO = re.compile(r"[\w.-]+/[\w.-]+")
_HASH = re.compile(r"[0-9a-f]{40}")
_LICENSE = re.compile(r"[a-z0-9][a-z0-9.+-]*")


class CatalogError(TaskFileError):
    """A catalog entry cannot be used."""


@dataclass(frozen=True)
class Variant:
    name: str
    device: str
    dtype: str
    files: dict[str, int]
    """Path inside the repository -> size in bytes."""


@dataclass(frozen=True)
class Recipe:
    task: str
    kind: str
    params: dict[str, Any]


@dataclass(frozen=True)
class Entry:
    path: Path
    id: str
    repo: str
    revision: str
    license: str
    license_source: str
    languages: tuple[str, ...]
    common_files: dict[str, int]
    variants: dict[str, Variant]
    recipes: tuple[Recipe, ...]

    def files(self, variant: str = "cpu") -> dict[str, int]:
        """Every file a visitor downloads for ``variant``, with sizes."""
        return {**self.common_files, **self.variants[variant].files}

    def first_load_bytes(self, variant: str = "cpu") -> int:
        """Weight of the model on a visitor's first visit, without the engine."""
        return sum(self.files(variant).values())


@dataclass(frozen=True)
class Candidate:
    """One way to do a task: a model and the recipe that uses it."""

    entry: Entry
    recipe: Recipe


def is_permissive(license: str) -> bool:
    """Whether a model with this license passes ``license: permissive`` in the settings."""
    return license in PERMISSIVE_LICENSES


def load(folder: Path | None = None) -> tuple[Entry, ...]:
    """Read and check every entry, in file name order. Raises :class:`CatalogError`."""
    folder = folder or CATALOG_DIR
    paths = sorted(folder.glob("*.yaml"))
    if not paths:
        raise CatalogError(folder, ["no catalog entries (*.yaml) in this folder"])
    return tuple(read_entry(path) for path in paths)


def candidates(task: str, entries: tuple[Entry, ...] | None = None) -> tuple[Candidate, ...]:
    """Every model and recipe that does ``task``, in catalog order.

    Filtering by license and size and sorting by weight belong to the selection step.
    """
    entries = load() if entries is None else entries
    return tuple(Candidate(entry, recipe) for entry in entries for recipe in entry.recipes
                 if recipe.task == task)


def read_entry(path: Path) -> Entry:
    """Read and check one catalog file. Raises :class:`CatalogError`."""
    path = Path(path)
    data = read_yaml(path, CatalogError)
    if not isinstance(data, dict):
        raise CatalogError(path, ["must be a group of `key: value` settings"])
    return _Checker(data, path).run()


class _Checker:
    def __init__(self, data: dict, path: Path) -> None:
        self.data = data
        self.path = path
        self.problems: list[str] = []

    def add(self, where: str, message: str) -> None:
        self.problems.append(f"{where}: {message}")

    def run(self) -> Entry:
        for key in self.data:
            if key not in KEYS:
                self.add(str(key), "unknown key" + (did_you_mean(str(key), KEYS)
                                                    or f"; the keys are {', '.join(KEYS)}"))
        for key in KEYS:
            if self.data.get(key) is None:
                self.add(key, "missing")

        entry_id = self._text("id", _ID, "lowercase letters, digits, dots and dashes")
        if entry_id is not None and entry_id != self.path.stem:
            self.add("id", f"must match the file name {self.path.stem}; got {describe(entry_id)}")
        repo = self._text("repo", _REPO, "organisation/model")
        revision = self._text("revision", _HASH, "the full commit hash, 40 hex characters")
        license_ = self._text("license", _LICENSE, "a lowercase license id, like apache-2.0")
        source = self._text("license_source", _REPO, "organisation/model")
        languages = self._languages()
        common = self._files("common_files", self.data.get("common_files"))
        variants = self._variants()
        recipes = self._recipes()

        if self.problems:
            raise CatalogError(self.path, self.problems)
        return Entry(self.path.resolve(), entry_id, repo, revision, license_, source,
                     languages, common, variants, recipes)

    def _text(self, key: str, pattern: re.Pattern[str], shape: str) -> str | None:
        value = self.data.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not pattern.fullmatch(value):
            self.add(key, f"must be {shape}; got {describe(value)}")
            return None
        return value

    def _languages(self) -> tuple[str, ...]:
        value = self.data.get("languages")
        if value is None:
            return ()
        if (not isinstance(value, list) or not value
                or not all(isinstance(item, str) and item in LANGUAGES for item in value)):
            self.add("languages", f"must be a list of {' or '.join(LANGUAGES)}, like [en]; "
                     f"got {describe(value)}")
            return ()
        return tuple(value)

    def _files(self, where: str, value: Any) -> dict[str, int]:
        if value is None:
            return {}
        if not isinstance(value, dict) or not value:
            self.add(where, f"must list files as `path: bytes`; got {describe(value)}")
            return {}
        files: dict[str, int] = {}
        for name, size in value.items():
            if (not isinstance(name, str) or name.startswith("/") or "\\" in name
                    or ".." in name.split("/")):
                self.add(where, f"{describe(name)} is not a path inside the repository")
            elif isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                self.add(f"{where}.{name}", f"must be a positive whole number of bytes; got {describe(size)}")
            else:
                files[name] = size
        return files

    def _variants(self) -> dict[str, Variant]:
        value = self.data.get("variants")
        if value is None:
            return {}
        if not isinstance(value, dict) or not value:
            self.add("variants", f"must be a group of variants, like `cpu:`; got {describe(value)}")
            return {}
        variants: dict[str, Variant] = {}
        for name, body in value.items():
            where = f"variants.{name}"
            if name not in DEVICES:
                self.add(where, f"unknown variant; the variants are {', '.join(DEVICES)}")
                continue
            if not isinstance(body, dict):
                self.add(where, f"must be a group of keys: {', '.join(VARIANT_KEYS)}")
                continue
            for key in body:
                if key not in VARIANT_KEYS:
                    self.add(f"{where}.{key}", "unknown key" + did_you_mean(str(key), VARIANT_KEYS))
            for key in VARIANT_KEYS:
                if body.get(key) is None:
                    self.add(f"{where}.{key}", "missing")
            device, dtype = body.get("device"), body.get("dtype")
            if device is not None and device != DEVICES[name]:
                self.add(f"{where}.device", f"must be {DEVICES[name]}; got {describe(device)}")
            if dtype is not None and dtype not in DTYPES:
                self.add(f"{where}.dtype", f"unknown dtype {describe(dtype)}; "
                         f"the dtypes are {', '.join(DTYPES)}")
            files = self._files(f"{where}.files", body.get("files"))
            variants[name] = Variant(name, DEVICES[name], dtype, files)
        return variants

    def _recipes(self) -> tuple[Recipe, ...]:
        value = self.data.get("recipes")
        if value is None:
            return ()
        if not isinstance(value, list):
            self.add("recipes", f"must be a list of recipes; got {describe(value)}")
            return ()
        if not value:
            self.add("recipes", "must list at least one recipe")
            return ()
        recipes: list[Recipe] = []
        for number, item in enumerate(value, start=1):
            where = f"recipes, item {number}"
            if not isinstance(item, dict):
                self.add(where, f"must be a group of keys: task, kind and parameters; got {describe(item)}")
                continue
            task, kind = item.get("task"), item.get("kind")
            if task not in TASKS:
                self.add(f"{where}: task", f"must be one of {', '.join(TASKS)}; got {describe(task)}")
            if not isinstance(kind, str) or kind not in RECIPES:
                hint = did_you_mean(kind, list(RECIPES)) if isinstance(kind, str) else ""
                self.add(f"{where}: kind", f"unknown recipe {describe(kind)}{hint}; "
                         f"the recipes are {', '.join(RECIPES)}")
                continue
            allowed = RECIPES[kind]
            params: dict[str, Any] = {}
            for key, param in item.items():
                if key in ("task", "kind"):
                    continue
                if key not in allowed:
                    names = ", ".join(allowed) or "none"
                    self.add(f"{where}: {key}", f"unknown parameter of {kind}; its parameters: {names}")
                    continue
                params[key] = param
            for key, (rule, default) in allowed.items():
                param = params.get(key, default)
                if isinstance(rule, tuple):
                    if param not in rule:
                        self.add(f"{where}: {key}", f"must be {', '.join(rule[:-1])} or {rule[-1]}; "
                                 f"got {describe(param)}")
                elif not isinstance(param, rule):
                    self.add(f"{where}: {key}", f"must be text; got {describe(param)}")
                params[key] = param
            if task in TASKS:
                recipes.append(Recipe(task, kind, params))
        return tuple(recipes)

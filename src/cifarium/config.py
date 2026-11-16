"""Reading and checking the settings file, ``cifarium.yaml`` by default.

Paths inside the file are relative to the file itself, not to the current folder.
Every problem found is collected, so one run shows all of them at once.
"""

from __future__ import annotations

import difflib
import json
import math
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_FILE = "cifarium.yaml"

TASKS = ("classify", "search", "summarize")
SUPPORTED_TASKS = ("classify",)
LICENSES = ("commercial", "any")
HOSTS = ("bundle", "hub")

DEFAULT_PASS_RATE = 0.85
DEFAULT_SIMILARITY = 0.75
DEFAULT_MAX_SIZE_MB = 300.0
DEFAULT_OUTPUT = "./cifarium-out"

KEYS = (
    "task",
    "examples",
    "labels",
    "documents",
    "pass_rate",
    "similarity",
    "limits",
    "license",
    "host",
    "output",
)
LIMIT_KEYS = ("max_size_mb", "max_latency_ms", "min_tokens_per_sec")

# Keys that mean something for one task only, and the task.
_ONLY_FOR = {
    "labels": "classify",
    "documents": "search",
    "similarity": "summarize",
    "limits.min_tokens_per_sec": "summarize",
}

# More than this many problems are summarised in one line.
MAX_SHOWN_PROBLEMS = 20


class TaskFileError(Exception):
    """A settings, examples or catalog file cannot be used. ``problems`` lists every issue."""

    def __init__(self, path: Path, problems: list[str]) -> None:
        self.path = Path(path)
        self.problems = list(problems)
        super().__init__(format_problems(self.path, self.problems))


class ConfigError(TaskFileError):
    """The settings file cannot be used."""


def format_problems(path: Path, problems: list[str]) -> str:
    if len(problems) == 1:
        return f"{path}: {problems[0]}"
    shown = problems[:MAX_SHOWN_PROBLEMS]
    lines = [f"{path} has {len(problems)} problems:"]
    lines += [f"  - {problem}" for problem in shown]
    if len(problems) > len(shown):
        lines.append(f"  ... and {len(problems) - len(shown)} more")
    return "\n".join(lines)


def read_text(path: Path, error: type[TaskFileError]) -> str:
    """Read a UTF-8 file, with or without the byte order mark Windows editors add."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise error(path, ["file not found"]) from None
    except OSError as err:
        reason = "it is a folder" if path.is_dir() else (err.strerror or str(err))
        raise error(path, [f"cannot read the file: {reason}"]) from None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as err:
        line = data[: err.start].count(b"\n") + 1
        raise error(path, [f"line {line}: not UTF-8 text; save the file as UTF-8"]) from None


def label_text(name: str) -> str:
    """One spelling for a label: trimmed and in Unicode NFC, so that "й" typed as one
    character and as two compares equal."""
    return unicodedata.normalize("NFC", name.strip())


def describe(value: Any) -> str:
    """How a parsed value looks to the person who wrote it."""
    if value is None:
        return "nothing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "a list"
    if isinstance(value, dict):
        return "a group of keys"
    return str(value)


def did_you_mean(word: str, options: tuple[str, ...] | list[str]) -> str:
    """``"; did you mean X?"`` when one of the options is close, otherwise ``""``."""
    match = difflib.get_close_matches(word, list(options), n=1, cutoff=0.6)
    if not match:
        lowered = {option.lower(): option for option in options}
        if word.lower() in lowered:
            match = [lowered[word.lower()]]
    return f"; did you mean {match[0]}?" if match else ""


@dataclass(frozen=True)
class Label:
    name: str
    description: str | None = None


@dataclass(frozen=True)
class Limits:
    max_size_mb: float = DEFAULT_MAX_SIZE_MB
    max_latency_ms: float | None = None
    min_tokens_per_sec: float | None = None


@dataclass(frozen=True)
class Config:
    """Checked settings. Paths are absolute."""

    path: Path
    task: str
    examples: Path
    labels: tuple[Label, ...] | None = None
    documents: Path | None = None
    pass_rate: float = DEFAULT_PASS_RATE
    similarity: float | None = None
    limits: Limits = field(default_factory=Limits)
    license: str = "commercial"
    host: str = "bundle"
    output: Path = Path(DEFAULT_OUTPUT)


class _Loader(yaml.SafeLoader):
    """Safe YAML loader that refuses a key written twice instead of keeping the last one."""


def _construct_mapping(loader: _Loader, node: yaml.MappingNode) -> dict:
    seen: set = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":  # `<<:` is allowed to repeat keys
            continue
        key = loader.construct_object(key_node, deep=True)
        try:
            repeated = key in seen
            seen.add(key)
        except TypeError:  # an unhashable key; construct_mapping reports it below
            continue
        if repeated:
            raise yaml.constructor.ConstructorError(
                None, None, f"the key {key} is written twice", key_node.start_mark
            )
    return loader.construct_mapping(node, deep=True)


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def read_yaml(path: Path, error: type[TaskFileError]) -> Any:
    """Read a UTF-8 YAML file safely; a key written twice is an error."""
    text = read_text(path, error)
    try:
        return yaml.load(text, Loader=_Loader)  # noqa: S506 - _Loader is a SafeLoader
    except yaml.MarkedYAMLError as err:
        mark = err.problem_mark or err.context_mark
        where = f"line {mark.line + 1}, column {mark.column + 1}: " if mark else ""
        problem = err.problem or err.context or "unreadable"
        raise error(path, [f"{where}not valid YAML: {problem}"]) from None
    except yaml.YAMLError as err:
        raise error(path, [f"not valid YAML: {err}"]) from None


def load(path: str | os.PathLike[str] = DEFAULT_FILE) -> Config:
    """Read and check the settings file. Raises :class:`ConfigError`."""
    path = Path(path)
    return parse(read_yaml(path, ConfigError), path)


def parse(data: Any, path: Path) -> Config:
    """Check settings already parsed from YAML. ``path`` anchors relative paths."""
    path = Path(path)
    if data is None:
        raise ConfigError(path, ["the file is empty; `cifarium init` writes a starter one"])
    if not isinstance(data, dict):
        raise ConfigError(path, ["must be a list of `key: value` settings, like `task: classify`"])
    return _Checker(data, path).run()


class _Checker:
    def __init__(self, data: dict, path: Path) -> None:
        self.data = data
        self.path = path
        self.base = path.resolve().parent
        self.problems: list[str] = []
        self.task: str | None = None

    def add(self, key: str, message: str) -> None:
        self.problems.append(f"{key}: {message}")

    def run(self) -> Config:
        for key in self.data:
            if key not in KEYS:
                name = str(key)
                self.add(name, "unknown key" + (did_you_mean(name, KEYS) or f"; the keys are {', '.join(KEYS)}"))

        self.task = self._task()
        examples = self._path("examples", required=True, must_exist=True)
        labels = self._labels() if self._applies("labels") else None
        documents = self._path("documents", required=True, must_exist=True) if self._applies("documents") else None
        pass_rate = self._fraction("pass_rate", DEFAULT_PASS_RATE)
        similarity = None
        if self._applies("similarity"):
            similarity = self._fraction("similarity", DEFAULT_SIMILARITY)
        limits = self._limits()
        license_ = self._choice("license", LICENSES, "commercial")
        host = self._choice("host", HOSTS, "bundle")
        output = self._path("output", default=DEFAULT_OUTPUT)

        if self.problems:
            raise ConfigError(self.path, self.problems)
        assert self.task is not None and examples is not None and output is not None
        return Config(
            path=self.path.resolve(),
            task=self.task,
            examples=examples,
            labels=labels,
            documents=documents,
            pass_rate=pass_rate,
            similarity=similarity,
            limits=limits,
            license=license_,
            host=host,
            output=output,
        )

    def _applies(self, key: str) -> bool:
        """Whether to read ``key``. Reports a key that belongs to another task.

        When ``task`` itself is broken there is nothing to compare with, so the
        task-specific keys are skipped silently until it is fixed.
        """
        task = _ONLY_FOR[key]
        if self.task == task:
            return True
        if self.task is not None and self._get(key) is not None:
            self.add(key, f"used only by task {task}; this file has task {self.task}")
        return False

    def _get(self, key: str) -> Any:
        if key.startswith("limits."):
            limits = self.data.get("limits")
            return limits.get(key.split(".", 1)[1]) if isinstance(limits, dict) else None
        return self.data.get(key)

    def _task(self) -> str | None:
        value = self.data.get("task")
        choices = ", ".join(TASKS)
        if value is None:
            self.add("task", "missing; write `task: classify`")
        elif not isinstance(value, str):
            self.add("task", f"must be one of {choices}; got {describe(value)}")
        elif value in SUPPORTED_TASKS:
            return value
        elif value in TASKS:
            # Returning None skips the keys of that task, so only this message shows.
            self.add("task", f"{value} is not supported yet; this version supports only classify")
        else:
            hint = did_you_mean(value, TASKS) or f"; the tasks are {choices}"
            self.add("task", f"unknown task {describe(value)}{hint}")
        return None

    def _path(
        self, key: str, *, required: bool = False, must_exist: bool = False, default: str | None = None
    ) -> Path | None:
        value = self.data.get(key)
        if value is None:
            if required:
                self.add(key, f"missing; write the path to the file, like `{key}: ./{key}.jsonl`")
                return None
            if default is None:
                return None
            value = default
        if not isinstance(value, str) or not value.strip():
            self.add(key, f"must be a path; got {describe(value)}")
            return None
        resolved = (self.base / Path(value.strip()).expanduser()).resolve()
        if must_exist and not resolved.is_file():
            if resolved.is_dir():
                self.add(key, f"{resolved} is a folder; name the file")
            else:
                self.add(key, f"no file at {resolved}; paths are relative to {self.path.name}")
            return None
        if not must_exist and resolved.is_file():
            self.add(key, f"{resolved} is a file; name a folder")
            return None
        if not must_exist and self.base.is_relative_to(resolved):
            # The build would land among the project's own files.
            self.add(key, f"{resolved} holds {self.path.name} itself; name a separate folder, "
                     f"like `{key}: {DEFAULT_OUTPUT}`")
            return None
        return resolved

    def _fraction(self, key: str, default: float) -> float:
        value = self.data.get(key)
        if value is None:
            return default
        example = f"like {default}"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.add(key, f"must be a number between 0 and 1, {example}; got {describe(value)}")
        elif 1 < value <= 100:
            self.add(key, f"must be a number between 0 and 1; for {value:g}% write {value / 100:g}")
        elif not 0 < value <= 1:
            self.add(key, f"must be a number between 0 and 1, {example}; got {value:g}")
        else:
            return float(value)
        return default

    def _choice(self, key: str, choices: tuple[str, ...], default: str) -> str:
        value = self.data.get(key)
        if value is None:
            return default
        if isinstance(value, str) and value in choices:
            return value
        hint = did_you_mean(value, choices) if isinstance(value, str) else ""
        self.add(key, f"must be {' or '.join(choices)}; got {describe(value)}{hint}")
        return default

    def _limits(self) -> Limits:
        value = self.data.get("limits")
        if value is None:
            return Limits()
        if not isinstance(value, dict):
            self.add("limits", f"must be a group of keys: {', '.join(LIMIT_KEYS)}; got {describe(value)}")
            return Limits()
        for key in value:
            if key not in LIMIT_KEYS:
                name = str(key)
                self.add(f"limits.{name}", "unknown key" + (
                    did_you_mean(name, LIMIT_KEYS) or f"; the keys are {', '.join(LIMIT_KEYS)}"))
        units = {
            "max_size_mb": "megabytes",
            "max_latency_ms": "milliseconds",
            "min_tokens_per_sec": "tokens per second",
        }
        numbers: dict[str, float | None] = {}
        for key in LIMIT_KEYS:
            full = f"limits.{key}"
            if full in _ONLY_FOR and not self._applies(full):
                continue
            number = value.get(key)
            if number is None:
                continue
            if (isinstance(number, bool) or not isinstance(number, (int, float))
                    or not math.isfinite(number) or number <= 0):
                self.add(full, f"must be a positive number of {units[key]}; got {describe(number)}")
            else:
                numbers[key] = float(number)
        return Limits(**numbers)

    def _labels(self) -> tuple[Label, ...] | None:
        value = self.data.get("labels")
        if value is None:
            return None
        pairs: list[tuple[str, Any, Any]] = []  # (where, name, description)
        if isinstance(value, dict):
            pairs = [(f"labels.{name}", name, description) for name, description in value.items()]
        elif isinstance(value, list):
            for number, item in enumerate(value, start=1):
                where = f"labels, item {number}"
                if isinstance(item, dict) and len(item) == 1:
                    (name, description), = item.items()
                    pairs.append((where, name, description))
                elif isinstance(item, dict):
                    self.add(where, "write one label per line: `- name: description`")
                else:
                    pairs.append((where, item, None))
        else:
            self.add("labels", "must be a list of label names, or names with descriptions; "
                     f"got {describe(value)}")
            return None

        labels: list[Label] = []
        names: dict[str, str] = {}  # casefolded name -> name as written
        for where, name, description in pairs:
            if not isinstance(name, str):
                if isinstance(name, (bool, int, float)):
                    self.add(where, f"label names are text; YAML read {describe(name)}, "
                             "put the name in quotes")
                else:
                    self.add(where, f"label name must be text; got {describe(name)}")
                continue
            name = label_text(name)
            if not name:
                self.add(where, "label name is empty")
                continue
            twin = names.get(name.casefold())
            if twin == name:
                self.add(where, f"label {describe(name)} is listed twice")
                continue
            if twin is not None:
                self.add(where, f"label {describe(name)} differs from {describe(twin)} only in "
                         "letter case; keep one spelling")
                continue
            if description is not None and not isinstance(description, str):
                self.add(where, f"description must be text; got {describe(description)}")
                continue
            names[name.casefold()] = name
            labels.append(Label(name, (description or "").strip() or None))
        if len(pairs) < 2:
            self.add("labels", "classification needs at least two labels")
        return tuple(labels)

"""Reading and checking the examples file, ``examples.jsonl`` by default.

One JSON object per line: ``{"input": "...", "expected": "..."}``. For classification
``expected`` is the label. Blank lines and any other fields are ignored.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .config import Config, Label, TaskFileError, describe, did_you_mean, read_text

DEFAULT_FILE = "examples.jsonl"

_SHAPE = '{"input": "...", "expected": "..."}'


class ExamplesError(TaskFileError):
    """The examples file cannot be used."""


@dataclass(frozen=True)
class Example:
    line: int
    """Line number in the file, for messages."""
    input: str
    expected: str


@dataclass(frozen=True)
class Examples:
    path: Path
    items: tuple[Example, ...]
    labels: tuple[Label, ...]
    """Classification labels: the ones from the settings, or else the ones met in the
    examples, in order of first appearance."""


def load(config: Config) -> Examples:
    """Read the examples named in the settings and check them against the settings."""
    if config.task != "classify":  # the settings check lets only classify through for now
        raise NotImplementedError(f"examples for task {config.task}")
    return read(config.examples, config.labels, settings_name=config.path.name)


def read(
    path: str | os.PathLike[str],
    labels: tuple[Label, ...] | None = None,
    *,
    settings_name: str = "the settings",
) -> Examples:
    """Read and check classification examples. Raises :class:`ExamplesError`.

    With ``labels`` every ``expected`` must be one of them and every label needs an
    example. Without, the labels are taken from the examples.
    """
    path = Path(path)
    text = read_text(path, ExamplesError)
    problems: list[str] = []
    items: list[Example] = []
    first_line: dict[str, int] = {}

    for number, raw in enumerate(text.split("\n"), start=1):
        line = raw.strip()
        if not line:
            continue
        if not items and not problems and line.startswith("["):
            problems.append(f"line {number}: write one example per line, not one JSON list; "
                            f"each line is {_SHAPE}")
            break
        try:
            data = json.loads(line)
        except json.JSONDecodeError as err:
            problems.append(f"line {number}: not valid JSON ({err.msg}, column {err.colno}); "
                            f"each line is one whole {_SHAPE}")
            continue
        if not isinstance(data, dict):
            problems.append(f"line {number}: must be an object like {_SHAPE}; got {describe(data)}")
            continue
        text_in = _text(data, "input", number, problems)
        expected = _text(data, "expected", number, problems, label=True)
        if text_in is None or expected is None:
            continue
        key = text_in.strip()
        if key in first_line:
            problems.append(f'line {number}: the same "input" as line {first_line[key]}; '
                            "keep one of them")
            continue
        first_line[key] = number
        items.append(Example(number, text_in, expected.strip()))

    if not items and not problems:
        problems.append(f"no examples; write one per line: {_SHAPE}")
    if problems:
        raise ExamplesError(path, problems)

    final = _check_labels(items, labels, settings_name, problems)
    if problems:
        raise ExamplesError(path, problems)
    return Examples(path=path.resolve(), items=tuple(items), labels=final)


def _text(data: dict, key: str, number: int, problems: list[str], *, label: bool = False) -> str | None:
    value = data.get(key)
    what = "one label as text" if label else "text"
    if key not in data:
        hint = did_you_mean(key, [k for k in data if isinstance(k, str)])
        problems.append(f'line {number}: "{key}" is missing{hint}')
    elif not isinstance(value, str):
        quote = f', like "{value}"' if isinstance(value, (int, float)) and not isinstance(value, bool) else ""
        problems.append(f'line {number}: "{key}" must be {what}{quote}; got {describe(value)}')
    elif not value.strip():
        problems.append(f'line {number}: "{key}" is empty')
    else:
        return value
    return None


def _check_labels(
    items: list[Example],
    labels: tuple[Label, ...] | None,
    settings_name: str,
    problems: list[str],
) -> tuple[Label, ...]:
    if labels is None:
        names = list(dict.fromkeys(example.expected for example in items))
        if len(names) < 2:
            problems.append(f"every example has the label {describe(names[0])}; "
                            "classification needs at least two labels")
        return tuple(Label(name) for name in names)

    names = [label.name for label in labels]
    used: set[str] = set()
    for example in items:
        if example.expected in names:
            used.add(example.expected)
        else:
            problems.append(f"line {example.line}: label {describe(example.expected)} is not listed "
                            f"under labels in {settings_name}{did_you_mean(example.expected, names)}")
    for name in names:
        if name not in used:
            problems.append(f"label {describe(name)} from {settings_name} has no examples; "
                            "add at least one, or remove the label")
    return labels

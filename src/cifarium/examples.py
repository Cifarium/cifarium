"""Reading and checking the examples file, ``examples.jsonl`` by default.

One JSON object per line: ``{"input": "...", "expected": "..."}``. For classification
``expected`` is the label. Blank lines and any other fields are ignored.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import Config, Label, TaskFileError, describe, did_you_mean, label_text, read_text

# Each example is scored on label centres computed without it, so a label with a
# single example has no centre left and that example always fails.
MIN_EXAMPLES_PER_LABEL = 2

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

    With ``labels`` every ``expected`` must be one of them. Without, the labels are
    taken from the examples. Either way every label needs at least two examples.
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
        items.append(Example(number, text_in, label_text(expected)))

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
    counts = Counter(example.expected for example in items)
    first = {}
    for example in items:
        first.setdefault(example.expected, example.line)

    if labels is None:
        names = list(counts)  # Counter keeps the order of first appearance
        if len(names) < 2:
            problems.append(f"every example has the label {describe(names[0])}; "
                            "classification needs at least two labels")
        seen: dict[str, str] = {}
        for name in names:
            twin = seen.setdefault(name.casefold(), name)
            if twin != name:
                problems.append(f"line {first[name]}: label {describe(name)} differs from "
                                f"{describe(twin)} (line {first[twin]}) only in letter case; "
                                "keep one spelling")
        final = tuple(Label(name) for name in names)
    else:
        names = [label.name for label in labels]
        for example in items:
            if example.expected not in names:
                problems.append(f"line {example.line}: label {describe(example.expected)} is not "
                                f"listed under labels in {settings_name}"
                                f"{did_you_mean(example.expected, names)}")
        for name in names:
            if name not in counts:
                problems.append(f"label {describe(name)} from {settings_name} has no examples; "
                                f"add at least {MIN_EXAMPLES_PER_LABEL}, or remove the label")
        final = labels

    known = [label.name for label in final]
    for label in final:
        if counts[label.name] == 1:
            others = [name for name in known if name != label.name]
            problems.append(f"line {first[label.name]}: label {describe(label.name)} has only one "
                            f"example; every label needs at least {MIN_EXAMPLES_PER_LABEL}"
                            f"{did_you_mean(label.name, others)}")
    return final

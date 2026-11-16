"""Settings file: reading and checking, no browser."""

from __future__ import annotations

from pathlib import Path

import pytest

from cifarium import config
from cifarium.config import ConfigError, Label, Limits

BASE = "task: classify\nexamples: ./examples.jsonl\n"


def write(folder: Path, text: str) -> Path:
    (folder / "examples.jsonl").write_text('{"input": "a", "expected": "x"}\n', encoding="utf-8")
    path = folder / "task.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def problems_of(folder: Path, text: str) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        config.load(write(folder, text))
    return caught.value.problems


def test_minimal_file_gets_the_defaults(tmp_path):
    cfg = config.load(write(tmp_path, BASE))

    assert cfg.task == "classify"
    assert cfg.path == (tmp_path / "task.yaml").resolve()
    assert cfg.examples == (tmp_path / "examples.jsonl").resolve()
    assert cfg.labels is None
    assert cfg.documents is None
    assert cfg.pass_rate == 0.85
    assert cfg.similarity is None
    assert cfg.limits == Limits(max_size_mb=300.0, max_latency_ms=None, min_tokens_per_sec=None)
    assert cfg.license == "commercial"
    assert cfg.host == "bundle"
    assert cfg.output == (tmp_path / "cifarium-out").resolve()


def test_every_key_is_read(tmp_path):
    cfg = config.load(write(tmp_path, BASE + """
labels:
  billing: payments and refunds
  bug:
pass_rate: 0.9
limits:
  max_size_mb: 50
  max_latency_ms: 120
license: any
host: hub
output: ./public/model
"""))

    assert cfg.labels == (Label("billing", "payments and refunds"), Label("bug", None))
    assert cfg.pass_rate == 0.9
    assert cfg.limits == Limits(max_size_mb=50.0, max_latency_ms=120.0)
    assert cfg.license == "any"
    assert cfg.host == "hub"
    assert cfg.output == (tmp_path / "public" / "model").resolve()


def test_an_empty_value_means_the_default(tmp_path):
    cfg = config.load(write(tmp_path, BASE + "pass_rate:\nlimits:\nlicense:\noutput:\n"))

    assert cfg.pass_rate == 0.85
    assert cfg.limits == Limits()
    assert cfg.license == "commercial"
    assert cfg.output == (tmp_path / "cifarium-out").resolve()


def test_paths_are_relative_to_the_file_not_the_current_folder(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    write(project, "task: classify\nexamples: examples.jsonl\noutput: ../site/model\n")
    monkeypatch.chdir(tmp_path)

    cfg = config.load(Path("project") / "task.yaml")

    assert cfg.examples == (project / "examples.jsonl").resolve()
    assert cfg.output == (tmp_path / "site" / "model").resolve()


@pytest.mark.parametrize(
    "labels",
    [
        "labels: [billing, bug]",
        "labels:\n  - billing\n  - bug\n",
        "labels:\n  billing:\n  bug:\n",
        "labels:\n  - billing: payments\n  - bug\n",
        "labels:\n  ' billing ': ''\n  bug:\n",
    ],
)
def test_labels_are_a_list_or_names_with_descriptions(tmp_path, labels):
    cfg = config.load(write(tmp_path, BASE + labels))

    assert [label.name for label in cfg.labels] == ["billing", "bug"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("examples: ./examples.jsonl", "task: missing; write `task: classify`"),
        ("task: clasify\nexamples: ./examples.jsonl", 'task: unknown task "clasify"; did you mean classify?'),
        ("task: translate\nexamples: ./examples.jsonl", "; the tasks are classify, search, summarize"),
        ("task: [classify]\nexamples: ./examples.jsonl", "task: must be one of classify, search, summarize"),
        ("task: search\nexamples: ./examples.jsonl", "task: search is not supported yet"),
        ("task: summarize\nexamples: ./examples.jsonl", "task: summarize is not supported yet"),
        ("task: classify", "examples: missing"),
        ("task: classify\nexamples: ./nope.jsonl", "examples: no file at"),
        ("task: classify\nexamples: .", "is a folder; name the file"),
        ("task: classify\nexamples: 5", "examples: must be a path; got 5"),
        (BASE + "pass_rate: 85", "pass_rate: must be a number between 0 and 1; for 85% write 0.85"),
        (BASE + "pass_rate: 0", "pass_rate: must be a number between 0 and 1, like 0.85; got 0"),
        (BASE + "pass_rate: yes", "pass_rate: must be a number between 0 and 1, like 0.85; got true"),
        (BASE + "pass_rate: '0.9'", 'pass_rate: must be a number between 0 and 1, like 0.85; got "0.9"'),
        (BASE + "pass-rate: 0.9", "pass-rate: unknown key; did you mean pass_rate?"),
        (BASE + "colour: red", "colour: unknown key; the keys are task, examples, labels"),
        (BASE + "limits: 300", "limits: must be a group of keys: max_size_mb, max_latency_ms"),
        (BASE + "limits:\n  max_size: 300", "limits.max_size: unknown key; did you mean max_size_mb?"),
        (BASE + "limits:\n  max_size_mb: -5", "limits.max_size_mb: must be a positive number of megabytes"),
        (BASE + "limits:\n  max_latency_ms: fast", "limits.max_latency_ms: must be a positive number"),
        (BASE + "limits:\n  min_tokens_per_sec: 10", "limits.min_tokens_per_sec: used only by task summarize"),
        (BASE + "similarity: 0.8", "similarity: used only by task summarize; this file has task classify"),
        (BASE + "documents: ./docs.jsonl", "documents: used only by task search"),
        (BASE + "license: MIT", 'license: must be commercial or any; got "MIT"'),
        (BASE + "host: cdn", 'host: must be bundle or hub; got "cdn"'),
        (BASE + "host: Hub", "did you mean hub?"),
        (BASE + "output: ./examples.jsonl", "is a file; name a folder"),
        (BASE + "labels: [billing]", "labels: classification needs at least two labels"),
        (BASE + "labels: [billing, bug, billing]", 'labels, item 3: label "billing" is listed twice'),
        (BASE + "labels: [yes, no]", "labels, item 1: label names are text; YAML read true, put the name in quotes"),
        (BASE + "labels: [1, 2]", "YAML read 1, put the name in quotes"),
        (BASE + "labels: billing", "labels: must be a list of label names"),
        (BASE + "labels:\n  billing: 5\n  bug:", "labels.billing: description must be text; got 5"),
        (BASE + "labels:\n  - billing: a\n    bug: b", "labels, item 1: write one label per line"),
        (BASE + "labels: ['', bug]", "labels, item 1: label name is empty"),
    ],
)
def test_each_problem_names_the_key_and_what_is_wrong(tmp_path, text, expected):
    found = problems_of(tmp_path, text)

    assert any(expected in problem for problem in found), found


def test_all_problems_are_reported_at_once(tmp_path):
    with pytest.raises(ConfigError) as caught:
        config.load(write(tmp_path, "task: classify\npass_rate: 2\nhost: cdn\n"))

    assert len(caught.value.problems) == 3
    message = str(caught.value)
    assert message.startswith(f"{tmp_path / 'task.yaml'} has 3 problems:")
    assert "\n  - examples: missing" in message


def test_one_problem_is_one_line(tmp_path):
    path = write(tmp_path, BASE + "host: cdn\n")

    with pytest.raises(ConfigError) as caught:
        config.load(path)

    assert str(caught.value) == f'{path}: host: must be bundle or hub; got "cdn"'


def test_a_key_written_twice_is_an_error_not_a_silent_override(tmp_path):
    found = problems_of(tmp_path, BASE + "pass_rate: 0.9\npass_rate: 0.5\n")

    assert found == ["line 4, column 1: not valid YAML: the key pass_rate is written twice"]


def test_broken_yaml_points_at_the_line(tmp_path):
    found = problems_of(tmp_path, BASE + "labels: [billing, bug\npass_rate: 0.9\n")

    assert len(found) == 1
    assert found[0].startswith("line ")
    assert "not valid YAML" in found[0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "the file is empty; `cifarium init` writes a starter one"),
        ("# only a comment\n", "the file is empty"),
        ("- task: classify\n", "must be a list of `key: value` settings"),
    ],
)
def test_file_that_is_not_settings(tmp_path, text, expected):
    found = problems_of(tmp_path, text)

    assert len(found) == 1
    assert expected in found[0]


def test_missing_file(tmp_path):
    assert problems_of_path(tmp_path / "task.yaml") == ["file not found"]


def test_byte_order_mark_from_windows_editors_is_fine(tmp_path):
    path = write(tmp_path, "")
    path.write_bytes(b"\xef\xbb\xbf" + BASE.encode())

    assert config.load(path).task == "classify"


def test_text_that_is_not_utf8_points_at_the_line(tmp_path):
    path = write(tmp_path, "")
    path.write_bytes(BASE.encode() + "host: bundle # ".encode() + "кэш".encode("cp1251") + b"\n")

    found = problems_of_path(path)

    assert found == ["line 3: not UTF-8 text; save the file as UTF-8"]


def problems_of_path(path: Path) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        config.load(path)
    return caught.value.problems

"""Examples file: reading and checking, no browser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cifarium import config, examples
from cifarium.config import Label
from cifarium.examples import Example, ExamplesError

LABELS = (Label("billing", "payments"), Label("bug"))


def line(text: str, label: str) -> str:
    return json.dumps({"input": text, "expected": label}, ensure_ascii=False)


GOOD = [line("charged twice", "billing"), line("app crashes", "bug"), line("refund please", "billing")]


def write(folder: Path, lines: list[str]) -> Path:
    path = folder / "examples.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def problems_of(path: Path, labels: tuple[Label, ...] | None = None) -> list[str]:
    with pytest.raises(ExamplesError) as caught:
        examples.read(path, labels, settings_name="task.yaml")
    return caught.value.problems


def test_labels_come_from_the_examples_in_order_of_first_appearance(tmp_path):
    result = examples.read(write(tmp_path, GOOD))

    assert result.items == (
        Example(1, "charged twice", "billing"),
        Example(2, "app crashes", "bug"),
        Example(3, "refund please", "billing"),
    )
    assert result.labels == (Label("billing"), Label("bug"))
    assert result.path == (tmp_path / "examples.jsonl").resolve()


def test_labels_from_the_settings_keep_their_descriptions(tmp_path):
    result = examples.read(write(tmp_path, GOOD), LABELS)

    assert result.labels == LABELS


def test_blank_lines_crlf_bom_and_extra_fields_are_fine(tmp_path):
    path = tmp_path / "examples.jsonl"
    body = "\r\n".join([
        '{"input": "charged twice", "expected": " billing ", "id": 7, "source": "mail"}',
        "",
        "   ",
        '{"input": "Счёт пришёл дважды", "expected": "billing"}',
        '{"input": "app crashes", "expected": "bug"}',
    ])
    path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))

    result = examples.read(path)

    assert [item.line for item in result.items] == [1, 4, 5]
    assert result.items[0].expected == "billing"
    assert result.items[1].input == "Счёт пришёл дважды"


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ([GOOD[0], "{'input': 'b', 'expected': 'bug'}"], "line 2: not valid JSON"),
        ([GOOD[0] + ","], "line 1: not valid JSON (Extra data, column"),
        (["{", '  "input": "a",', '  "expected": "bug"', "}"], "each line is one whole"),
        (["[", GOOD[0] + ",", GOOD[1], "]"], "line 1: write one example per line, not one JSON list"),
        (['"just a sentence"'], "line 1: must be an object like"),
        (['{"input": "a"}'], 'line 1: "expected" is missing'),
        (['{"text": "a", "expected": "bug"}'], 'line 1: "input" is missing'),
        (['{"input": "a", "expect": "bug"}'], '"expected" is missing; did you mean expect?'),
        (['{"input": "a", "expected": 1}'], 'line 1: "expected" must be one label as text, like "1"; got 1'),
        (['{"input": "a", "expected": ["bug", "billing"]}'], '"expected" must be one label as text; got a list'),
        (['{"input": 5, "expected": "bug"}'], 'line 1: "input" must be text, like "5"; got 5'),
        (['{"input": "   ", "expected": "bug"}'], 'line 1: "input" is empty'),
        (['{"input": "a", "expected": ""}'], 'line 1: "expected" is empty'),
        ([GOOD[0], line(" charged twice ", "bug")], 'line 2: the same "input" as line 1; keep one of them'),
        ([""], "no examples; write one per line"),
        ([GOOD[0], GOOD[2]], 'every example has the label "billing"; classification needs at least two labels'),
    ],
)
def test_each_problem_names_the_line_and_what_is_wrong(tmp_path, lines, expected):
    found = problems_of(write(tmp_path, lines))

    assert any(expected in problem for problem in found), found


def test_every_expected_must_be_one_of_the_labels(tmp_path):
    lines = [GOOD[0], line("card declined", "biling"), GOOD[1]]

    found = problems_of(write(tmp_path, lines), LABELS)

    assert found == ['line 2: label "biling" is not listed under labels in task.yaml; did you mean billing?']


def test_every_label_needs_an_example(tmp_path):
    labels = LABELS + (Label("account"),)

    found = problems_of(write(tmp_path, GOOD), labels)

    assert found == ['label "account" from task.yaml has no examples; add at least one, or remove the label']


def test_bytes_that_are_not_utf8_point_at_the_line(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_bytes((GOOD[0] + "\n").encode() + line("счёт", "billing").encode("cp1251"))

    assert problems_of(path) == ["line 2: not UTF-8 text; save the file as UTF-8"]


def test_a_long_list_of_problems_is_cut(tmp_path):
    path = write(tmp_path, ["oops"] * 30)

    with pytest.raises(ExamplesError) as caught:
        examples.read(path)

    assert len(caught.value.problems) == 30
    message = str(caught.value)
    assert message.startswith(f"{path} has 30 problems:")
    assert message.count("\n  - ") == config.MAX_SHOWN_PROBLEMS
    assert message.endswith("  ... and 10 more")


def test_missing_file(tmp_path):
    assert problems_of(tmp_path / "examples.jsonl") == ["file not found"]


def test_load_reads_the_file_named_in_the_settings(tmp_path):
    write(tmp_path, GOOD)
    settings = tmp_path / "task.yaml"
    settings.write_text("task: classify\nexamples: ./examples.jsonl\nlabels: [billing, bug, account]\n",
                        encoding="utf-8")
    cfg = config.load(settings)

    with pytest.raises(ExamplesError) as caught:
        examples.load(cfg)

    assert caught.value.problems == [
        'label "account" from task.yaml has no examples; add at least one, or remove the label'
    ]

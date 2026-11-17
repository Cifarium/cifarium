"""`cifarium init`: starter files, no browser."""

from __future__ import annotations

from collections import Counter

import pytest

from cifarium import cli, config, examples


def test_init_writes_starters_that_pass_the_checks_unchanged(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert cli.main(["init"]) == 0

    assert sorted(path.name for path in tmp_path.iterdir()) == ["cifarium.yaml", "examples.jsonl"]
    assert f"Wrote cifarium.yaml and examples.jsonl to {tmp_path}" in capsys.readouterr().out
    cfg = config.load(tmp_path / "cifarium.yaml")
    result = examples.load(cfg)
    assert cfg.task == "classify"
    assert [label.name for label in result.labels] == ["billing", "bug", "account"]
    assert all(label.description for label in result.labels)
    assert len(result.items) >= 10
    assert min(Counter(item.expected for item in result.items).values()) >= examples.MIN_EXAMPLES_PER_LABEL


def test_settings_load_without_an_argument_from_the_current_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["init"])

    assert config.load().path == (tmp_path / "cifarium.yaml").resolve()


def test_starter_settings_name_every_key_or_mention_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["init"])
    text = (tmp_path / "cifarium.yaml").read_text(encoding="utf-8")

    for key in ("task", "examples", "labels", "pass_rate", "limits", "max_size_mb",
                "max_latency_ms", "license", "host", "output"):
        assert f"{key}:" in text, key


@pytest.mark.parametrize("existing", cli.STARTER_FILES)
def test_init_never_overwrites(tmp_path, monkeypatch, capsys, existing):
    monkeypatch.chdir(tmp_path)
    (tmp_path / existing).write_text("mine", encoding="utf-8")

    assert cli.main(["init"]) == 2  # exitcodes.BAD_INPUT: the input is the folder, and it is taken

    assert (tmp_path / existing).read_text(encoding="utf-8") == "mine"
    assert [path.name for path in tmp_path.iterdir()] == [existing]
    assert "never overwrites" in capsys.readouterr().err


def test_init_where_it_cannot_write_says_so_and_leaves_nothing(tmp_path, capsys):
    not_a_folder = tmp_path / "file.txt"
    not_a_folder.write_text("", encoding="utf-8")

    assert cli.run_init(not_a_folder) == 3  # exitcodes.ENVIRONMENT: the disk, not the input

    assert "Nothing written" in capsys.readouterr().err
    assert [path.name for path in tmp_path.iterdir()] == ["file.txt"]


def test_init_removes_the_first_file_when_the_second_fails(tmp_path, monkeypatch, capsys):
    real_open = open

    def open_failing_on_examples(path, mode="r", *args, **kwargs):
        if str(path).endswith(cli.STARTER_FILES[1]):
            raise PermissionError(13, "Permission denied")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(cli, "open", open_failing_on_examples, raising=False)

    assert cli.run_init(tmp_path) == 3

    assert list(tmp_path.iterdir()) == []
    assert "cannot write examples.jsonl: Permission denied" in capsys.readouterr().err


def test_init_that_writes_exits_with_code_0(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert cli.main(["init"]) == 0


def test_help_lists_init(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])

    assert caught.value.code == 0
    assert "init" in capsys.readouterr().out

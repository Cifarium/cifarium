"""`cifarium init`: starter files, no browser."""

from __future__ import annotations

from collections import Counter

import pytest

from cifarium import cli, config, examples


def test_init_writes_starters_that_pass_the_checks_unchanged(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert cli.main(["init"]) == 0

    assert sorted(path.name for path in tmp_path.iterdir()) == ["examples.jsonl", "task.yaml"]
    assert f"Wrote task.yaml and examples.jsonl to {tmp_path}" in capsys.readouterr().out
    cfg = config.load(tmp_path / "task.yaml")
    result = examples.load(cfg)
    assert cfg.task == "classify"
    assert [label.name for label in result.labels] == ["billing", "bug", "account"]
    assert all(label.description for label in result.labels)
    assert len(result.items) >= 10
    # Leave-one-out scoring needs at least two examples under every label.
    assert min(Counter(item.expected for item in result.items).values()) >= 2


def test_starter_settings_name_every_key_or_mention_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["init"])
    text = (tmp_path / "task.yaml").read_text(encoding="utf-8")

    for key in ("task", "examples", "labels", "pass_rate", "limits", "max_size_mb",
                "max_latency_ms", "license", "host", "output"):
        assert f"{key}:" in text, key


@pytest.mark.parametrize("existing", cli.STARTER_FILES)
def test_init_never_overwrites(tmp_path, monkeypatch, capsys, existing):
    monkeypatch.chdir(tmp_path)
    (tmp_path / existing).write_text("mine", encoding="utf-8")

    assert cli.main(["init"]) == 1

    assert (tmp_path / existing).read_text(encoding="utf-8") == "mine"
    assert [path.name for path in tmp_path.iterdir()] == [existing]
    assert "never overwrites" in capsys.readouterr().err


def test_help_lists_init(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])

    assert caught.value.code == 0
    assert "init" in capsys.readouterr().out

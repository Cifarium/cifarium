"""The catalog-update command: comparing entries with a fake Hugging Face, no network."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import RevisionNotFoundError

from cifarium import catalog, catalog_update, cli, hub
from cifarium.catalog_update import Problem, pick

OLD_HASH = "0123456789abcdef0123456789abcdef01234567"
NEW_HASH = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

ENTRY = f"""\
id: tiny-model
repo: Org/tiny-model
revision: {OLD_HASH}
license: apache-2.0
license_source: org/tiny-source
languages: [en]
common_files:
  config.json: 10
  tokenizer.json: 20
  tokenizer_config.json: 30
variants:
  cpu:
    device: wasm
    dtype: q8
    files:
      onnx/model_quantized.onnx: 100
recipes:
  - task: classify
    kind: embed-centroid
    pooling: mean
    prefix: "query: "
"""

SHARED = {
    "config.json": 10,
    "tokenizer.json": 20,
    "tokenizer_config.json": 30,
    "onnx/model_quantized.onnx": 100,
    "onnx/model_q4.onnx": 150,
}


BLOBS = {name: f"blob-{name}" for name in SHARED}
"""The id every file has in every revision, until a test changes one."""
RETOUCHED = {**BLOBS, "onnx/model_quantized.onnx": "blob-retouched"}
"""Same files and sizes, but the weights are not the same bytes."""


def card(files: dict[str, int] | None = None, *, sha: str = OLD_HASH, license: str | None = None,
         base_model: str | None = None, gated: bool = False,
         blobs: dict[str, str] | None = None) -> SimpleNamespace:
    """What `HfApi.model_info` returns for one repository."""
    blobs = BLOBS if blobs is None else blobs
    return SimpleNamespace(
        sha=sha,
        siblings=[SimpleNamespace(rfilename=name, size=size, blob_id=blobs.get(name))
                  for name, size in (files or SHARED).items()],
        card_data=SimpleNamespace(license=license, base_model=base_model),
        gated=gated,
    )


def tiny_repos(**changes: object) -> dict[str, object]:
    """The Hub as the shipped test entry expects it, with `changes` applied per repository."""
    repos = {
        "Org/tiny-model": card(base_model="org/tiny-source"),
        "org/tiny-source": card(license="apache-2.0"),
    }
    repos.update(changes)
    return repos


class FakeApi:
    """The newest revision comes from ``repos``. A call with a ``revision`` comes from ``pinned``;
    a repository missing there still has its pinned revision, holding the files the entry lists."""

    def __init__(self, repos: dict[str, object], pinned: dict[str, object] | None = None) -> None:
        self.repos, self.pinned, self.calls = repos, pinned or {}, []

    def model_info(self, repo_id, *, revision=None, files_metadata=False):
        if revision is None:
            self.calls.append(repo_id)
            value = self.repos[repo_id]
        else:
            self.calls.append(f"{repo_id}@{revision[:7]}")
            value = self.pinned.get(repo_id, card(sha=revision))
        if isinstance(value, Exception):
            raise value
        return value


class ConnectError(Exception):
    """Stands for the HTTP client's error: not an OSError."""


def revision_gone() -> Exception:
    """The Hub client's error for a revision the Hub does not have. Built without the
    constructor, whose arguments differ between huggingface_hub versions."""
    return RevisionNotFoundError.__new__(RevisionNotFoundError)


def folder_with(tmp_path: Path, **entries: str) -> Path:
    folder = tmp_path / "catalog"
    folder.mkdir()
    for name, text in entries.items():
        (folder / f"{name}.yaml").write_bytes(text.encode("utf-8"))
    return folder


def go(folder: Path, repos: dict[str, object], pinned: dict[str, object] | None = None,
       **options) -> tuple[int, list[str], FakeApi]:
    lines: list[str] = []
    api = FakeApi(repos, pinned)
    code = catalog_update.run(folder=folder, api=api, out=lines.append, **options)
    return code, lines, api


def text_of(folder: Path, name: str = "tiny-model") -> str:
    return (folder / f"{name}.yaml").read_bytes().decode("utf-8")


# --- what the command reports and when it writes ---------------------------------------


def test_nothing_changed_reports_so_writes_nothing_and_exits_zero(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})

    code, lines, api = go(folder, tiny_repos(), write=True)

    assert code == 0
    assert lines[0] == "tiny-model  unchanged"
    assert lines[1] == "1 entries: 1 unchanged, 0 changed, 0 to check by hand, 0 with errors; 0 files rewritten"
    assert text_of(folder) == ENTRY
    assert api.calls == ["Org/tiny-model", "org/tiny-source"]


def test_a_new_hash_over_the_same_files_is_not_a_change_and_write_leaves_the_file_alone(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    repos = tiny_repos(**{"Org/tiny-model": card(sha=NEW_HASH, base_model="org/tiny-source")})

    code, lines, api = go(folder, repos, write=True)

    assert code == 0
    assert lines[0] == "tiny-model  unchanged (newer revision bbbbbbb has the same files)"
    assert lines[1] == "1 entries: 1 unchanged, 0 changed, 0 to check by hand, 0 with errors; 0 files rewritten"
    assert text_of(folder) == ENTRY
    assert api.calls == ["Org/tiny-model", "Org/tiny-model@0123456", "org/tiny-source"]


def test_a_new_hash_with_a_file_of_the_same_size_but_another_blob_id_is_a_change(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    repos = tiny_repos(**{"Org/tiny-model": card(sha=NEW_HASH, base_model="org/tiny-source", blobs=RETOUCHED)})

    code, lines, _ = go(folder, repos)

    assert code == 1
    assert lines[0] == ("tiny-model  changed: revision 0123456 -> bbbbbbb; "
                        "onnx/model_quantized.onnx content changed, same size (100 bytes)")
    assert lines[-1].startswith("Nothing written. Run with --write")
    assert text_of(folder) == ENTRY


def test_a_file_without_a_blob_id_cannot_prove_the_files_are_the_same(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    no_ids = {name: id_ for name, id_ in BLOBS.items() if name != "tokenizer.json"}
    repos = tiny_repos(**{"Org/tiny-model": card(sha=NEW_HASH, base_model="org/tiny-source", blobs=no_ids)})

    code, lines, _ = go(folder, repos)

    assert code == 1
    assert lines[0] == ("tiny-model  changed: revision 0123456 -> bbbbbbb; "
                        "tokenizer.json cannot be compared, the Hub gave no blob id")


def test_write_replaces_only_the_hash_when_a_file_changed_at_the_same_size(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    repos = tiny_repos(**{"Org/tiny-model": card(sha=NEW_HASH, base_model="org/tiny-source", blobs=RETOUCHED)})

    code, lines, _ = go(folder, repos, write=True)

    assert code == 0
    assert lines[0] == ("tiny-model  changed (rewritten): revision 0123456 -> bbbbbbb; "
                        "onnx/model_quantized.onnx content changed, same size (100 bytes)")
    assert text_of(folder) == ENTRY.replace(OLD_HASH, NEW_HASH)
    assert catalog.read_entry(folder / "tiny-model.yaml").revision == NEW_HASH


def test_a_changed_size_comes_with_the_new_hash_in_the_line_and_the_file(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    files = {**SHARED, "tokenizer.json": 25}
    repos = tiny_repos(**{"Org/tiny-model": card(files, sha=NEW_HASH, base_model="org/tiny-source")})

    code, lines, _ = go(folder, repos, write=True)

    assert code == 0
    assert lines[0] == "tiny-model  changed (rewritten): revision 0123456 -> bbbbbbb; tokenizer.json 20 -> 25 bytes"
    assert text_of(folder) == ENTRY.replace(OLD_HASH, NEW_HASH).replace("tokenizer.json: 20", "tokenizer.json: 25")


# --- the pinned revision ---------------------------------------------------------------


def test_a_pinned_revision_the_hub_no_longer_has_is_an_error_and_write_does_not_touch_it(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    repos = tiny_repos(**{"Org/tiny-model": card(sha=NEW_HASH, base_model="org/tiny-source")})

    code, lines, _ = go(folder, repos, {"Org/tiny-model": revision_gone()}, write=True)

    assert code == 1
    assert lines[0] == ("tiny-model  changed: revision 0123456 -> bbbbbbb | ERROR: the pinned revision 0123456 "
                        "is gone from Hugging Face, so downloading this model fails for everyone who installed cifarium")
    assert lines[1] == "1 entries: 0 unchanged, 0 changed, 0 to check by hand, 1 with errors; 0 files rewritten"
    assert text_of(folder) == ENTRY


def test_a_network_error_on_the_pinned_revision_is_an_error_that_says_so(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    repos = tiny_repos(**{"Org/tiny-model": card(sha=NEW_HASH, base_model="org/tiny-source")})

    code, lines, _ = go(folder, repos, {"Org/tiny-model": ConnectError("timed out")})

    assert code == 1
    assert lines[0] == ("tiny-model  ERROR: could not check the pinned revision 0123456: "
                        "Could not read Org/tiny-model on Hugging Face: timed out")


def test_a_pinned_revision_that_is_the_newest_is_not_asked_for_again(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})

    _, _, api = go(folder, tiny_repos(), {"Org/tiny-model": revision_gone()})

    assert api.calls == ["Org/tiny-model", "org/tiny-source"]


def test_a_newer_hash_over_the_same_files_does_not_hide_a_license_mismatch(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    repos = tiny_repos(**{
        "Org/tiny-model": card(sha=NEW_HASH, base_model="org/tiny-source"),
        "org/tiny-source": card(license="MIT"),
    })

    code, lines, _ = go(folder, repos, write=True)

    assert code == 1
    assert lines[0] == ("tiny-model  check by hand: license is apache-2.0, the card of org/tiny-source says mit"
                        " | newer revision bbbbbbb has the same files")
    assert text_of(folder) == ENTRY


def test_a_changed_size_names_the_file(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    files = {**SHARED, "tokenizer.json": 25, "onnx/model_quantized.onnx": 90}
    repos = tiny_repos(**{"Org/tiny-model": card(files, base_model="org/tiny-source")})

    code, lines, _ = go(folder, repos, write=True)

    assert code == 0
    assert lines[0] == ("tiny-model  changed (rewritten): tokenizer.json 20 -> 25 bytes; "
                        "onnx/model_quantized.onnx 100 -> 90 bytes")
    assert text_of(folder) == (ENTRY.replace("tokenizer.json: 20", "tokenizer.json: 25")
                               .replace("model_quantized.onnx: 100", "model_quantized.onnx: 90"))


def test_the_dtype_switches_from_q8_to_q4_and_the_files_follow(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    files = {**SHARED, "onnx/model_q4.onnx": 80, "onnx/model_q4.onnx_data": 5}
    repos = tiny_repos(**{"Org/tiny-model": card(files, base_model="org/tiny-source")})

    code, lines, _ = go(folder, repos, write=True)

    assert code == 0
    assert lines[0] == ("tiny-model  changed (rewritten): dtype q8 -> q4; +onnx/model_q4.onnx (80 bytes); "
                        "+onnx/model_q4.onnx_data (5 bytes); -onnx/model_quantized.onnx")
    assert text_of(folder) == ENTRY.replace("dtype: q8", "dtype: q4").replace(
        "      onnx/model_quantized.onnx: 100\n",
        "      onnx/model_q4.onnx: 80\n      onnx/model_q4.onnx_data: 5\n",
    )
    assert catalog.read_entry(folder / "tiny-model.yaml").variants["cpu"].dtype == "q4"


def test_a_vanished_common_file_is_an_error_for_that_entry_and_the_others_go_on(tmp_path):
    other = (ENTRY.replace("id: tiny-model", "id: other-model").replace("Org/tiny-model", "Org/other-model")
             .replace("org/tiny-source", "org/other-source"))
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY, "other-model": other})
    without_tokenizer = {name: size for name, size in SHARED.items() if name != "tokenizer.json"}
    repos = tiny_repos(**{
        "Org/tiny-model": card(without_tokenizer, base_model="org/tiny-source"),
        "Org/other-model": card(sha=NEW_HASH, base_model="org/other-source", blobs=RETOUCHED),
        "org/other-source": card(license="apache-2.0"),
    })

    code, lines, _ = go(folder, repos, write=True)

    assert code == 1
    assert lines[0] == ("other-model  changed (rewritten): revision 0123456 -> bbbbbbb; "
                        "onnx/model_quantized.onnx content changed, same size (100 bytes)")
    assert lines[1] == "tiny-model   ERROR: common file missing on the Hub: tokenizer.json"
    assert "1 changed, 0 to check by hand, 1 with errors; 1 files rewritten" in lines[2]
    assert text_of(folder) == ENTRY
    assert NEW_HASH in text_of(folder, "other-model")


def test_a_gated_copy_is_an_error_and_is_never_rewritten(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    repos = tiny_repos(**{"Org/tiny-model": card(sha=NEW_HASH, base_model="org/tiny-source", gated="manual")})

    code, lines, _ = go(folder, repos, write=True)

    assert code == 1
    assert lines[0] == "tiny-model  ERROR: the repository now needs a signed-in account (gated)"
    assert text_of(folder) == ENTRY


def test_a_gated_source_model_is_fine(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    repos = tiny_repos(**{"org/tiny-source": card(license="apache-2.0", gated="auto")})

    code, lines, _ = go(folder, repos)

    assert (code, lines[0]) == (0, "tiny-model  unchanged")


def test_a_network_error_names_the_repository_and_the_other_entries_are_still_checked(tmp_path):
    other = (ENTRY.replace("id: tiny-model", "id: other-model").replace("Org/tiny-model", "Org/other-model")
             .replace("org/tiny-source", "org/other-source"))
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY, "other-model": other})
    repos = tiny_repos(**{
        "Org/tiny-model": ConnectError("name resolution failed\ndetails"),
        "Org/other-model": card(base_model="org/other-source"),
        "org/other-source": card(license="apache-2.0"),
    })

    code, lines, _ = go(folder, repos, write=True)

    assert code == 1
    assert lines[0] == "other-model  unchanged"
    assert lines[1] == ("tiny-model   ERROR: Could not read Org/tiny-model on Hugging Face: "
                        "name resolution failed")
    assert text_of(folder) == ENTRY


# --- the fields a person checks by hand ----------------------------------------------


def test_a_license_mismatch_is_only_shown_and_the_exit_code_stays_one_even_with_write(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    repos = tiny_repos(**{
        "Org/tiny-model": card(sha=NEW_HASH, base_model="org/tiny-source", blobs=RETOUCHED),
        "org/tiny-source": card(license="MIT"),
    })

    code, lines, _ = go(folder, repos, write=True)

    assert code == 1
    assert lines[0] == ("tiny-model  changed (rewritten): revision 0123456 -> bbbbbbb; "
                        "onnx/model_quantized.onnx content changed, same size (100 bytes) | "
                        "check by hand: license is apache-2.0, the card of org/tiny-source says mit")
    assert text_of(folder) == ENTRY.replace(OLD_HASH, NEW_HASH)  # license, recipes and the rest are as they were


def test_a_card_without_a_license_is_unknown_and_matches_the_entry(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY.replace("license: apache-2.0", "license: unknown")})
    repos = tiny_repos(**{"org/tiny-source": card(license=None)})

    code, lines, _ = go(folder, repos)

    assert (code, lines[0]) == (0, "tiny-model  unchanged")


def test_a_changed_base_model_is_shown(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    repos = tiny_repos(**{"Org/tiny-model": card(base_model="org/moved-source")})

    code, lines, _ = go(folder, repos)

    assert code == 1
    assert lines[0] == ("tiny-model  check by hand: license_source is org/tiny-source, "
                        "the card now names org/moved-source as base_model")


def test_an_unreadable_license_source_is_shown_not_fatal(tmp_path):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    repos = tiny_repos(**{"org/tiny-source": ConnectError("gone")})

    code, lines, _ = go(folder, repos)

    assert code == 1
    assert lines[0] == ("tiny-model  check by hand: license not checked: "
                        "Could not read org/tiny-source on Hugging Face: gone")


def test_an_entry_with_other_variants_is_refused_rather_than_cut_down(tmp_path):
    with_gpu = ENTRY.replace(
        "recipes:",
        "  gpu:\n    device: webgpu\n    dtype: q4f16\n    files:\n      onnx/model_q4f16.onnx: 70\nrecipes:",
    )
    folder = folder_with(tmp_path, **{"tiny-model": with_gpu})
    repos = tiny_repos(**{"Org/tiny-model": card(sha=NEW_HASH, base_model="org/tiny-source")})

    code, lines, _ = go(folder, repos, write=True)

    assert code == 1
    assert lines[0].startswith("tiny-model  ERROR: entry has variants other than cpu")
    assert text_of(folder) == with_gpu


# --- the writer ----------------------------------------------------------------------


def test_rewriting_every_shipped_entry_from_its_own_facts_reproduces_it_byte_for_byte():
    for entry in catalog.load():
        text = entry.path.read_bytes().decode("utf-8")

        assert catalog_update.rewrite(text, catalog_update.fresh_of(entry)) == text, entry.id


def test_a_comment_between_the_blocks_and_crlf_line_endings_survive():
    text = ENTRY.replace("variants:", "# picked by size\nvariants:").replace("\n", "\r\n")
    fresh = catalog_update.Fresh(NEW_HASH, {"config.json": 1, "tokenizer.json": 2, "tokenizer_config.json": 3},
                                 "q8", {"onnx/model_quantized.onnx": 4})

    rewritten = catalog_update.rewrite(text, fresh)

    assert "# picked by size\r\nvariants:\r\n" in rewritten
    assert "\n" not in rewritten.replace("\r\n", "")
    assert f"revision: {NEW_HASH}\r\n" in rewritten
    assert "  config.json: 1\r\n  tokenizer.json: 2\r\n  tokenizer_config.json: 3\r\nvariants:" in rewritten.replace(
        "# picked by size\r\n", "")
    assert rewritten.endswith('    prefix: "query: "\r\n')


def test_a_file_that_fails_the_catalog_check_after_writing_is_restored(tmp_path, monkeypatch):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})
    real = catalog.read_entry

    def strict(path):
        if NEW_HASH in Path(path).read_text(encoding="utf-8"):
            raise catalog.CatalogError(Path(path), ["revision: pretend problem"])
        return real(path)

    monkeypatch.setattr(catalog, "read_entry", strict)
    repos = tiny_repos(**{"Org/tiny-model": card(sha=NEW_HASH, base_model="org/tiny-source", blobs=RETOUCHED)})

    code, lines, _ = go(folder, repos, write=True)

    assert code == 1
    assert "ERROR: not rewritten:" in lines[0]
    assert text_of(folder) == ENTRY


# --- the rule that picks files ---------------------------------------------------------


def info_of(files: dict[str, int], sha: str = NEW_HASH) -> hub.RepoInfo:
    common = {"config.json": 1, "tokenizer.json": 2, "tokenizer_config.json": 3}
    return hub.RepoInfo("org/m", sha, {**common, **files}, None, None, False)


def test_at_equal_weight_q8_wins():
    fresh = pick(info_of({"onnx/model_quantized.onnx": 100, "onnx/model_q4.onnx": 100}))

    assert (fresh.dtype, fresh.files) == ("q8", {"onnx/model_quantized.onnx": 100})


def test_the_lighter_of_q8_and_q4_wins_counting_the_data_parts():
    files = {
        "onnx/model_quantized.onnx": 5, "onnx/model_quantized.onnx_data": 300,
        "onnx/model_q4.onnx": 5, "onnx/model_q4.onnx_data": 200, "onnx/model_q4.onnx_data_1": 10,
        "onnx/model_no_gather_q4.onnx": 5, "onnx/model_no_gather_q4.onnx_data": 100,  # not q4
        "onnx/model_q4f16.onnx": 1, "onnx/model_fp16.onnx": 1, "onnx/model.onnx": 1,  # not taken
    }

    fresh = pick(info_of(files))

    assert fresh.dtype == "q4"
    assert fresh.files == {"onnx/model_q4.onnx": 5, "onnx/model_q4.onnx_data": 200, "onnx/model_q4.onnx_data_1": 10}
    assert fresh.common_files == {"config.json": 1, "tokenizer.json": 2, "tokenizer_config.json": 3}
    assert fresh.revision == NEW_HASH


def test_a_single_available_dtype_is_taken():
    assert pick(info_of({"onnx/model_q4.onnx": 100})).dtype == "q4"


def test_no_q8_or_q4_file_is_a_problem():
    with pytest.raises(Problem, match="no q8 .* or q4 .* file"):
        pick(info_of({"onnx/model.onnx": 100, "onnx/model_fp16.onnx": 50}))


def test_a_missing_common_file_is_named():
    info = hub.RepoInfo("org/m", NEW_HASH, {"config.json": 1, "onnx/model_quantized.onnx": 5}, None, None, False)

    with pytest.raises(Problem, match="tokenizer.json, tokenizer_config.json"):
        pick(info)


# --- --only and the command line -------------------------------------------------------


def test_only_checks_the_named_entries(tmp_path):
    other = (ENTRY.replace("id: tiny-model", "id: other-model").replace("Org/tiny-model", "Org/other-model")
             .replace("org/tiny-source", "org/other-source"))
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY, "other-model": other})

    code, lines, api = go(folder, tiny_repos(), only=["tiny-model"])

    assert code == 0
    assert lines[0] == "tiny-model  unchanged"
    assert api.calls == ["Org/tiny-model", "org/tiny-source"]


def test_an_unknown_entry_name_is_an_error_with_a_hint(tmp_path, capsys):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY})

    code, lines, api = go(folder, tiny_repos(), only=["tiny-modle"])

    assert code == 1
    assert lines == [] and api.calls == []
    assert capsys.readouterr().err == "--only tiny-modle: no such catalog entry; did you mean tiny-model?\n"


def test_a_broken_catalog_stops_before_any_network_call(tmp_path, capsys):
    folder = folder_with(tmp_path, **{"tiny-model": ENTRY.replace("dtype: q8", "dtype: q3")})

    code, lines, api = go(folder, tiny_repos())

    assert code == 1 and lines == [] and api.calls == []
    assert "variants.cpu.dtype" in capsys.readouterr().err


def test_the_command_is_wired_up_but_not_listed_in_help(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(catalog_update, "run", lambda **options: seen.update(options) or 3)

    code = cli.main(["catalog-update", "--write", "--only", "a", "--only", "b"])

    assert code == 3
    assert seen == {"write": True, "only": ["a", "b"]}
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "catalog-update" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["catalog-update", "--help"])
    assert "--write" in capsys.readouterr().out

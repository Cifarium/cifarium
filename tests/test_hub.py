"""Model download: folder layout and error reporting, without the network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from huggingface_hub.errors import RevisionNotFoundError

from cifarium import hub


def test_files_go_to_the_folder_the_bench_server_serves(tmp_path, monkeypatch):
    calls = []

    def fake_download(repo_id, filename, *, revision, local_dir):
        calls.append((repo_id, filename, revision, local_dir))

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)

    folder = hub.download_files(
        "Xenova/all-MiniLM-L6-v2", "abc123", ["config.json", "onnx/model_quantized.onnx"], tmp_path
    )

    assert folder == tmp_path / "models" / "Xenova" / "all-MiniLM-L6-v2"
    assert calls == [
        ("Xenova/all-MiniLM-L6-v2", "config.json", "abc123", folder),
        ("Xenova/all-MiniLM-L6-v2", "onnx/model_quantized.onnx", "abc123", folder),
    ]


def test_download_errors_name_the_file(tmp_path, monkeypatch):
    def offline(repo_id, filename, *, revision, local_dir):
        raise OSError("network is unreachable\nmore details")

    monkeypatch.setattr(hub, "hf_hub_download", offline)

    with pytest.raises(hub.HubError, match="config.json of org/model: network is unreachable$"):
        hub.download_files("org/model", "abc123", ["config.json"], tmp_path)


class ConnectError(Exception):
    """Stands for the HTTP client's error: not an OSError."""


def test_a_download_error_that_is_not_an_oserror_is_a_hub_error_too(tmp_path, monkeypatch):
    def offline(repo_id, filename, *, revision, local_dir):
        raise ConnectError("name resolution failed")

    monkeypatch.setattr(hub, "hf_hub_download", offline)

    with pytest.raises(hub.HubError, match="config.json of org/model: name resolution failed$"):
        hub.download_files("org/model", "abc123", ["config.json"], tmp_path)


class FakeApi:
    def __init__(self, info=None, error=None):
        self.info, self.error, self.calls = info, error, []

    def model_info(self, repo_id, *, revision=None, files_metadata=False):
        self.calls.append((repo_id, revision, files_metadata))
        if self.error:
            raise self.error
        return self.info


def fake_info(**changes):
    sha = "a" * 40
    fields = dict(
        sha=sha,
        siblings=[
            SimpleNamespace(rfilename="config.json", size=650, blob_id="c" * 40),
            SimpleNamespace(rfilename="onnx/model_quantized.onnx", size=23_026_053, blob_id="d" * 40),
            SimpleNamespace(rfilename="tokenizer.json", size=700, blob_id=None),
            SimpleNamespace(rfilename="README.md", size=None, blob_id="e" * 40),
        ],
        card_data=SimpleNamespace(license="apache-2.0", base_model=["org/source"]),
        gated=False,
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_repo_info_reads_hash_sizes_blob_ids_license_and_gating():
    api = FakeApi(fake_info())

    info = hub.repo_info("Xenova/model", api=api)

    assert api.calls == [("Xenova/model", None, True)]
    assert info == hub.RepoInfo(
        repo="Xenova/model",
        revision="a" * 40,
        files={"config.json": 650, "onnx/model_quantized.onnx": 23_026_053, "tokenizer.json": 700},
        license="apache-2.0",
        base_model="org/source",
        gated=False,
        # tokenizer.json has no id and README.md no size: neither is listed where it is missing.
        blobs={"config.json": "c" * 40, "onnx/model_quantized.onnx": "d" * 40},
    )


def test_a_revision_the_hub_does_not_have_is_told_apart_from_other_errors():
    # Built without the constructor: its arguments differ between huggingface_hub versions.
    gone = RevisionNotFoundError.__new__(RevisionNotFoundError)

    with pytest.raises(hub.RevisionNotFound, match="org/model has no revision " + "b" * 40):
        hub.repo_info("org/model", "b" * 40, api=FakeApi(error=gone))
    assert issubclass(hub.RevisionNotFound, hub.HubError)


def test_repo_info_without_card_and_with_a_gate():
    info = hub.repo_info("org/model", "b" * 40, api=FakeApi(fake_info(card_data=None, gated="manual")))

    assert (info.license, info.base_model, info.gated) == (None, None, True)


@pytest.mark.parametrize("error", [OSError("name resolution failed\ndetails"),
                                   ConnectError("name resolution failed")])
def test_repo_info_errors_name_the_repository(error):
    with pytest.raises(hub.HubError, match="Could not read org/model on Hugging Face: name resolution failed$"):
        hub.repo_info("org/model", api=FakeApi(error=error))

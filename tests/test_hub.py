"""Model download: folder layout and error reporting, without the network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
            SimpleNamespace(rfilename="config.json", size=650),
            SimpleNamespace(rfilename="onnx/model_quantized.onnx", size=23_026_053),
            SimpleNamespace(rfilename="README.md", size=None),
        ],
        card_data=SimpleNamespace(license="apache-2.0", base_model=["org/source"]),
        gated=False,
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_repo_info_reads_hash_sizes_license_and_gating():
    api = FakeApi(fake_info())

    info = hub.repo_info("Xenova/model", api=api)

    assert api.calls == [("Xenova/model", None, True)]
    assert info == hub.RepoInfo(
        repo="Xenova/model",
        revision="a" * 40,
        files={"config.json": 650, "onnx/model_quantized.onnx": 23_026_053},
        license="apache-2.0",
        base_model="org/source",
        gated=False,
    )


def test_repo_info_without_card_and_with_a_gate():
    info = hub.repo_info("org/model", "b" * 40, api=FakeApi(fake_info(card_data=None, gated="manual")))

    assert (info.license, info.base_model, info.gated) == (None, None, True)


class ConnectError(Exception):
    """Stands for the HTTP client's error: not an OSError."""


@pytest.mark.parametrize("error", [OSError("name resolution failed\ndetails"),
                                   ConnectError("name resolution failed")])
def test_repo_info_errors_name_the_repository(error):
    with pytest.raises(hub.HubError, match="Could not read org/model on Hugging Face: name resolution failed$"):
        hub.repo_info("org/model", api=FakeApi(error=error))

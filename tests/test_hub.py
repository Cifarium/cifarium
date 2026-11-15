"""Model download: folder layout and error reporting, without the network."""

from __future__ import annotations

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

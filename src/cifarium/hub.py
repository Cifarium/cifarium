"""Model files from Hugging Face."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from . import paths


class HubError(RuntimeError):
    """Model files or facts about them could not be fetched."""


@dataclass(frozen=True)
class RepoInfo:
    """What Hugging Face says about one model repository at one revision."""

    repo: str
    revision: str
    """Full commit hash, 40 characters."""
    files: dict[str, int]
    """Path inside the repository -> size in bytes."""
    license: str | None
    """License from the model card metadata, if the card names one."""
    base_model: str | None
    """Source repository named in the card, for copies such as ONNX conversions."""
    gated: bool
    """Whether downloading needs a signed-in account that accepted the terms."""


def repo_info(repo_id: str, revision: str | None = None, api: HfApi | None = None) -> RepoInfo:
    """Commit hash, file sizes, card license and gating of one model, without downloading."""
    api = api or HfApi()
    try:
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
    # Without the network huggingface_hub 2.4 raises the HTTP client's own ConnectError,
    # which is neither OSError nor HfHubHTTPError; only this one call is guarded.
    except Exception as error:  # noqa: BLE001
        lines = str(error).strip().splitlines()
        reason = lines[0] if lines else type(error).__name__
        raise HubError(f"Could not read {repo_id} on Hugging Face: {reason}") from error

    def first(value: object) -> str | None:
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        return value if isinstance(value, str) and value else None

    card = info.card_data
    return RepoInfo(
        repo=repo_id,
        revision=info.sha,
        files={item.rfilename: item.size for item in info.siblings or () if item.size is not None},
        license=first(getattr(card, "license", None)),
        base_model=first(getattr(card, "base_model", None)),
        gated=bool(info.gated),
    )


def model_dir(repo_id: str, cache: Path | None = None) -> Path:
    """Folder of one model: ``<cache>/models/<organisation>/<model>/``."""
    return paths.models_dir(cache).joinpath(*repo_id.split("/"))


def download_files(
    repo_id: str, revision: str, filenames: Iterable[str], cache: Path | None = None
) -> Path:
    """Download the listed files of one pinned revision and return their folder.

    With ``local_dir`` the files land under their own names, in the layout
    Transformers.js expects from a local folder, so the bench server can serve
    them as they are. It also avoids symbolic links, which are a problem on
    Windows. Files that are already there are not downloaded again.
    """
    folder = model_dir(repo_id, cache)
    for filename in filenames:
        try:
            hf_hub_download(repo_id, filename, revision=revision, local_dir=folder)
        # Like repo_info: without the network huggingface_hub 2.4 raises the HTTP client's own
        # ConnectError, which is neither OSError nor EntryNotFoundError.
        except Exception as error:  # noqa: BLE001
            lines = str(error).strip().splitlines()
            reason = lines[0] if lines else type(error).__name__
            raise HubError(f"Could not download {filename} of {repo_id}: {reason}") from error
    return folder

"""Model files from Hugging Face."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from . import paths


class HubError(RuntimeError):
    """Model files could not be downloaded."""


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
        except (OSError, EntryNotFoundError) as error:
            lines = str(error).strip().splitlines()
            reason = lines[0] if lines else type(error).__name__
            raise HubError(f"Could not download {filename} of {repo_id}: {reason}") from error
    return folder

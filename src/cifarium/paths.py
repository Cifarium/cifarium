"""Where cifarium keeps its files."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def cache_dir() -> Path:
    """Root of the cache: ``%LOCALAPPDATA%\\cifarium`` on Windows."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    else:
        base = os.environ.get("XDG_CACHE_HOME")
        root = Path(base) if base else Path.home() / ".cache"
    return root / "cifarium"


def vendor_dir(cache: Path | None = None) -> Path:
    """Engine files, one folder per Transformers.js version."""
    return (cache or cache_dir()) / "vendor"


def models_dir(cache: Path | None = None) -> Path:
    """Model files laid out as ``<organisation>/<model>/...``."""
    return (cache or cache_dir()) / "models"


def profile_dir(cache: Path | None = None) -> Path:
    """Chrome profile used only by cifarium."""
    return (cache or cache_dir()) / "chrome-profile"


def web_dir() -> Path:
    """Browser files shipped inside the package."""
    return Path(__file__).resolve().parent / "web"

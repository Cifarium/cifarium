"""End-to-end run in the real Chrome. Not part of the default run: `pytest -m e2e`."""

from __future__ import annotations

import pytest

from cifarium import paths, smoke


@pytest.mark.e2e
def test_vectors_come_back_from_chrome(capsys):
    assert smoke.run() == 0

    output = capsys.readouterr().out
    assert "20 x 384" in output
    assert "processor (wasm)" in output
    assert "crossOriginIsolated   true" in output
    # The browser's copy of the model must not stay on disk after the run.
    assert not paths.profile_dir().exists()

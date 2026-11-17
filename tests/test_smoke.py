"""The temporary smoke check: its labelled sets, and the end-to-end run in the real Chrome.

The end-to-end run is not part of the default run: `pytest -m e2e`.
"""

from __future__ import annotations

from collections import Counter

import pytest

from cifarium import paths, smoke


@pytest.mark.parametrize("items", [smoke.ENGLISH, smoke.RUSSIAN], ids=["english", "russian"])
def test_a_set_has_three_labels_with_equal_counts_and_no_repeats(items):
    counts = Counter(label for _, label in items)
    assert len(counts) == 3
    assert len(set(counts.values())) == 1
    texts = [text.strip() for text, _ in items]
    assert len(set(texts)) == len(texts)


def test_the_smoke_entry_is_in_the_catalog():
    entry = smoke.find_entry(smoke.ENTRY)
    assert entry.repo == "Xenova/all-MiniLM-L6-v2"


@pytest.mark.e2e
def test_examples_are_scored_on_vectors_from_chrome(capsys):
    assert smoke.run() == 0

    output = capsys.readouterr().out
    assert f"{len(smoke.ENGLISH)} x 384" in output
    assert "processor (wasm)" in output
    assert "crossOriginIsolated   true" in output
    assert "pass rate 85%: passed" in output
    # The browser's copy of the model must not stay on disk after the run.
    assert not paths.profile_dir().exists()

"""How well a recipe classifies the examples. Plain numpy, no browser.

Recipe "by examples" (``embed-centroid``): label centres. The bench turns every example into a
vector once; the rest happens here (ARCHITECTURE.md, section 3, "Classification"):

- the centre of a label is the mean of its examples' vectors, scaled to unit length;
- a text gets the label whose centre is nearest, that is, has the largest dot product;
- the score at build time checks every example against centres computed without it.

Vectors must be of unit length, as the bench returns them.

Recipe "does the label fit" (``zero-shot``): the bench returns the score of every label for every
example, and :func:`zero_shot` takes the label with the highest one. The examples taught the model
nothing, so every example is checked as it is.

Labels are named by text and kept in one order, ``names``: the order of the settings, or of first
appearance in the examples (``Examples.labels``). That order breaks ties, in both recipes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# How far the length of an input vector may be from 1. The bench normalises in
# float32 and the weights are quantized, so the lengths are 1 to about 1e-6.
UNIT_TOLERANCE = 1e-3

# Guards the comparison of the share right with pass_rate against rounding:
# 17 of 20 must pass 0.85 however both are rounded.
_RATE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Miss:
    """One example the centres got wrong."""

    index: int
    """Position of the example in the vectors passed in."""
    expected: str
    predicted: str
    """The label that was named instead."""
    score: float
    """Dot product with the centre of the named label, from -1 to 1; for ``zero-shot`` the score of
    the named label, from 0 to 1."""
    expected_score: float | None
    """Dot product with the centre of the expected label; None when the label had no
    centre left, because its only example was the one being checked. For ``zero-shot`` the
    score of the expected label."""


@dataclass(frozen=True)
class Evaluation:
    """How many examples got their label, against the share the settings ask for."""

    total: int
    correct: int
    pass_rate: float
    misses: tuple[Miss, ...]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    @property
    def passed(self) -> bool:
        return self.accuracy + _RATE_TOLERANCE >= self.pass_rate


def centroids(vectors: np.ndarray, expected: Sequence[str], names: Sequence[str]) -> np.ndarray:
    """Centre of every label, one row per name in ``names`` order.

    Each centre is the mean of the vectors of that label's examples, scaled to
    unit length, so that its dot product with a unit vector is the cosine.
    Raises :class:`ValueError` when a label has no examples.
    """
    vectors, codes = _check(vectors, expected, names)
    sums, _ = _label_sums(vectors, codes, names)
    return _unit_rows(sums)


def predict(vectors: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest centre for every vector: its row number in ``centres`` and the dot product.

    A tie goes to the centre that comes first, that is, the label listed first.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    centres = np.asarray(centres, dtype=np.float64)
    if vectors.ndim != 2 or centres.ndim != 2 or vectors.shape[1] != centres.shape[1]:
        raise ValueError(f"vectors {vectors.shape} and centres {centres.shape} do not match")
    scores = vectors @ centres.T
    best = np.argmax(scores, axis=1)  # the first of equal values
    return best, scores[np.arange(len(best)), best]


def leave_one_out(
    vectors: np.ndarray, expected: Sequence[str], names: Sequence[str], pass_rate: float
) -> Evaluation:
    """Check every example on centres computed without it.

    Only the centre of the example's own label changes when the example is left
    out, so one pass over the vectors is enough. An example that is the only one
    of its label leaves that label without a centre: it cannot get its label and
    counts as a miss. Every other example still sees that label's full centre.
    """
    vectors, codes = _check(vectors, expected, names)
    sums, counts = _label_sums(vectors, codes, names)
    full = _unit_rows(sums)
    scores = vectors @ full.T

    rows = np.arange(len(codes))
    own_count = counts[codes] - 1
    own_sum = sums[codes] - vectors
    own_norm = np.linalg.norm(own_sum, axis=1)
    has_centre = (own_count > 0) & (own_norm > 0)
    own_score = np.full(len(codes), -np.inf)
    own_score[has_centre] = (
        np.einsum("ij,ij->i", vectors[has_centre], own_sum[has_centre]) / own_norm[has_centre]
    )
    scores[rows, codes] = own_score
    return _evaluation(scores, codes, names, pass_rate)


def held_out(
    centres: np.ndarray,
    vectors: np.ndarray,
    expected: Sequence[str],
    names: Sequence[str],
    pass_rate: float,
) -> Evaluation:
    """Check examples the centres were not computed from, such as a held-out part."""
    vectors, codes = _check(vectors, expected, names)
    centres = np.asarray(centres, dtype=np.float64)
    if centres.shape != (len(names), vectors.shape[1]):
        raise ValueError(f"expected {len(names)} centres of length {vectors.shape[1]}, "
                         f"got {centres.shape}")
    return _evaluation(vectors @ centres.T, codes, names, pass_rate)


def zero_shot(
    scores: np.ndarray, expected: Sequence[str], names: Sequence[str], pass_rate: float
) -> Evaluation:
    """Recipe ``zero-shot``: every example gets the label with the highest score.

    ``scores`` has one row per example and one column per label in ``names`` order, as the bench
    returns them: the softmax over the labels, so a row adds up to 1. Nothing is left out, because
    the examples taught the model nothing: the share is that of every example as it is. A miss
    names the score of the label named and of the expected one.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != len(expected):
        raise ValueError(f"expected one row of scores per example ({len(expected)}), got shape {scores.shape}")
    _check_names(expected, names)
    if scores.shape[1] != len(names):
        raise ValueError(f"expected {len(names)} scores per example, one per label, got {scores.shape[1]}")
    if not np.isfinite(scores).all():
        raise ValueError("the scores contain NaN or infinity")
    outside = np.flatnonzero(((scores < -UNIT_TOLERANCE) | (scores > 1 + UNIT_TOLERANCE)).any(axis=1))
    if outside.size:
        raise ValueError(f"the scores of example {outside[0]} are not between 0 and 1: {scores[outside[0]].tolist()}")
    sums = scores.sum(axis=1)
    off = np.flatnonzero(np.abs(sums - 1) > UNIT_TOLERANCE)
    if off.size:
        raise ValueError(f"the scores of example {off[0]} add up to {sums[off[0]]:.6g}; "
                         "the bench must return a softmax over the labels")
    return _evaluation(scores, _codes(expected, names), names, pass_rate)


def _check(
    vectors: np.ndarray, expected: Sequence[str], names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Vectors as float64 and every example's label as its row number in ``names``."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(expected) or vectors.shape[1] == 0:
        raise ValueError(f"expected one vector per example ({len(expected)}), got shape {vectors.shape}")
    _check_names(expected, names)
    if not np.isfinite(vectors).all():
        raise ValueError("the vectors contain NaN or infinity")
    lengths = np.linalg.norm(vectors, axis=1)
    off = np.flatnonzero(np.abs(lengths - 1) > UNIT_TOLERANCE)
    if off.size:
        raise ValueError(f"vector {off[0]} has length {lengths[off[0]]:.6g}; "
                         "the bench must return vectors of unit length")
    return vectors, _codes(expected, names)


def _check_names(expected: Sequence[str], names: Sequence[str]) -> None:
    if len(expected) == 0:
        raise ValueError("no examples")
    if len(set(names)) != len(names):
        raise ValueError(f"label names repeat: {list(names)}")
    if len(names) < 2:
        raise ValueError("classification needs at least two labels")


def _codes(expected: Sequence[str], names: Sequence[str]) -> np.ndarray:
    """Every example's label as its row number in ``names``."""
    position = {name: k for k, name in enumerate(names)}
    unknown = sorted({label for label in expected if label not in position})
    if unknown:
        raise ValueError(f"labels {unknown} are not among {list(names)}")
    return np.array([position[label] for label in expected])


def _label_sums(
    vectors: np.ndarray, codes: np.ndarray, names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Sum of the vectors and number of examples of every label. Every label needs one."""
    sums = np.zeros((len(names), vectors.shape[1]))
    np.add.at(sums, codes, vectors)
    counts = np.bincount(codes, minlength=len(names))
    empty = [names[k] for k in range(len(names)) if counts[k] == 0]
    if empty:
        raise ValueError(f"no examples for label {', '.join(map(repr, empty))}; "
                         "a centre needs at least one")
    return sums, counts


def _unit_rows(rows: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(rows, axis=1, keepdims=True)
    if (lengths == 0).any():
        raise ValueError("the examples of a label cancel out: its centre has no direction")
    return rows / lengths


def _evaluation(scores: np.ndarray, codes: np.ndarray, names: Sequence[str], pass_rate: float) -> Evaluation:
    best = np.argmax(scores, axis=1)  # ties go to the label listed first
    rows = np.arange(len(codes))
    misses = tuple(
        Miss(
            index=int(i),
            expected=names[codes[i]],
            predicted=names[best[i]],
            score=float(scores[i, best[i]]),
            expected_score=float(scores[i, codes[i]]) if np.isfinite(scores[i, codes[i]]) else None,
        )
        for i in rows[best != codes]
    )
    return Evaluation(total=len(codes), correct=len(codes) - len(misses),
                      pass_rate=pass_rate, misses=misses)

"""Label centres and the leave-one-out score, on made-up vectors."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cifarium import evaluate
from cifarium.evaluate import Evaluation


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def angle(degrees: float) -> np.ndarray:
    return np.array([math.cos(math.radians(degrees)), math.sin(math.radians(degrees))])


def random_task(seed: int, sizes: dict[str, int], dims: int = 8) -> tuple[np.ndarray, list[str], list[str]]:
    """Unit vectors scattered around one random direction per label, shuffled."""
    rng = np.random.default_rng(seed)
    names = list(sizes)
    rows, expected = [], []
    for name in names:
        direction = rng.normal(size=dims)
        for _ in range(sizes[name]):
            rows.append(direction + rng.normal(scale=1.2, size=dims))
            expected.append(name)
    order = rng.permutation(len(rows))
    vectors = np.array(rows)[order]
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors, [expected[i] for i in order], names


def brute_force(vectors: np.ndarray, expected: list[str], names: list[str]) -> list[tuple[str, float]]:
    """Leave-one-out the slow way: drop the example, recompute every centre, predict."""
    answers = []
    for i in range(len(vectors)):
        rest = [j for j in range(len(vectors)) if j != i]
        scores = []
        for name in names:
            mine = [j for j in rest if expected[j] == name]
            if not mine:
                scores.append(-np.inf)
                continue
            centre = vectors[mine].mean(axis=0)
            scores.append(float(vectors[i] @ (centre / np.linalg.norm(centre))))
        best = int(np.argmax(scores))
        answers.append((names[best], scores[best]))
    return answers


# --- centres ---

def test_a_centre_is_the_mean_scaled_to_unit_length():
    vectors = np.array([angle(0), angle(90), angle(180), angle(200)])
    centres = evaluate.centroids(vectors, ["a", "a", "b", "b"], ["a", "b"])
    np.testing.assert_allclose(centres[0], angle(45))
    np.testing.assert_allclose(centres[1], angle(190))
    np.testing.assert_allclose(np.linalg.norm(centres, axis=1), 1.0)


def test_centres_follow_the_order_of_names():
    vectors = np.array([angle(0), angle(90)])
    centres = evaluate.centroids(vectors, ["a", "b"], ["b", "a"])
    np.testing.assert_allclose(centres, [angle(90), angle(0)], atol=1e-12)


def test_a_label_without_examples_has_no_centre():
    vectors = np.array([angle(0), angle(90)])
    with pytest.raises(ValueError, match="no examples for label 'c'"):
        evaluate.centroids(vectors, ["a", "b"], ["a", "b", "c"])


def test_examples_that_cancel_out_give_no_centre():
    vectors = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="cancel out"):
        evaluate.centroids(vectors, ["a", "a", "b"], ["a", "b"])


# --- prediction ---

def test_the_nearest_centre_wins_and_its_dot_product_is_the_score():
    centres = np.array([angle(0), angle(90), angle(180)])
    best, scores = evaluate.predict(np.array([angle(80), angle(170)]), centres)
    assert best.tolist() == [1, 2]
    np.testing.assert_allclose(scores, [math.cos(math.radians(10))] * 2)


def test_a_tie_goes_to_the_label_listed_first():
    # Exact numbers: sin and cos of 45 degrees may differ in the last bit on some systems.
    centres = np.array([[1.0, 0.0], [0.0, 1.0]])
    best, _ = evaluate.predict(np.array([unit(1, 1)]), centres)
    assert best.tolist() == [0]
    best, _ = evaluate.predict(np.array([unit(1, 1)]), centres[::-1])
    assert best.tolist() == [0]


def test_prediction_refuses_vectors_of_another_length():
    with pytest.raises(ValueError, match="do not match"):
        evaluate.predict(np.ones((2, 3)), np.ones((2, 4)))


# --- leave one out ---

@pytest.mark.parametrize("seed", range(6))
def test_leave_one_out_matches_recomputing_the_centres(seed):
    sizes = {"billing": 7, "bug": 6, "account": 7} if seed % 2 else {"a": 2, "b": 3, "c": 1, "d": 9}
    vectors, expected, names = random_task(seed, sizes)

    result = evaluate.leave_one_out(vectors, expected, names, pass_rate=0.85)

    answers = brute_force(vectors, expected, names)
    wrong = [i for i, (name, _) in enumerate(answers) if name != expected[i]]
    assert [miss.index for miss in result.misses] == wrong
    assert result.correct == len(vectors) - len(wrong)
    for miss in result.misses:
        assert miss.predicted == answers[miss.index][0]
        assert miss.score == pytest.approx(answers[miss.index][1])
        assert miss.expected == expected[miss.index]


def test_an_example_right_on_full_centres_can_be_wrong_when_left_out():
    vectors = np.array([angle(0), angle(55), angle(80), angle(100)])
    expected, names = ["a", "a", "b", "b"], ["a", "b"]

    full = evaluate.centroids(vectors, expected, names)
    best, _ = evaluate.predict(vectors[1:2], full)
    assert names[best[0]] == "a"  # its own vector pulls the centre of a towards it

    result = evaluate.leave_one_out(vectors, expected, names, pass_rate=0.85)
    (miss,) = [miss for miss in result.misses if miss.index == 1]
    assert miss.predicted == "b"
    assert miss.score == pytest.approx(math.cos(math.radians(35)))
    assert miss.expected_score == pytest.approx(math.cos(math.radians(55)))


def test_the_only_example_of_a_label_counts_as_a_miss():
    vectors = np.array([angle(0), angle(10), angle(90), angle(100), angle(30)])
    expected, names = ["a", "a", "b", "b", "lonely"], ["a", "b", "lonely"]

    result = evaluate.leave_one_out(vectors, expected, names, pass_rate=0.5)

    (miss,) = result.misses
    assert (miss.index, miss.expected, miss.predicted) == (4, "lonely", "a")
    assert miss.expected_score is None
    assert result.correct == 4


def test_the_other_examples_still_see_the_full_centre_of_a_one_example_label():
    vectors = np.array([angle(10), angle(60), angle(150), angle(160), angle(62)])
    expected, names = ["a", "a", "b", "b", "lonely"], ["a", "b", "lonely"]

    result = evaluate.leave_one_out(vectors, expected, names, pass_rate=0.5)

    # Example 1 at 60 degrees is nearer to "lonely" (62) than to "a" without itself (10).
    assert {(miss.index, miss.predicted) for miss in result.misses} == {(1, "lonely"), (4, "a")}


def test_a_tie_in_the_score_goes_to_the_label_listed_first():
    centres = np.array([[1.0, 0.0], [0.0, 1.0]])  # exactly, so the two dot products are equal
    result = evaluate.held_out(centres, np.array([unit(1, 1)]), ["b"], ["a", "b"], pass_rate=0.5)
    (miss,) = result.misses
    assert miss.predicted == "a"
    assert miss.score == miss.expected_score


def test_every_example_right():
    expected = ["a"] * 5 + ["b"] * 5
    vectors = np.array([unit(1, 0, 0, 0) if label == "a" else unit(0, 1, 0, 0) for label in expected])
    result = evaluate.leave_one_out(vectors, expected, ["a", "b"], pass_rate=1.0)
    assert (result.total, result.correct, result.misses, result.passed) == (10, 10, (), True)
    assert result.accuracy == 1.0


# --- held-out examples ---

def test_held_out_examples_are_scored_on_the_given_centres():
    train, train_expected, names = random_task(3, {"a": 6, "b": 6, "c": 6})
    test, test_expected, _ = random_task(3, {"a": 6, "b": 6, "c": 6})
    centres = evaluate.centroids(train, train_expected, names)

    result = evaluate.held_out(centres, test, test_expected, names, pass_rate=0.85)

    best, scores = evaluate.predict(test, centres)
    wrong = [i for i in range(len(test)) if names[best[i]] != test_expected[i]]
    assert [miss.index for miss in result.misses] == wrong
    for miss in result.misses:
        assert miss.score == pytest.approx(scores[miss.index])


def test_held_out_needs_one_centre_per_label():
    vectors, expected, names = random_task(1, {"a": 2, "b": 2})
    with pytest.raises(ValueError, match="expected 2 centres"):
        evaluate.held_out(np.ones((3, vectors.shape[1])), vectors, expected, names, pass_rate=0.85)


# --- pass rate ---

def test_17_of_20_pass_085_and_16_do_not():
    assert Evaluation(total=20, correct=17, pass_rate=0.85, misses=()).passed
    assert not Evaluation(total=20, correct=16, pass_rate=0.85, misses=()).passed


@pytest.mark.parametrize("total", range(1, 41))
def test_exactly_the_asked_share_passes_whatever_the_rounding(total):
    for correct in range(1, total + 1):
        rate = round(correct / total, 2)  # how a person writes it in the settings
        needed = math.ceil(rate * total - 1e-9)
        assert Evaluation(total, needed, rate, ()).passed
        assert not Evaluation(total, needed - 1, rate, ()).passed


# --- checks of the input ---

def test_vectors_must_be_of_unit_length():
    vectors = np.array([angle(0), angle(90) * 2])
    with pytest.raises(ValueError, match="vector 1 has length 2"):
        evaluate.leave_one_out(vectors, ["a", "b"], ["a", "b"], pass_rate=0.85)


def test_one_vector_per_example():
    with pytest.raises(ValueError, match="one vector per example"):
        evaluate.leave_one_out(np.array([angle(0)]), ["a", "b"], ["a", "b"], pass_rate=0.85)


def test_labels_must_be_among_the_names():
    vectors = np.array([angle(0), angle(90)])
    with pytest.raises(ValueError, match=r"\['c'\] are not among"):
        evaluate.leave_one_out(vectors, ["a", "c"], ["a", "b"], pass_rate=0.85)


def test_nan_is_refused():
    vectors = np.array([angle(0), [np.nan, 0.0]])
    with pytest.raises(ValueError, match="NaN"):
        evaluate.centroids(vectors, ["a", "b"], ["a", "b"])


def test_float32_vectors_from_the_bench_are_accepted():
    vectors, expected, names = random_task(2, {"a": 4, "b": 4})
    result = evaluate.leave_one_out(vectors.astype("<f4"), expected, names, pass_rate=0.5)
    assert result.total == 8


# --- recipe zero-shot: the label with the highest score ---


def softmax(rows: list[list[float]]) -> np.ndarray:
    logits = np.array(rows, dtype=np.float64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def test_zero_shot_names_the_label_with_the_highest_score_and_counts_every_example():
    scores = softmax([[3, 0, 0], [0, 2, 1], [0, 0, 5], [4, 1, 0]])
    expected = ["a", "b", "c", "b"]

    result = evaluate.zero_shot(scores, expected, ["a", "b", "c"], pass_rate=0.75)

    assert (result.total, result.correct, result.accuracy, result.passed) == (4, 3, 0.75, True)
    [miss] = result.misses
    assert (miss.index, miss.expected, miss.predicted) == (3, "b", "a")
    assert miss.score == pytest.approx(scores[3, 0]) and miss.expected_score == pytest.approx(scores[3, 1])


def test_zero_shot_gives_a_tie_to_the_label_listed_first():
    scores = np.array([[0.5, 0.5], [0.5, 0.5]])

    result = evaluate.zero_shot(scores, ["a", "b"], ["a", "b"], pass_rate=0.5)

    assert result.correct == 1 and result.misses[0].predicted == "a" and result.misses[0].expected == "b"


def test_zero_shot_seventeen_of_twenty_pass_eighty_five_percent():
    expected = ["a", "b"] * 10
    rows = [[0.9, 0.1] if label == "a" else [0.1, 0.9] for label in expected]
    for k in (1, 3, 5):  # three examples of b get the score of a
        rows[k] = [0.9, 0.1]

    result = evaluate.zero_shot(np.array(rows), expected, ["a", "b"], pass_rate=0.85)

    assert (result.correct, result.passed) == (17, True)


def test_zero_shot_accepts_float32_scores_that_add_up_to_one_in_float32():
    scores = softmax([[0.3, 1.7, -2.0]] * 2).astype("<f4")
    assert evaluate.zero_shot(scores, ["b", "b"], ["a", "b", "c"], pass_rate=1.0).correct == 2


@pytest.mark.parametrize(
    ("scores", "said"),
    [
        (np.array([[0.5, 0.5]]), "one row of scores per example"),
        (np.array([0.5, 0.5]), "one row of scores per example"),
        (np.array([[1.0], [1.0]]), "2 scores per example, one per label, got 1"),
        (np.array([[np.nan, 0.5], [0.5, 0.5]]), "NaN"),
        (np.array([[1.5, -0.5], [0.5, 0.5]]), "not between 0 and 1"),
        (np.array([[0.5, 0.5], [0.9, 0.9]]), "example 1 add up to 1.8"),
    ],
    ids=["too-few-rows", "flat", "one-column", "nan", "outside", "not-a-softmax"],
)
def test_zero_shot_refuses_scores_that_cannot_be_read(scores, said):
    with pytest.raises(ValueError, match=said):
        evaluate.zero_shot(scores, ["a", "b"], ["a", "b"], pass_rate=0.5)


@pytest.mark.parametrize(
    ("expected", "names", "said"),
    [
        ([], ["a", "b"], "no examples"),
        (["a", "a"], ["a", "a"], "repeat"),
        (["a", "a"], ["a"], "two labels"),
        (["a", "c"], ["a", "b"], r"\['c'\] are not among"),
    ],
)
def test_zero_shot_checks_the_labels_as_the_centres_do(expected, names, said):
    scores = np.full((len(expected), len(names)), 1 / len(names))
    with pytest.raises(ValueError, match=said):
        evaluate.zero_shot(scores, expected, names, pass_rate=0.5)

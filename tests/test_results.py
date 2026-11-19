"""The cache of results without a browser: what makes a key, what is found again, what is not
used, what is kept and what goes. ARCHITECTURE.md, section 6 and section 9 ("The cache of results")."""

from __future__ import annotations

import functools
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import cifarium
from cifarium import catalog, results, vendor
from cifarium.results import Cache, text_hash

DAY = 24 * 3600


@functools.cache
def candidate() -> catalog.Candidate:
    """A real candidate of the catalog: a vector model with a pooling, a prefix and a dtype."""
    return next(c for c in catalog.candidates("classify")
                if c.entry.id == "all-minilm-l6-v2" and c.recipe.kind == "embed-centroid")


def unit(rows: int, dims: int, seed: int = 0) -> np.ndarray:
    vectors = np.random.default_rng(seed).normal(size=(rows, dims)).astype("<f4")
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype("<f4")


def vectors_of(texts: list[str], seed: int = 0) -> dict[bytes, np.ndarray]:
    dims = candidate().recipe.params["dimensions"]
    return dict(zip((text_hash(t) for t in texts), unit(len(texts), dims, seed)))


# --- the key ----------------------------------------------------------------------------------


def with_entry(**changes):
    return lambda c: replace(c, entry=replace(c.entry, **changes))


def with_variant(**changes):
    def change(c):
        variants = {**c.entry.variants, "cpu": replace(c.entry.variants["cpu"], **changes)}
        return replace(c, entry=replace(c.entry, variants=variants))
    return change


def with_params(**changes):
    return lambda c: replace(c, recipe=replace(c.recipe, params={**c.recipe.params, **changes}))


def with_file_sizes(c):
    cpu = c.entry.variants["cpu"]
    name, size = next(iter(cpu.files.items()))
    return with_variant(files={**cpu.files, name: size + 1})(c)


def with_common_file(c):
    return with_entry(common_files={**c.entry.common_files, "tokenizer.json": 1})(c)


CHANGES = {
    "model": with_entry(repo="someone/else"),
    "revision": with_entry(revision="f" * 40),
    "dtype": with_variant(dtype="fp16"),
    "device": with_variant(device="webgpu"),
    "variant name": with_variant(name="gpu"),
    "file of the variant": with_file_sizes,
    "file of the model": with_common_file,
    "pooling": with_params(pooling="cls"),
    "prefix": with_params(prefix="query: "),
    "recipe": lambda c: replace(c, recipe=replace(c.recipe, kind="zero-shot")),
}


@pytest.mark.parametrize("name", CHANGES)
def test_every_term_a_vector_depends_on_changes_the_key(tmp_path, name):
    cache = Cache(tmp_path)

    assert cache.path_of(CHANGES[name](candidate())) != cache.path_of(candidate())


@pytest.mark.parametrize("what", ["cifarium", "transformers.js", "onnxruntime-web", "page", "format"])
def test_the_versions_of_the_package_the_engine_and_the_page_are_in_the_key(tmp_path, monkeypatch, what):
    cache = Cache(tmp_path)
    before = cache.path_of(candidate())

    {
        "cifarium": lambda: monkeypatch.setattr(cifarium, "__version__", "9.9.9"),
        "transformers.js": lambda: monkeypatch.setattr(vendor, "TRANSFORMERS_VERSION", "0.0.0"),
        "onnxruntime-web": lambda: monkeypatch.setattr(vendor, "ORT_WEB_VERSION", "0.0.0"),
        "page": lambda: monkeypatch.setattr(results, "page_digest", lambda: "another page"),
        "format": lambda: monkeypatch.setattr(results, "FORMAT", results.FORMAT + 1),
    }[what]()

    assert cache.path_of(candidate()) != before


def test_the_page_digest_follows_the_files_of_the_bench_page(tmp_path, monkeypatch):
    from cifarium import paths

    folder = tmp_path / "web" / "bench"
    folder.mkdir(parents=True)
    (folder / "bench.js").write_text("one", encoding="utf-8")
    monkeypatch.setattr(paths, "web_dir", lambda: tmp_path / "web")
    results.page_digest.cache_clear()
    try:
        first = results.page_digest()
        (folder / "bench.js").write_text("two", encoding="utf-8")
        results.page_digest.cache_clear()
        second = results.page_digest()
    finally:
        results.page_digest.cache_clear()

    assert first != second


def test_what_does_not_change_a_vector_is_not_in_the_key(tmp_path):
    cache = Cache(tmp_path)
    same = replace(candidate(), entry=replace(candidate().entry, license="mit", languages=("en",),
                                              path=Path("elsewhere.yaml")))

    assert cache.path_of(same) == cache.path_of(candidate())


def test_the_file_is_named_for_its_model_and_safe_for_any_id(tmp_path):
    odd = replace(candidate(), entry=replace(candidate().entry, id="a/b c:d"))

    assert Cache(tmp_path).path_of(candidate()).name.startswith("all-minilm-l6-v2-")
    assert Cache(tmp_path).path_of(odd).parent == tmp_path
    assert "/" not in Cache(tmp_path).path_of(odd).name and " " not in Cache(tmp_path).path_of(odd).name


def test_a_text_is_known_by_its_hash_and_any_text_has_one():
    assert text_hash("a question") == text_hash("a question") != text_hash("a question ")
    assert len(text_hash("")) == results.DIGEST_BYTES
    assert text_hash("\ud800")  # a lone surrogate from an escape in the JSON of the examples


# --- keeping and finding vectors --------------------------------------------------------------


def test_the_vectors_come_back_as_they_went_in(tmp_path):
    cache, texts = Cache(tmp_path), ["one", "two", "three"]
    kept = vectors_of(texts)

    cache.store(candidate(), kept)
    found = cache.load(candidate())

    assert list(found) == list(kept)
    assert all(np.array_equal(found[digest], kept[digest]) for digest in kept)
    assert all(vector.dtype == np.dtype("<f4") for vector in found.values())


def test_no_file_is_no_vectors_and_no_error(tmp_path):
    assert Cache(tmp_path / "not there").load(candidate()) == {}


def test_a_key_that_differs_never_finds_the_vectors_of_another(tmp_path):
    cache = Cache(tmp_path)
    cache.store(candidate(), vectors_of(["one"]))

    for name, change in CHANGES.items():
        assert cache.load(change(candidate())) == {}, name


def test_a_file_that_says_another_key_is_not_used_whatever_its_name(tmp_path):
    cache = Cache(tmp_path)
    other = with_params(pooling="cls")(candidate())
    cache.store(other, vectors_of(["one"]))
    # Somebody copies the file of one key over the name of another.
    cache.path_of(candidate()).write_bytes(cache.path_of(other).read_bytes())

    assert cache.load(candidate()) == {}


def test_the_texts_never_reach_the_disk(tmp_path):
    cache = Cache(tmp_path)
    cache.store(candidate(), vectors_of(["a very private sentence about the shop"]))

    assert b"private" not in cache.path_of(candidate()).read_bytes()


def test_a_second_store_keeps_what_the_file_held_after_what_this_build_has(tmp_path):
    cache = Cache(tmp_path)
    cache.store(candidate(), vectors_of(["old one", "old two"]))

    cache.store(candidate(), vectors_of(["new one", "old two"], seed=7))
    found = cache.load(candidate())

    assert list(found) == [text_hash(t) for t in ("new one", "old two", "old one")]
    # The vector of a text that came in again is the new one.
    assert np.array_equal(found[text_hash("old two")], vectors_of(["new one", "old two"], seed=7)[text_hash("old two")])


def test_a_file_holds_at_most_max_vectors_and_the_newest_texts_stay(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "MAX_VECTORS", 3)
    cache = Cache(tmp_path)
    cache.store(candidate(), vectors_of(["a", "b", "c"]))

    cache.store(candidate(), vectors_of(["d", "e"]))

    assert list(cache.load(candidate())) == [text_hash(t) for t in ("d", "e", "a")]


def test_vectors_that_cannot_be_used_are_not_written(tmp_path):
    cache = Cache(tmp_path)
    dims = candidate().recipe.params["dimensions"]
    wrong_length = {text_hash("x"): unit(1, dims)[0] * 2}
    not_a_number = {text_hash("y"): np.full(dims, np.nan, dtype="<f4")}
    wrong_dims = {text_hash("z"): unit(1, dims + 1)[0]}

    for bad in (wrong_length, not_a_number, wrong_dims):
        cache.store(candidate(), bad)

    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_one_bad_vector_keeps_the_good_ones_out_too_and_the_old_file_stays(tmp_path):
    cache = Cache(tmp_path)
    cache.store(candidate(), vectors_of(["kept"]))
    dims = candidate().recipe.params["dimensions"]

    cache.store(candidate(), {**vectors_of(["fresh"]), text_hash("bad"): np.zeros(dims, dtype="<f4")})

    assert list(cache.load(candidate())) == [text_hash("kept")]


def test_a_disk_that_refuses_raises_os_error_and_leaves_no_leftover(tmp_path):
    blocker = tmp_path / "results"
    blocker.write_text("a file where the folder should be", encoding="utf-8")

    with pytest.raises(OSError):
        Cache(blocker).store(candidate(), vectors_of(["one"]))

    assert list(tmp_path.iterdir()) == [blocker]


def test_a_write_that_fails_halfway_leaves_the_file_as_it_was(tmp_path, monkeypatch):
    cache = Cache(tmp_path)
    cache.store(candidate(), vectors_of(["before"]))

    def cut(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", cut)
    with pytest.raises(OSError):
        cache.store(candidate(), vectors_of(["after"]))

    assert list(cache.load(candidate())) == [text_hash("before")]
    assert [path.name for path in tmp_path.iterdir()] == [cache.path_of(candidate()).name]


# --- a file that is not usable is no vectors, never an error ----------------------------------


def damaged(path: Path, how: str) -> None:
    data = path.read_bytes()
    path.write_bytes({
        "empty": b"",
        "noise": os.urandom(4096),
        "cut": data[: len(data) // 2],
        "flipped": data[:200] + bytes(b ^ 0xFF for b in data[200:400]) + data[400:],
        "text": b"not a cache at all\n",
        "a pickle": b"\x80\x04\x95\x05\x00\x00\x00\x00\x00\x00\x00\x8c\x03abc\x94.",
    }[how])


@pytest.mark.parametrize("how", ["empty", "noise", "cut", "flipped", "text", "a pickle"])
def test_a_damaged_file_is_not_used_and_the_next_store_replaces_it(tmp_path, how):
    cache = Cache(tmp_path)
    cache.store(candidate(), vectors_of(["one", "two"]))
    damaged(cache.path_of(candidate()), how)

    assert cache.load(candidate()) == {}
    cache.store(candidate(), vectors_of(["one", "two"]))

    assert list(cache.load(candidate())) == [text_hash("one"), text_hash("two")]


def write_arrays(cache: Cache, **arrays) -> None:
    arrays.setdefault("meta", np.array(results.key_text(candidate())))
    cache.folder.mkdir(parents=True, exist_ok=True)
    with open(cache.path_of(candidate()), "wb") as handle:
        np.savez(handle, **arrays)


def hashes(count: int) -> np.ndarray:
    return np.frombuffer(b"".join(text_hash(str(k)) for k in range(count)), dtype=np.uint8).reshape(-1, 32)


def test_a_file_of_the_right_key_that_holds_the_wrong_arrays_is_not_used(tmp_path):
    cache = Cache(tmp_path)
    dims = candidate().recipe.params["dimensions"]
    good = unit(2, dims)
    cases = {
        "a missing array": dict(texts=hashes(2)),
        "fewer texts than vectors": dict(texts=hashes(1), vectors=good),
        "texts of the wrong width": dict(texts=hashes(2)[:, :16], vectors=good),
        "doubles": dict(texts=hashes(2), vectors=good.astype("<f8")),
        "a flat vector": dict(texts=hashes(2), vectors=good.reshape(-1)),
        "other dimensions": dict(texts=hashes(2), vectors=unit(2, dims + 1)),
        "not a number": dict(texts=hashes(2), vectors=np.where(good > 0, np.nan, good).astype("<f4")),
        "not unit length": dict(texts=hashes(2), vectors=(good * 1.5).astype("<f4")),
        "nothing": dict(texts=hashes(0), vectors=unit(0, dims)),
    }
    for name, arrays in cases.items():
        write_arrays(cache, **arrays)
        assert cache.load(candidate()) == {}, name

    write_arrays(cache, texts=hashes(2), vectors=good)  # the same file, right
    assert len(cache.load(candidate())) == 2


def test_a_file_with_more_vectors_than_a_file_ever_holds_is_not_used(tmp_path, monkeypatch):
    cache = Cache(tmp_path)
    dims = candidate().recipe.params["dimensions"]
    write_arrays(cache, texts=hashes(3), vectors=unit(3, dims))
    monkeypatch.setattr(results, "MAX_VECTORS", 2)

    assert cache.load(candidate()) == {}


def test_a_file_far_bigger_than_the_cache_writes_is_not_even_opened(tmp_path, monkeypatch):
    cache = Cache(tmp_path)
    cache.store(candidate(), vectors_of(["one"]))
    monkeypatch.setattr(results, "MAX_FILE_BYTES", 100)

    assert cache.load(candidate()) == {}


def test_a_folder_in_place_of_the_file_is_no_vectors(tmp_path):
    cache = Cache(tmp_path)
    cache.path_of(candidate()).mkdir()

    assert cache.load(candidate()) == {}


# --- what goes -------------------------------------------------------------------------------


def aged(path: Path, days: float, now: float) -> None:
    os.utime(path, (now - days * DAY, now - days * DAY))


def test_a_file_nobody_used_for_two_months_goes_and_a_used_one_stays(tmp_path):
    cache, now = Cache(tmp_path), time.time()
    for name in ("old", "young"):
        (tmp_path / f"{name}{results.SUFFIX}").write_bytes(b"x")
    aged(tmp_path / f"old{results.SUFFIX}", results.MAX_AGE_DAYS + 1, now)
    aged(tmp_path / f"young{results.SUFFIX}", results.MAX_AGE_DAYS - 1, now)

    assert cache.prune(now) == 1
    assert [path.name for path in tmp_path.iterdir()] == [f"young{results.SUFFIX}"]


def test_reading_a_file_keeps_it_from_going(tmp_path):
    cache = Cache(tmp_path)
    cache.store(candidate(), vectors_of(["one"]))
    path = cache.path_of(candidate())
    now = time.time()
    aged(path, results.MAX_AGE_DAYS + 1, now)

    assert cache.load(candidate())  # the build that used it
    assert cache.prune(now) == 0 and path.exists()


def test_when_all_files_are_too_big_the_oldest_go_first(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "MAX_BYTES", 250)
    cache, now = Cache(tmp_path), time.time()
    for age, name in enumerate(("newest", "middle", "oldest")):
        (tmp_path / f"{name}{results.SUFFIX}").write_bytes(b"x" * 100)
        aged(tmp_path / f"{name}{results.SUFFIX}", age + 1, now)

    assert cache.prune(now) == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == [f"middle{results.SUFFIX}", f"newest{results.SUFFIX}"]


def test_what_is_not_a_file_of_the_cache_is_never_touched(tmp_path):
    cache, now = Cache(tmp_path), time.time()
    for name in ("notes.txt", "x.json", "x.npz.bak"):
        (tmp_path / name).write_bytes(b"x" * 10)
        aged(tmp_path / name, 1000, now)
    (tmp_path / "folder.npz").mkdir()

    assert cache.prune(now) == 0
    assert len(list(tmp_path.iterdir())) == 4


def test_the_leftover_of_a_write_that_was_cut_goes_after_a_day_and_not_before(tmp_path):
    cache, now = Cache(tmp_path), time.time()
    fresh = tmp_path / f"a{results.SUFFIX}.123.tmp"
    stale = tmp_path / f"b{results.SUFFIX}.456.tmp"
    for path in (fresh, stale):
        path.write_bytes(b"x")
    aged(fresh, 0.5, now)
    aged(stale, 2, now)

    assert cache.prune(now) == 1
    assert fresh.exists() and not stale.exists()


def test_a_folder_that_is_not_there_or_not_a_folder_is_nothing_to_prune(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")

    assert Cache(tmp_path / "not there").prune() == 0
    assert Cache(blocker).prune() == 0

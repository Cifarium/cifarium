"""Temporary end-to-end check: one catalog model, a small labelled set, on the processor.

Downloads the engine and the model of the catalog entry into the cache, starts
the bench, computes the vectors in Chrome the way the entry's recipe says and
scores the examples by leaving each one out (evaluate.py). It exists until the
real commands do: remove it, and its entry in cli.py, once `cifarium build`
covers the same path.

:func:`measure` runs any catalog entry on any labelled set in an open bench;
the numbers for other models in ARCHITECTURE.md, section 3, come from it with
the two sets below.
"""

from __future__ import annotations

import functools
import re
import statistics
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from . import catalog, evaluate, hub, vendor
from .bench import Bench, BenchError, decode_vectors, embed_job, open_bench
from .config import DEFAULT_PASS_RATE

ENTRY = "all-minilm-l6-v2"
TIME_LIMIT_S = 300.0

# Requests to an online shop, three labels, seven examples each.
ENGLISH: tuple[tuple[str, str], ...] = (
    ("Where is my package? The tracking page hasn't updated in four days.", "shipping"),
    ("I'd like to return these shoes, they are too small.", "returns"),
    ("The app logs me out every few minutes.", "technical"),
    ("Can you ship to Canada, and how long does it take?", "shipping"),
    ("How many days do I have to send an item back?", "returns"),
    ("I can't upload a photo, the page just spins forever.", "technical"),
    ("The courier left my parcel at the wrong address.", "shipping"),
    ("When will I get my money back for the jacket I returned?", "returns"),
    ("The checkout button is greyed out on my phone.", "technical"),
    ("My order says delivered but nothing arrived.", "shipping"),
    ("Do I have to pay for the return label?", "returns"),
    ("Your website shows a blank page in Firefox.", "technical"),
    ("Is express delivery available for this item?", "shipping"),
    ("The blender I received is not what I ordered and I want to send it back.", "returns"),
    ("Push notifications stopped working after the last update.", "technical"),
    ("How much does shipping cost for an order under $20?", "shipping"),
    ("Can I exchange the shirt for a larger size?", "returns"),
    ("I get error 403 when I open my order history.", "technical"),
    ("Can I change the delivery address after placing the order?", "shipping"),
    ("I returned the order two weeks ago and still have no refund.", "returns"),
    ("The search bar finds nothing, even for products that exist.", "technical"),
)

# The same kind of requests in Russian, five examples per label.
RUSSIAN: tuple[tuple[str, str], ...] = (
    ("Где моя посылка? Трек-номер не обновляется уже неделю.", "доставка"),
    ("Хочу вернуть кроссовки, они не подошли по размеру.", "возврат"),
    ("Приложение вылетает при входе в личный кабинет.", "сбой"),
    ("Курьер не приехал в назначенное время.", "доставка"),
    ("Когда мне вернут деньги за возвращённый товар?", "возврат"),
    ("Не могу оплатить заказ: кнопка оплаты не нажимается.", "сбой"),
    ("Можно ли изменить адрес доставки после оформления заказа?", "доставка"),
    ("Сколько дней есть на возврат покупки?", "возврат"),
    ("Сайт выдаёт ошибку 500 при оформлении заказа.", "сбой"),
    ("Сколько стоит доставка в Казань?", "доставка"),
    ("Можно обменять куртку на другой цвет?", "возврат"),
    ("После обновления перестали приходить уведомления.", "сбой"),
    ("Заказ отмечен как доставленный, но я его не получал.", "доставка"),
    ("Пришёл не тот товар, оформите, пожалуйста, возврат.", "возврат"),
    ("Поиск по каталогу ничего не находит.", "сбой"),
)

_say = functools.partial(print, flush=True)


def run(*, headed: bool = False) -> int:
    """Run the check on all-MiniLM and the English set and return the process exit code."""
    _say("cifarium smoke test")
    try:
        entry = find_entry(ENTRY)
        vendor.ensure_engine(report=lambda message: _say(f"  {message}"))
        _say(
            f"  engine    Transformers.js {vendor.TRANSFORMERS_VERSION}, "
            f"onnxruntime-web {vendor.ORT_WEB_VERSION}"
        )
        download(entry)
        variant = entry.variants["cpu"]
        _say(f"  model     {entry.repo} ({variant.dtype}), {entry.first_load_bytes() / 1e6:.1f} MB")
        with open_bench(headed=headed) as bench:
            _say(f"  browser   {_describe_browser(bench)}")
            result, vectors, score = measure(
                bench, entry, ENGLISH, on_progress=lambda message: _say(f"  page      {message}")
            )
    except (catalog.CatalogError, vendor.VendorError, hub.HubError, BenchError, ValueError) as error:
        print(f"\nFAILED: {error}", file=sys.stderr)
        return 1

    times = result["timesMs"]
    device = "processor (wasm)" if result["device"] == "wasm" else "graphics card (webgpu)"
    rows = [
        ("vectors", f"{vectors.shape[0]} x {vectors.shape[1]}"),
        ("device", f"{device}, {result['threads']} threads"),
        ("crossOriginIsolated", str(result["crossOriginIsolated"]).lower()),
        ("model load", f"{result['loadMs'] / 1000:.2f} s"),
        (
            "per example",
            f"{statistics.median(times[1:]):.1f} ms median, first call {times[0]:.1f} ms",
        ),
        ("leave one out", describe_score(score)),
    ]
    _say()
    for name, value in rows:
        _say(f"  {name:<21} {value}")
    for line in describe_misses(score, [text for text, _ in ENGLISH]):
        _say(f"  {line}")

    if not score.passed:
        print(f"\nFAILED: {entry.id} got {score.correct} of {score.total} examples, "
              f"below the pass rate {score.pass_rate:g}", file=sys.stderr)
        return 1
    _say("\nOK")
    return 0


def find_entry(entry_id: str) -> catalog.Entry:
    """The catalog entry with this id."""
    for entry in catalog.load():
        if entry.id == entry_id:
            return entry
    raise catalog.CatalogError(catalog.CATALOG_DIR, [f"no entry {entry_id}"])


def download(entry: catalog.Entry) -> None:
    """Put the files of the entry's processor variant into the models folder of the cache."""
    hub.download_files(entry.repo, entry.revision, entry.files())


def measure(
    bench: Bench,
    entry: catalog.Entry,
    items: Sequence[tuple[str, str]],
    *,
    pass_rate: float = DEFAULT_PASS_RATE,
    overrides: dict[str, Any] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], np.ndarray, evaluate.Evaluation]:
    """Vectors of ``items`` (text, label) from the bench and their leave-one-out score.

    The job follows the entry's ``embed-centroid`` recipe; ``overrides`` replaces
    fields of it, for example ``{"prefix": ""}`` to see what the prefix changes.
    Labels are ordered by first appearance, as without labels in the settings.
    """
    recipe = next(r for r in entry.recipes if r.kind == "embed-centroid")
    job = {**embed_job(entry, recipe, [text for text, _ in items]), **(overrides or {})}
    result = bench.run(job, timeout=TIME_LIMIT_S, on_progress=on_progress)
    vectors = decode_vectors(result)
    expected = [label for _, label in items]
    names = list(dict.fromkeys(expected))
    if vectors.shape[0] != len(items):
        raise ValueError(f"expected {len(items)} vectors, got {vectors.shape[0]}")
    return result, vectors, evaluate.leave_one_out(vectors, expected, names, pass_rate)


def describe_score(score: evaluate.Evaluation) -> str:
    verdict = "passed" if score.passed else "FAILED"
    return (f"{score.correct} of {score.total} ({score.accuracy:.0%}), "
            f"pass rate {score.pass_rate:.0%}: {verdict}")


def describe_misses(score: evaluate.Evaluation, texts: Sequence[str]) -> list[str]:
    """One line per miss: the label named and its score, the expected one and its score."""
    lines = []
    for miss in score.misses:
        own = "no centre" if miss.expected_score is None else f"{miss.expected_score:.3f}"
        lines.append(f"miss #{miss.index}: {miss.predicted} {miss.score:.3f} "
                     f"instead of {miss.expected} {own}: {texts[miss.index][:60]}")
    return lines


def _describe_browser(bench: Bench) -> str:
    version = re.search(r"Chrome/(\d+)", bench.device["userAgent"])
    name = f"Chrome {version.group(1)}" if version else "Chrome"
    window = "without a window" if bench.headless else "with a window"
    gpu = bench.device["webgpu"]
    if not gpu:
        adapter = "none"
    else:
        adapter = " ".join(part for part in (gpu["vendor"], gpu["architecture"]) if part) or "unnamed"
        if gpu["isFallbackAdapter"]:
            adapter += " (software)"
    return f"{name} {window}, WebGPU adapter: {adapter}"

"""The build in the real Chrome on all-MiniLM-L6-v2, and the two labelled sets of the manual runs.

The end-to-end run is not part of the default run: `pytest -m e2e`. This file names what the
run downloads (the English set picks all-MiniLM-L6-v2 from the catalog), so the cache key of
the end-to-end job in .github/workflows/tests.yml is made of it, vendor.py and the catalog.

The sets are requests to an online shop with three labels. ARCHITECTURE.md, section 3, has
the scores of every catalog model on them; :func:`write_task` turns one into the two files of
a task for a manual `cifarium build`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest
from publishable import example_leaks

from cifarium import bundle, paths

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


def write_task(folder: Path, items: tuple[tuple[str, str], ...], settings: str = "") -> Path:
    """Write cifarium.yaml and examples.jsonl for ``items`` into ``folder``; return the settings."""
    lines = (json.dumps({"input": text, "expected": label}, ensure_ascii=False) for text, label in items)
    (folder / "examples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    path = folder / "cifarium.yaml"
    path.write_text(f"task: classify\nexamples: examples.jsonl\n{settings}", encoding="utf-8")
    return path


@pytest.mark.parametrize("items", [ENGLISH, RUSSIAN], ids=["english", "russian"])
def test_a_set_has_three_labels_with_equal_counts_and_no_repeats(items):
    counts = Counter(label for _, label in items)
    assert len(counts) == 3
    assert len(set(counts.values())) == 1
    texts = [text.strip() for text, _ in items]
    assert len(set(texts)) == len(texts)


@pytest.mark.e2e
def test_build_chooses_all_minilm_for_the_english_set_and_writes_its_folder(english_build):
    assert english_build.code == 0, english_build.err
    output = english_build.out
    assert "Chosen: all-minilm-l6-v2 (Xenova/all-MiniLM-L6-v2, q8 on the processor" in output
    assert "crossOriginIsolated" in output and "not crossOriginIsolated" not in output
    assert "one thread    " in output and "not measured" not in output
    assert "speed limit   none" in output
    assert re.search(r"Wrote .*cifarium-out: \d+ files, [\d.]+ MB on disk; a first visit downloads [\d.]+ MB", output)
    # The folder of ARCHITECTURE.md, section 7, and the browser's copy of the model does not stay on disk.
    manifest = json.loads((english_build.folder / "cifarium.json").read_text(encoding="utf-8"))
    assert (manifest["format"], manifest["task"], manifest["recipe"]) == (1, "classify", "embed-centroid")
    assert manifest["model"]["repo"] == "Xenova/all-MiniLM-L6-v2"
    assert set(manifest["variants"]) == {"cpu"}
    assert [label["name"] for label in manifest["labels"]] == ["shipping", "returns", "technical"]
    assert not (english_build.folder / "data" / "labels.json").exists()
    assert (english_build.folder / "data" / "labels.bin").stat().st_size == 3 * 384 * 4
    total, files = bundle.first_load(manifest, "cpu")
    assert all((english_build.folder / name).stat().st_size > 0 for name in files)
    assert total == sum((english_build.folder / name).stat().st_size for name in files)
    # One weight: the selection counted it before the folder existed (the table, the line of the
    # chosen model, the limit), and the folder says the same afterwards, to the byte the tenth of a
    # megabyte shows.
    shown = f"{total / 1e6:.1f} MB"
    assert f"first load    {shown}: " in output and f"a first visit downloads {shown}" in output
    assert re.search(rf"all-minilm-l6-v2 +embed-centroid +{re.escape(shown)} +passed", output)
    for name in ("index.html", "cifarium.js", "cifarium-worker.js", "LICENSES.txt", "hosting/_headers",
                 "hosting/vercel.json", "hosting/nginx.conf"):
        assert (english_build.folder / name).is_file(), name
    assert not paths.profile_dir().exists()


@pytest.mark.e2e
def test_the_folder_of_a_real_build_holds_no_text_of_the_examples(english_build):
    # ARCHITECTURE.md, sections 7 and 14: everything in the folder may be published.
    assert english_build.code == 0, english_build.err

    assert example_leaks(english_build.folder, [text for text, _ in ENGLISH]) == []

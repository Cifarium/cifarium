"""Build the golden task from its two public sources. Standard library only.

    python examples/prepare.py [--sources FOLDER]

Downloads Banking77 and MASSIVE (into FOLDER, or a temporary folder), checks them against the
SHA-256 sums below, and writes ``english/`` and ``russian/`` next to this file: ``cifarium.yaml`` and
``examples.jsonl`` (20 examples to build from) and ``heldout.jsonl`` (100 the build never sees).
The same sources and the same SEED give the same files, byte for byte. README.md says where the
labels and the texts come from and what is not what it looks like.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import tarfile
import tempfile
import unicodedata
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SEED = 2026
"""Fixes every draw. The label of a draw is part of its seed, so changing one part leaves the other as it was."""

PER_LABEL = 30
BUILD_PER_LABEL = 5
"""Four labels of 30: 20 examples to build from (5 per label) and 100 held out."""

BANKING = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/"
MASSIVE = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
SOURCES = {
    "banking77-train.csv": (BANKING + "banking_data/train.csv",
                            "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"),
    "banking77-test.csv": (BANKING + "banking_data/test.csv",
                           "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"),
    "massive-1.1.tar.gz": (MASSIVE,
                           "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"),
}
MASSIVE_RUSSIAN = "1.1/data/ru-RU.jsonl"


@dataclass(frozen=True)
class Label:
    name: str
    description: str
    groups: tuple[str, ...]
    """The source intents the label is made of: the draw takes the same number from each."""


# The labels were fixed before any model saw the texts (README.md, "How the labels were chosen").
ENGLISH_LABELS = (
    Label("new_card", "ordering, delivery and activation of a new card",
          ("order_physical_card", "get_physical_card", "card_arrival", "card_delivery_estimate",
           "activate_my_card")),
    Label("card_security", "a lost, stolen, compromised or blocked card, and the PIN",
          ("lost_or_stolen_card", "compromised_card", "card_swallowed", "pin_blocked", "change_pin")),
    Label("transfers", "a bank transfer that is late, failed, lost or to be cancelled",
          ("pending_transfer", "failed_transfer", "transfer_not_received_by_recipient",
           "transfer_timing", "cancel_transfer")),
    Label("top_up", "adding money to the account: failed, reverted, pending or limited top-ups",
          ("top_up_failed", "top_up_reverted", "pending_top_up", "top_up_limits", "verify_top_up")),
)
RUSSIAN_LABELS = (
    Label("будильник", "поставить, проверить или убрать будильник",
          ("alarm_set", "alarm_query", "alarm_remove")),
    Label("календарь", "события и встречи в календаре",
          ("calendar_set", "calendar_query", "calendar_remove")),
    Label("почта", "письма: прочитать, отправить, найти контакт",
          ("email_query", "email_sendemail", "email_querycontact", "email_addcontact")),
    Label("соцсети", "публикации и сообщения в соцсетях", ("social_post", "social_query")),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sources", type=Path, help="folder for the downloads; kept, and reused when it has them")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    if args.sources:
        args.sources.mkdir(parents=True, exist_ok=True)
        build(args.sources, here)
    else:
        with tempfile.TemporaryDirectory() as folder:
            build(Path(folder), here)


def build(sources: Path, here: Path) -> None:
    files = {name: fetch(sources / name, url, digest) for name, (url, digest) in SOURCES.items()}
    english = banking77(files["banking77-train.csv"], files["banking77-test.csv"])
    russian = massive(files["massive-1.1.tar.gz"])
    write_part(here / "english", ENGLISH_LABELS, draw("english", ENGLISH_LABELS, english))
    write_part(here / "russian", RUSSIAN_LABELS, draw("russian", RUSSIAN_LABELS, russian))


def fetch(target: Path, url: str, digest: str) -> bytes:
    """The bytes of a source, downloaded when ``target`` is not there yet and checked against ``digest``."""
    if not target.is_file():
        with urllib.request.urlopen(url, timeout=300) as response:
            target.write_bytes(response.read())
    data = target.read_bytes()
    found = hashlib.sha256(data).hexdigest()
    if found != digest:
        raise SystemExit(f"{target.name}: sha256 {found} is not the pinned {digest}; "
                         "the source changed, so the set would too")
    return data


@dataclass(frozen=True)
class Text:
    text: str
    group: str
    """The intent or scenario it came from."""
    source: str
    """Where to find it again."""


def clean(text: str) -> str:
    return unicodedata.normalize("NFC", " ".join(text.split()))


def banking77(train: bytes, test: bytes) -> list[Text]:
    rows: list[Text] = []
    for data in (train, test):
        for row in csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")):
            rows.append(Text(clean(row["text"]), row["category"], f"banking77/{row['category']}"))
    return unambiguous(rows)


def massive(archive: bytes) -> list[Text]:
    """The training part of ru-RU that all three annotators took for its intent. A text's group is its
    intent, such as ``alarm_set``; the scenario (``alarm``) is the start of the name and is what the
    labels stand for."""
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        member = tar.extractfile(MASSIVE_RUSSIAN)
        assert member is not None
        lines = member.read().decode("utf-8").splitlines()
    rows = []
    for line in lines:
        row = json.loads(line)
        # Three annotators each said whether the text fits its intent (1 yes, 0 no, 2 unsure).
        # Only texts all three took are used: the source labels some of the rest wrongly.
        if row["partition"] == "train" and all(j["intent_score"] == 1 for j in row["judgments"]):
            rows.append(Text(clean(row["utt"]), row["intent"],
                             f"massive/ru-RU/{row['intent']}/{row['id']}"))
    return unambiguous(rows)


def unambiguous(rows: list[Text]) -> list[Text]:
    """Drop empty texts, repeats, and texts that the source gives to more than one group."""
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[row.text].add(row.group)
    seen: set[str] = set()
    kept = []
    for row in rows:
        if row.text and len(groups[row.text]) == 1 and row.text not in seen:
            seen.add(row.text)
            kept.append(row)
    return kept


def draw(part: str, labels: tuple[Label, ...], rows: list[Text]) -> list[tuple[Text, str, bool]]:
    """30 texts per label, spread evenly over the label's source groups, and which of them to build from.

    Returns (text, label, for_build) in the order of the files: shuffled, so no label comes in a block.
    """
    by_group: dict[str, list[Text]] = defaultdict(list)
    for row in rows:
        by_group[row.group].append(row)
    drawn: list[tuple[Text, str, bool]] = []
    for label in labels:
        rng = random.Random(f"{SEED}/{part}/{label.name}")
        extra = set(rng.sample(range(len(label.groups)), PER_LABEL % len(label.groups)))
        chosen: list[Text] = []
        for number, group in enumerate(label.groups):
            count = PER_LABEL // len(label.groups) + (number in extra)
            chosen += rng.sample(sorted(by_group[group], key=lambda row: row.source), count)
        build = set(rng.sample(range(len(chosen)), BUILD_PER_LABEL))
        drawn += [(row, label.name, number in build) for number, row in enumerate(chosen)]
    random.Random(f"{SEED}/{part}/order").shuffle(drawn)
    return drawn


def write_part(folder: Path, labels: tuple[Label, ...], drawn: list[tuple[Text, str, bool]]) -> None:
    folder.mkdir(exist_ok=True)
    for name, wanted in (("examples.jsonl", True), ("heldout.jsonl", False)):
        lines = [json.dumps({"input": row.text, "expected": label, "source": row.source}, ensure_ascii=False)
                 for row, label, for_build in drawn if for_build is wanted]
        (folder / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    header = "task: classify\nexamples: examples.jsonl\nlabels:\n"
    # A JSON string is a YAML string too, and a description may hold a colon.
    body = "".join(f"  {label.name}: {json.dumps(label.description, ensure_ascii=False)}\n" for label in labels)
    (folder / "cifarium.yaml").write_text(header + body, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare expanded-v2 without changing frozen data (Python standard library only).

All records, including quality failures, participate in disk-backed duplicate/group
components BEFORE selection. Frozen evaluation and BanglishRev components exclude
new material; baseline-train components cannot become extra evaluation. Other
components use SHA256('bn-corpus-v2:' + smallest original group), modulo 100:
0..1 test, 2..3 validation, otherwise train. Original group names are never changed.

Near duplicates use the existing Go 64-bit word SimHash, >=12 whitespace words,
Hamming distance <=3. Five disjoint bands (13/13/13/13/12 bits), indexed by
all ten band-pairs, guarantee candidate coverage: at most three bands can differ,
so two remain identical. Pair keys are 25/26 bits rather than 16, avoiding billions
of incidental comparisons at multi-million-record scale. There is no sampling or
bucket cap. Bit-sliced counters compute all 64 word-bit majorities together.
Canonical keys retain Unicode letters, numbers AND marks, lowercased per rune.

New records additionally require 40 characters and five words (or 60 letters), so
short fragments cannot satisfy growth. Quality thresholds are recorded in stats.
SQLite cache and bounded LRU caches limit RAM independently of corpus size; disk
and runtime grow with input and matching-band candidates. Pathological shared-band
sets can still be expensive; no approximate shortcut silently weakens protection.
The work database is created exclusively; remove it explicitly before a fresh run.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import tempfile
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

FIELDS = ("text", "bucket", "source", "group", "attribution", "split")
BUCKETS = {"bangla", "english", "banglish", "code-mixed"}
DEFAULT_BASELINE = Path("data/processed/frozen-v1")
DEFAULT_RESERVED = Path("data/raw/banglishrev/reviews-v1-head64m.json")
FROZEN_HASHES = {
    "train": "bda48c8388fcd1cd706b4578619367a609ccf75a7f3aa659fede8c88aa198b39",
    "validation": "047e0acb94d621508467fcd0e7b3369204f7e040d4b9144b58614faac9f1cc2d",
    "test": "3f39a8faf7ecc280ec32ce231d379cec788eb79a52d9a6e66917073500879834",
}
# Go unicode.IsSpace, unlike Python str.split(), excludes U+001C..U+001F.
SPACE = re.compile(r"[\t-\r \u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+")
PRIVATE = re.compile(
    r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}|"
    r"(?:\+?880|\b0)1[3-9][0-9]{8}\b|"
    r"(?:api[_-]?key|password|secret)\s*[:=]\s*\S+|"
    r"\b(?:\+?\d[ .()\-]*){10,15}\b",
    re.I,
)
URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
BOILERPLATE = re.compile(
    r"(?:enable javascript|accept all cookies|cookie (?:policy|preferences)|"
    r"all rights reserved|subscribe to our newsletter|click here to (?:login|sign up))",
    re.I,
)
JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
MASK = (1 << 64) - 1
BASE_TRAIN, FROZEN_EVAL, RESERVED = 1, 2, 4
NEAR_METHOD = {
    "algorithm": "Go-compatible 64-bit word SHA256 SimHash; first 8 bytes little-endian",
    "minimum_words": 12,
    "maximum_hamming_distance": 3,
    "candidate_index": (
        "all ten pairs of five disjoint 13/13/13/13/12-bit bands; exhaustive at radius 3, no caps"
    ),
    "short_text_coverage": "canonical exact and group protection only below 12 words",
    "training_removal": (
        "deterministic greedy representatives, baseline first then canonical SHA256"
    ),
    "group_protection": "transitive exact/near/group components, including quality-rejected input",
}


def normalize(text: str) -> str:
    """NFC plus precisely Go's Unicode whitespace set; retain marks and joiners."""
    # JSON can encode lone UTF-16 surrogates; Go's JSON decoder replaces them.
    if any(0xD800 <= ord(c) <= 0xDFFF for c in text):
        text = re.sub("[\ud800-\udfff]", "\ufffd", text)
    return SPACE.sub(" ", unicodedata.normalize("NFC", text)).strip(" ")


def lower_runes(text: str) -> str:
    # Go uses simple rune lowercasing, not Python's context-sensitive full mapping.
    # U+0130 is the only expanding lowercase mapping; its simple mapping is 'i'.
    return "".join(c.lower()[0] for c in text)


def canonical(text: str) -> bytes:
    key = "".join(c.lower()[0] for c in text if unicodedata.category(c)[0] in "LNM")
    return hashlib.sha256(key.encode("utf-8")).digest()


@lru_cache(maxsize=65536)
def word_hash(word: str) -> int:
    return int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:8], "little")


def simhash(text: str) -> int | None:
    words = lower_runes(text).split(" ")
    if len(words) < 12:
        return None
    counters: list[int] = []
    for word in words:
        carry = word_hash(word)
        bit = 0
        while carry:
            if bit == len(counters):
                counters.append(carry)
                break
            carry, counters[bit] = counters[bit] & carry, counters[bit] ^ carry
            bit += 1
    threshold = (len(words) + 1) // 2
    greater, equal = 0, MASK
    for bit in range(max(len(counters), threshold.bit_length()) - 1, -1, -1):
        count = counters[bit] if bit < len(counters) else 0
        if threshold & (1 << bit):
            equal &= count
        else:
            greater |= equal & count
            equal &= ~count & MASK
    return greater | equal


def bands(signature: int):
    parts = [
        (signature >> shift) & ((1 << width) - 1)
        for shift, width in ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))
    ]
    tag = 0
    for first in range(4):
        for second in range(first + 1, 5):
            yield (tag << 26) | parts[first] | (parts[second] << 13)
            tag += 1


def quality(text: str, record: dict, baseline: bool) -> str:
    if not 12 <= len(text) <= 2000:
        return "length"
    if "\ufffd" in text or any(unicodedata.category(c) in {"Cs", "Co", "Cn"} for c in text):
        return "replacement_or_invalid_unicode"
    if any(
        unicodedata.category(c) == "Cc"
        or (unicodedata.category(c) == "Cf" and c not in "\u200c\u200d")
        for c in text
    ):
        return "control_corruption"
    if PRIVATE.search(text):
        return "private_information"
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if len(letters) / max(1, len(text.replace(" ", ""))) < 0.45:
        return "low_letter_ratio"
    words = text.split(" ")
    if not baseline and (len(text) < 40 or (len(words) < 5 and len(letters) < 60)):
        return "short_new_fragment"
    if (
        sum(len(m.group()) for m in URL.finditer(text)) > len(text) * 0.15
        or len(URL.findall(text)) > 1
    ):
        return "url_heavy"
    if BOILERPLATE.search(text) or re.search(r"<\s*/?\s*(?:html|script|div|iframe)\b", text, re.I):
        return "boilerplate"
    if re.search(r"(.)\1{7,}", text) or re.search(r"(.{2,30})\1{4,}", text):
        return "character_repetition"
    tokens = [lower_runes(w) for w in words]
    if len(tokens) >= 8 and max(Counter(tokens).values()) / len(tokens) > 0.35:
        return "word_repetition"
    if len(tokens) >= 12:
        triples = Counter(tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2))
        if sum(n - 1 for n in triples.values()) / (len(tokens) - 2) > 0.5:
            return "phrase_repetition"
    bn = sum("\u0980" <= c <= "\u09ff" for c in letters)
    latin = sum("LATIN" in unicodedata.name(c, "") for c in letters)
    bucket = record["bucket"]
    if (
        bucket == "bangla"
        and bn / len(letters) < 0.5
        or bucket in {"english", "banglish"}
        and latin / len(letters) < 0.8
        or bucket == "code-mixed"
        and (not bn or not latin)
    ):
        return "bucket_script_mismatch"
    source = record["source"].lower()
    if (
        record["group"].startswith("wiki-en:")
        and latin / len(letters) < 0.8
        or record["group"].startswith("wiki-bn:")
        and bn / len(letters) < 0.5
        or "wiki-en" in source
        and bucket == "banglish"
    ):
        return "source_language_mismatch"
    return ""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path: Path):
    with path.open(encoding="utf-8", errors="strict") as handle:
        for number, line in enumerate(iter(lambda: handle.readline(2 * 1024 * 1024), ""), 1):
            if len(line) == 2 * 1024 * 1024 and not line.endswith("\n"):
                raise ValueError(f"{path}:{number}: JSONL record exceeds 2 MiB")
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or any(
                    not isinstance(record.get(f, ""), str) for f in FIELDS
                ):
                    raise ValueError("all six fields must be strings")
                for field in FIELDS:
                    record.setdefault(field, "")
                if record["bucket"] not in BUCKETS or any(
                    not record[f] for f in ("text", "group", "source", "attribution")
                ):
                    raise ValueError("requires text, known bucket, group, source and attribution")
                yield {field: record[field] for field in FIELDS}
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error


def reserved_texts(path: Path):
    """Stream strings even from the documented head-truncated evaluation JSON.

    Reserve review bodies AND seller replies. Only complete JSON strings are used;
    an incomplete trailing string is ignored, never used as training material.
    A 2 MiB token limit makes malformed/unexpected input fail rather than grow RAM.
    """
    pending = ""
    want_value = False
    with path.open(encoding="utf-8", errors="strict") as handle:
        while True:
            chunk = handle.read(65536)
            pending += chunk
            end = 0
            while True:
                start = pending.find('"', end)
                if start < 0:
                    end = len(pending)
                    break
                match = JSON_STRING.match(pending, start)
                if match is None:
                    end = start
                    break
                value = json.loads(match.group())
                if want_value:
                    yield value
                    want_value = False
                elif value in {"Review Content", "Reply", "text"}:
                    want_value = True
                end = match.end()
            pending = pending[end:]
            if len(pending) > 2 * 1024 * 1024:
                raise ValueError("reserved evaluation contains a JSON token larger than 2 MiB")
            if not chunk:
                break


class Index:
    def __init__(self, path: Path):
        # Exclusive creation prevents accidental reuse of stale decisions.
        with path.open("xb"):
            pass
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            PRAGMA cache_size=-131072;
            CREATE TABLE groups(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
                                parent INTEGER NOT NULL, flags INTEGER NOT NULL,
                                size INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE records(key BLOB PRIMARY KEY, group_id INTEGER NOT NULL,
                text TEXT NOT NULL, bucket TEXT NOT NULL, source TEXT NOT NULL,
                attribution TEXT NOT NULL, baseline INTEGER NOT NULL, reason TEXT NOT NULL,
                rank BLOB NOT NULL, signature BLOB) WITHOUT ROWID;
            CREATE TABLE signatures(signature BLOB PRIMARY KEY, group_id INTEGER NOT NULL)
                WITHOUT ROWID;
            CREATE TABLE bands(band INTEGER NOT NULL, signature BLOB NOT NULL,
                               PRIMARY KEY(band, signature)) WITHOUT ROWID;
        """)
        self.group_id = lru_cache(maxsize=32768)(self._group_id)
        self.exact_links = 0
        self.near_links = 0

    def _group_id(self, name: str) -> int:
        row = self.db.execute("SELECT id FROM groups WHERE name=?", (name,)).fetchone()
        if row:
            return row[0]
        cursor = self.db.execute("INSERT INTO groups(name,parent,flags) VALUES (?,0,0)", (name,))
        value = cursor.lastrowid
        self.db.execute("UPDATE groups SET parent=? WHERE id=?", (value, value))
        return value

    def root(self, group: int) -> int:
        trail = []
        while True:
            parent = self.db.execute("SELECT parent FROM groups WHERE id=?", (group,)).fetchone()[0]
            if parent == group:
                break
            trail.append(group)
            group = parent
        for child in trail:
            self.db.execute("UPDATE groups SET parent=? WHERE id=?", (group, child))
        return group

    def join(self, first: int, second: int) -> bool:
        first, second = self.root(first), self.root(second)
        if first == second:
            return False
        sizes = dict(
            self.db.execute("SELECT id,size FROM groups WHERE id IN (?,?)", (first, second))
        )
        if (sizes[first], -first) < (sizes[second], -second):
            first, second = second, first
        self.db.execute(
            "UPDATE groups SET flags=flags | (SELECT flags FROM groups WHERE id=?), "
            "size=size+? WHERE id=?",
            (second, sizes[second], first),
        )
        self.db.execute("UPDATE groups SET parent=? WHERE id=?", (first, second))
        return True

    def mark(self, group: int, flag: int):
        self.db.execute("UPDATE groups SET flags=flags | ? WHERE id=?", (flag, self.root(group)))

    def signature(self, signature: int | None, group: int):
        if signature is None:
            return
        raw = signature.to_bytes(8, "little")
        previous = self.db.execute(
            "SELECT group_id FROM signatures WHERE signature=?", (raw,)
        ).fetchone()
        if previous:
            self.near_links += self.join(group, previous[0])
            return
        # Stream matching bands rather than materializing dense buckets in RAM.
        for band in bands(signature):
            for other, owner in self.db.execute(
                "SELECT b.signature,s.group_id FROM bands b "
                "JOIN signatures s USING(signature) WHERE band=?",
                (band,),
            ):
                if (signature ^ int.from_bytes(other, "little")).bit_count() <= 3:
                    self.near_links += self.join(group, owner)
        self.db.execute("INSERT INTO signatures VALUES (?,?)", (raw, group))
        self.db.executemany(
            "INSERT INTO bands VALUES (?,?)", ((band, raw) for band in bands(signature))
        )

    def stage(self, record: dict, origin: str, counts: Counter, reasons: Counter):
        text = normalize(record["text"])
        group = self.group_id(record["group"])
        flag = (
            BASE_TRAIN
            if origin == "train"
            else FROZEN_EVAL
            if origin in {"validation", "test"}
            else RESERVED
            if origin == "reserved"
            else 0
        )
        if flag:
            self.mark(group, flag)
        key = canonical(text)
        signature = simhash(text)
        reason = (
            quality(text, record, origin == "train")
            if origin in {"train", "candidate"}
            else "evaluation_only"
        )
        counts[(origin, record["source"], record["bucket"])] += 1
        if reason and origin in {"train", "candidate"}:
            reasons[(origin, record["source"], record["bucket"], reason)] += 1
        # Prefer quality-accepted original training provenance, then deterministic JSON.
        rank = (
            bytes([bool(reason), origin != "train"])
            + hashlib.sha256(
                json.dumps(record, ensure_ascii=True, sort_keys=True).encode()
            ).digest()
        )
        previous = self.db.execute(
            "SELECT group_id,rank FROM records WHERE key=?", (key,)
        ).fetchone()
        if previous:
            self.exact_links += self.join(group, previous[0])
        if not previous or rank < previous[1]:
            self.db.execute(
                "INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    key,
                    group,
                    text,
                    record["bucket"],
                    record["source"],
                    record["attribution"],
                    origin == "train",
                    reason,
                    rank,
                    signature.to_bytes(8, "little") if signature is not None else None,
                ),
            )
        self.signature(signature, group)

    def components(self):
        self.db.execute(
            "CREATE TABLE components(root INTEGER PRIMARY KEY, name TEXT NOT NULL, "
            "flags INTEGER NOT NULL)"
        )
        self.db.execute("CREATE TABLE membership(id INTEGER PRIMARY KEY, root INTEGER NOT NULL)")
        for group, name in self.db.execute("SELECT id,name FROM groups ORDER BY id"):
            root = self.root(group)
            self.db.execute("INSERT INTO membership VALUES (?,?)", (group, root))
            flags = self.db.execute("SELECT flags FROM groups WHERE id=?", (root,)).fetchone()[0]
            self.db.execute(
                "INSERT INTO components VALUES (?,?,?) ON CONFLICT(root) "
                "DO UPDATE SET name=min(name,excluded.name)",
                (root, name, flags),
            )
        self.db.execute("CREATE INDEX records_order ON records(baseline DESC,key)")
        self.db.commit()


class NearIndex:
    """Separate disk index for exported representatives or independent audit."""

    def __init__(self, db: sqlite3.Connection, prefix: str):
        self.db, self.prefix = db, prefix
        db.execute(
            f"CREATE TABLE {prefix}(band INTEGER, signature BLOB, split TEXT, "
            "PRIMARY KEY(band,signature,split)) WITHOUT ROWID"
        )

    def matches(self, signature: int | None):
        if signature is None:
            return
        seen = set()
        for band in bands(signature):
            for raw, split in self.db.execute(
                f"SELECT signature,split FROM {self.prefix} WHERE band=?", (band,)
            ):
                if (
                    split not in seen
                    and (signature ^ int.from_bytes(raw, "little")).bit_count() <= 3
                ):
                    seen.add(split)
                    yield split

    def add(self, signature: int | None, split: str):
        if signature is not None:
            raw = signature.to_bytes(8, "little")
            self.db.executemany(
                f"INSERT OR IGNORE INTO {self.prefix} VALUES (?,?,?)",
                ((band, raw, split) for band in bands(signature)),
            )


def component_split(name: str) -> str:
    value = (
        int.from_bytes(hashlib.sha256(("bn-corpus-v2:" + name).encode()).digest()[:8], "big") % 100
    )
    return "extra-test" if value < 2 else "extra-validation" if value < 4 else "train"


def export(index: Index, output: Path):
    counts, characters, rejected = Counter(), Counter(), Counter()
    near = NearIndex(index.db, "selected_near")
    paths = {
        split: output / f"{split}.jsonl" for split in ("train", "extra-validation", "extra-test")
    }
    handles = {
        split: path.open("w", encoding="utf-8", newline="\n") for split, path in paths.items()
    }
    query = """SELECT r.text,r.bucket,r.source,g.name,r.attribution,r.baseline,
                      r.reason,r.signature,c.name,c.flags
               FROM records r JOIN groups g ON g.id=r.group_id
               JOIN membership m ON m.id=g.id JOIN components c ON c.root=m.root
               ORDER BY r.baseline DESC,r.key"""
    try:
        for (
            text,
            bucket,
            source,
            group,
            attribution,
            baseline,
            reason,
            raw,
            component,
            flags,
        ) in index.db.execute(query):
            if reason:
                rejected[
                    (
                        "train" if baseline else "candidate",
                        source,
                        bucket,
                        "quality_rejected_canonical_representative",
                    )
                ] += 1
                continue
            if flags & (FROZEN_EVAL | RESERVED):
                rejected[
                    (
                        "train" if baseline else "candidate",
                        source,
                        bucket,
                        "protected_evaluation_component",
                    )
                ] += 1
                continue
            split = "train" if flags & BASE_TRAIN else component_split(component)
            signature = int.from_bytes(raw, "little") if raw is not None else None
            if next(near.matches(signature), None) is not None:
                rejected[
                    (
                        "train" if baseline else "candidate",
                        source,
                        bucket,
                        "near_duplicate_representative",
                    )
                ] += 1
                continue
            near.add(signature, split)
            record = dict(
                zip(
                    FIELDS,
                    (text, bucket, source, group, attribution, split.removeprefix("extra-")),
                    strict=True,
                )
            )
            handles[split].write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            counts[(split, source, bucket)] += 1
            characters[(split, source, bucket)] += len(text)
    finally:
        for handle in handles.values():
            handle.close()
    index.db.commit()
    return counts, characters, rejected


def distribution(counter: Counter):
    return [
        {"split": key[0], "source": key[1], "bucket": key[2], "count": count}
        for key, count in sorted(counter.items())
    ]


def audit(
    output: Path, reserved: Path, db: sqlite3.Connection, baseline_hashes: dict, baseline_dir: Path
):
    """Re-read saved files and independently reconstruct overlap/size facts."""
    db.executescript("""
        CREATE TABLE audit_keys(key BLOB, split TEXT, PRIMARY KEY(key,split)) WITHOUT ROWID;
        CREATE TABLE audit_groups(name TEXT, split TEXT, PRIMARY KEY(name,split)) WITHOUT ROWID;
        CREATE TABLE original_train_keys(key BLOB PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE original_train_groups(name TEXT PRIMARY KEY) WITHOUT ROWID;
    """)
    near = NearIndex(db, "audit_near")
    original_near = NearIndex(db, "original_train_near")
    for record in read_records(baseline_dir / "train.jsonl"):
        text = normalize(record["text"])
        db.execute("INSERT OR IGNORE INTO original_train_keys VALUES (?)", (canonical(text),))
        db.execute("INSERT OR IGNORE INTO original_train_groups VALUES (?)", (record["group"],))
        original_near.add(simhash(text), "original-train")
    overlap, counts, characters, coverage = Counter(), Counter(), Counter(), Counter()
    unique_train_characters = 0
    hashes = {}
    for split in ("validation", "test", "train", "extra-validation", "extra-test"):
        path = output / f"{split}.jsonl"
        hashes[split] = digest_file(path)
        for record in read_records(path):
            text = normalize(record["text"])
            key, signature = canonical(text), simhash(text)
            if split.startswith("extra-"):
                if db.execute("SELECT 1 FROM original_train_keys WHERE key=?", (key,)).fetchone():
                    overlap["extra_eval_original_train_canonical"] += 1
                if db.execute(
                    "SELECT 1 FROM original_train_groups WHERE name=?", (record["group"],)
                ).fetchone():
                    overlap["extra_eval_original_train_group"] += 1
                if next(original_near.matches(signature), None) is not None:
                    overlap["extra_eval_original_train_near"] += 1
            existing = [
                row[0] for row in db.execute("SELECT split FROM audit_keys WHERE key=?", (key,))
            ]
            if split == "train" and split not in existing:
                unique_train_characters += len(text)
            for other in existing:
                if other != split:
                    overlap["canonical_cross_split"] += 1
                elif split == "train":
                    overlap["training_canonical_duplicates"] += 1
            others = [
                row[0]
                for row in db.execute(
                    "SELECT split FROM audit_groups WHERE name=?", (record["group"],)
                )
            ]
            overlap["group_cross_split"] += sum(other != split for other in others)
            for other in near.matches(signature):
                if other != split:
                    overlap["near_cross_split"] += 1
                elif split == "train":
                    overlap["training_near_duplicates"] += 1
            db.execute("INSERT OR IGNORE INTO audit_keys VALUES (?,?)", (key, split))
            db.execute("INSERT OR IGNORE INTO audit_groups VALUES (?,?)", (record["group"], split))
            near.add(signature, split)
            counts[(split, record["source"], record["bucket"])] += 1
            characters[(split, record["source"], record["bucket"])] += len(text)
            coverage[
                (split, "simhash_eligible" if signature is not None else "canonical_only")
            ] += 1
    for text in reserved_texts(reserved):
        text = normalize(text)
        for (split,) in db.execute("SELECT split FROM audit_keys WHERE key=?", (canonical(text),)):
            if split not in {"validation", "test"}:
                overlap["reserved_canonical"] += 1
        for split in near.matches(simhash(text)):
            if split not in {"validation", "test"}:
                overlap["reserved_near"] += 1
    for split in ("validation", "test"):
        if hashes[split] != baseline_hashes[split]:
            overlap["frozen_hash_mismatch"] += 1
    db.commit()
    return {
        "method": "independent reread of emitted files; rebuilt canonical/group/SimHash indexes",
        "accepted_unique_training_records": db.execute(
            "SELECT count(*) FROM audit_keys WHERE split='train'"
        ).fetchone()[0],
        "accepted_training_normalized_characters": unique_train_characters,
        "overlaps": {
            name: overlap[name]
            for name in (
                "canonical_cross_split",
                "training_canonical_duplicates",
                "training_near_duplicates",
                "group_cross_split",
                "near_cross_split",
                "reserved_canonical",
                "reserved_near",
                "frozen_hash_mismatch",
                "extra_eval_original_train_canonical",
                "extra_eval_original_train_group",
                "extra_eval_original_train_near",
            )
        },
        "near_duplicate_method": NEAR_METHOD,
        "near_duplicate_coverage": [
            {"split": key[0], "coverage": key[1], "count": n} for key, n in sorted(coverage.items())
        ],
        "source_bucket_records": distribution(counts),
        "source_bucket_characters": distribution(characters),
        "split_sha256": hashes,
        "frozen_sha256": baseline_hashes,
        "reserved_evaluation_sha256": digest_file(reserved),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    result.add_argument(
        "--candidates",
        action="append",
        required=True,
        help="repeatable JSONL path/glob (shell quote globs)",
    )
    result.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    result.add_argument("--output", type=Path, default=Path("data/processed/expanded-v2"))
    result.add_argument("--min-train-records", type=int, default=2817380)
    result.add_argument("--min-train-characters", type=int, default=329839300)
    result.add_argument(
        "--work-db", type=Path, help="new SQLite file; temporary database otherwise"
    )
    result.add_argument(
        "--reserved-eval",
        type=Path,
        default=DEFAULT_RESERVED,
        help="BanglishRev exclusion-only JSON; required even for small fixtures",
    )
    return result


def prepare(args: argparse.Namespace) -> bool:
    candidates = set()
    for pattern in args.candidates:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            raise ValueError(f"candidate path/glob has no matches: {pattern}")
        candidates.update(Path(match).resolve() for match in matches)
    candidates = sorted(candidates)
    baseline_hashes = {
        split: digest_file(args.baseline_dir / f"{split}.jsonl") for split in FROZEN_HASHES
    }
    if (
        args.baseline_dir.resolve() == DEFAULT_BASELINE.resolve()
        and baseline_hashes != FROZEN_HASHES
    ):
        raise ValueError("frozen-v1 baseline fingerprints do not match the frozen contract")
    if not args.reserved_eval.is_file():
        raise ValueError(f"reserved evaluation exclusion file is required: {args.reserved_eval}")
    if args.output.resolve() == args.baseline_dir.resolve():
        raise ValueError("output must not be the frozen baseline directory")
    args.output.mkdir(parents=True, exist_ok=True)
    destinations = [
        args.output / f"{split}.jsonl"
        for split in ("train", "validation", "test", "extra-validation", "extra-test")
    ]
    if any(path.exists() for path in destinations):
        raise ValueError("output corpus files already exist; choose a fresh output directory")
    forbidden = {path.resolve() for path in destinations} | {
        (args.baseline_dir / f"{split}.jsonl").resolve() for split in FROZEN_HASHES
    }
    if any(path in forbidden for path in candidates):
        raise ValueError("candidate paths must not include baseline or output files")
    before, reasons = Counter(), Counter()
    with tempfile.TemporaryDirectory(prefix="banglatok-corpus-v2-") as temporary:
        work = args.work_db or Path(temporary) / "work.sqlite3"
        index = Index(work)
        try:
            for split in FROZEN_HASHES:
                for record in read_records(args.baseline_dir / f"{split}.jsonl"):
                    if record["split"] and record["split"] != split:
                        raise ValueError(f"baseline record has inconsistent {split} split")
                    index.stage(record, split, before, reasons)
                index.db.commit()
            reserved_count = 0
            for reserved_count, text in enumerate(reserved_texts(args.reserved_eval), 1):
                record = dict(
                    zip(
                        FIELDS,
                        (
                            text,
                            "code-mixed",
                            "reserved-banglishrev",
                            f"reserved-banglishrev:{reserved_count}",
                            "exclusion-only; never training",
                            "",
                        ),
                        strict=True,
                    )
                )
                index.stage(record, "reserved", before, reasons)
                if reserved_count % 10000 == 0:
                    index.db.commit()
            if not reserved_count:
                raise ValueError("reserved evaluation contained no recognized complete text fields")
            candidate_hashes = {}
            for path in candidates:
                candidate_hashes[str(path)] = digest_file(path)
                for number, record in enumerate(read_records(path), 1):
                    if record["split"]:
                        raise ValueError(
                            f"{path}:{number}: new candidate split must be empty; "
                            "official holdouts cannot train"
                        )
                    if (
                        "banglishrev"
                        in (record["source"] + record["group"] + record["attribution"]).lower()
                    ):
                        raise ValueError(
                            f"{path}:{number}: BanglishRev is prohibited as candidate material"
                        )
                    index.stage(record, "candidate", before, reasons)
                    if number % 10000 == 0:
                        index.db.commit()
                index.db.commit()
            index.components()
            after, chars, rejected = export(index, args.output)
            for split in ("validation", "test"):
                shutil.copyfile(
                    args.baseline_dir / f"{split}.jsonl", args.output / f"{split}.jsonl"
                )
            report = audit(
                args.output, args.reserved_eval, index.db, baseline_hashes, args.baseline_dir
            )
            gates = {
                "minimum_training_records": report["accepted_unique_training_records"]
                >= args.min_train_records,
                "minimum_training_characters": report["accepted_training_normalized_characters"]
                >= args.min_train_characters,
                "no_leakage": not any(report["overlaps"].values()),
            }
            report["gates"] = gates
            stats = {
                "schema_version": 2,
                "normalization": (
                    "nfc-whitespace-v1; Go unicode.IsSpace; "
                    "per-rune simple lowercase L/N/M canonical SHA256"
                ),
                "unicode_version": unicodedata.unidata_version,
                "unicode_table_limitations": (
                    "NFC, categories and simple lowercase use this Python Unicode version; "
                    "edge fixtures verify semantics, not identity with a different Go "
                    "Unicode table version."
                ),
                "selection_policy": __doc__,
                "quality_policy": {
                    "characters": [12, 2000],
                    "new_minimum_characters": 40,
                    "new_minimum_words_or_letters": [5, 60],
                    "minimum_letter_ratio": 0.45,
                    "max_url_character_ratio": 0.15,
                    "max_urls": 1,
                    "max_word_frequency_ratio_at_8_words": 0.35,
                    "max_repeated_trigram_ratio_at_12_words": 0.5,
                    "bangla_min_bn_letter_ratio": 0.5,
                    "english_banglish_min_latin_letter_ratio": 0.8,
                    "joiners_preserved": True,
                },
                "input_source_bucket_records": distribution(before),
                "quality_rejections": [
                    {
                        "origin": key[0],
                        "source": key[1],
                        "bucket": key[2],
                        "reason": key[3],
                        "count": n,
                    }
                    for key, n in sorted(reasons.items())
                ],
                "selection_rejections": [
                    {
                        "origin": key[0],
                        "source": key[1],
                        "bucket": key[2],
                        "reason": key[3],
                        "count": n,
                    }
                    for key, n in sorted(rejected.items())
                ],
                "canonical_records_staged": index.db.execute(
                    "SELECT count(*) FROM records"
                ).fetchone()[0],
                "canonical_duplicates_removed_during_staging": sum(before.values())
                - index.db.execute("SELECT count(*) FROM records").fetchone()[0],
                "exact_group_links": index.exact_links,
                "near_group_links": index.near_links,
                "candidate_source_sha256": candidate_hashes,
                "reserved_complete_text_strings": reserved_count,
                "frozen_sha256": baseline_hashes,
                "output_source_bucket_records": distribution(after),
                "output_source_bucket_characters": distribution(chars),
                "growth_requirements": {
                    "records": args.min_train_records,
                    "normalized_characters": args.min_train_characters,
                },
                "near_duplicate_method": NEAR_METHOD,
                "gates": gates,
            }
            for name, payload in (("stats.json", stats), ("audit.json", report)):
                (args.output / name).write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(
                json.dumps(
                    {
                        "accepted_unique_training_records": report[
                            "accepted_unique_training_records"
                        ],
                        "accepted_training_normalized_characters": report[
                            "accepted_training_normalized_characters"
                        ],
                        "gates": gates,
                    },
                    sort_keys=True,
                )
            )
            return all(gates.values())
        finally:
            index.db.close()


def main() -> int:
    args = parser().parse_args()
    try:
        return 0 if prepare(args) else 1
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"prepare_corpus_v2: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

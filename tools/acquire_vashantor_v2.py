#!/usr/bin/env python3
"""Convert pinned Vashantor v2 forms into corpus-v2 candidates and holdouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import tempfile
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

ARCHIVE_SHA256 = "5abb437220ad6fede5344b974f1abe48f4be8d5a9c250b807c0fea6fe4899919"
ARCHIVE_BYTES = 1_073_445
EXPECTED_ROWS = 12_500
DOI = "10.17632/bj5jgk878b.2"
ATTRIBUTION = (
    "Vashantor v2, DOI 10.17632/bj5jgk878b.2; CC BY 4.0; "
    "author-supplied human translations and source utterances"
)
SPACE = " \t\n\r\v\f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).strip(SPACE).split())


def canonical(text: str) -> bytes:
    key = "".join(c.lower()[0] for c in text if unicodedata.category(c)[0] in "LNM")
    return hashlib.sha256(key.encode()).digest()


def records(archive: Path):
    rows = 0
    with zipfile.ZipFile(archive) as zipped:
        for name in sorted(zipped.namelist()):
            if not name.endswith(".csv"):
                continue
            split = Path(name).parent.name.lower()
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"unknown Vashantor split in {name}")
            with zipped.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
                for number, fields in enumerate(reader, 2):
                    row = {key.strip(): normalize(value) for key, value in fields.items()}
                    regional_bn = [key for key in row if key.endswith("_bangla_speech")]
                    regional_bl = [key for key in row if key.endswith("_banglish_speech")]
                    if len(row) != 6 or len(regional_bn) != 1 or len(regional_bl) != 1:
                        raise ValueError(f"unexpected Vashantor schema at {name}:{number}")
                    forms = (
                        ("bangla", row["bangla_speech"]),
                        ("banglish", row["banglish_speech"]),
                        ("english", row["english_speech"]),
                        ("bangla", row[regional_bn[0]]),
                        ("banglish", row[regional_bl[0]]),
                    )
                    if any(not text for _, text in forms):
                        raise ValueError(f"empty Vashantor form at {name}:{number}")
                    group = f"vashantor-v2:{canonical(row['bangla_speech']).hex()}"
                    for bucket, text in forms:
                        yield (
                            split,
                            {
                                "text": text,
                                "bucket": bucket,
                                "source": "vashantor-v2",
                                "group": group,
                                "attribution": ATTRIBUTION,
                                "split": "" if split == "train" else split,
                            },
                        )
                    rows += 1
    if rows != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS:,} source rows, found {rows:,}")


def validate_archive(archive: Path) -> None:
    if archive.stat().st_size != ARCHIVE_BYTES or sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Vashantor archive checksum mismatch")


def paths(output: Path) -> tuple[Path, Path, Path]:
    return (
        output / "candidates/vashantor-v2-train.jsonl",
        output / "reserved/vashantor-v2-validation-test.jsonl",
        output / "metadata.json",
    )


def build(archive: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    validate_archive(archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vashantor-v2-", dir=output.parent) as temporary:
        root = Path(temporary)
        candidate, reserved, metadata_path = paths(root)
        candidate.parent.mkdir(parents=True)
        reserved.parent.mkdir(parents=True)
        counts, characters = Counter(), Counter()
        with (
            candidate.open("w", encoding="utf-8") as training,
            reserved.open("w", encoding="utf-8") as holdout,
        ):
            for split, record in records(archive):
                target = training if split == "train" else holdout
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[(split, record["bucket"])] += 1
                characters[(split, record["bucket"])] += len(record["text"])
        metadata = {
            "source": "Vashantor v2",
            "doi": DOI,
            "license": "CC BY 4.0",
            "archive": {
                "bytes": ARCHIVE_BYTES,
                "sha256": ARCHIVE_SHA256,
                "rows": EXPECTED_ROWS,
            },
            "policy": (
                "Official train forms are candidates; official validation/test forms are "
                "exclusion-only. Human-authored/translated research data, not natural traffic."
            ),
            "records": {
                f"{split}/{bucket}": value for (split, bucket), value in sorted(counts.items())
            },
            "normalized_characters": {
                f"{split}/{bucket}": value for (split, bucket), value in sorted(characters.items())
            },
            "files": {
                str(candidate.relative_to(root)): sha256(candidate),
                str(reserved.relative_to(root)): sha256(reserved),
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.move(root, output)
    return metadata


def verify(archive: Path, output: Path) -> dict:
    validate_archive(archive)
    candidate, reserved, metadata_path = paths(output)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for path in (candidate, reserved):
        relative = str(path.relative_to(output))
        if sha256(path) != metadata["files"][relative]:
            raise ValueError(f"output checksum mismatch: {path}")
    with candidate.open(encoding="utf-8") as training, reserved.open(encoding="utf-8") as holdout:
        for split, expected in records(archive):
            stream = training if split == "train" else holdout
            actual = json.loads(stream.readline())
            if actual != expected:
                raise ValueError(f"output differs from source in {split}")
        if training.readline() or holdout.readline():
            raise ValueError("output contains trailing records")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/bangladual/Vashantor_CSV_Format.zip"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw/expanded-v2-vashantor"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify(args.archive, args.output) if args.verify else build(args.archive, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

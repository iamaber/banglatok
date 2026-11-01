#!/usr/bin/env python3
"""Acquire scoped Pralekha Bengali documents for corpus v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

from acquire_corpus_v2 import acquire, paragraph, sha256, validate_artifact

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "data/corpus-v2-domain-sources.lock.json"
CARD = (
    "https://huggingface.co/datasets/ai4bharat/Pralekha/blob/"
    "4c16a270141891dd5c3e98f6f4f207f11bd11236/README.md"
)
ATTRIBUTION = "Pralekha; AI4Bharat; CC BY 4.0; pinned dataset card: " + CARD


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def serialize(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False) + "\n"


def candidate_records(paths: list[Path]):
    import pyarrow.parquet as pq

    for path in paths:
        parquet = pq.ParquetFile(path, pre_buffer=False)
        expected = {"n_id", "doc_id", "lang", "text"}
        if set(parquet.schema_arrow.names) != expected:
            raise ValueError(f"unexpected alignable schema: {path}")
        for batch in parquet.iter_batches(
            batch_size=256, columns=["n_id", "doc_id", "lang", "text"]
        ):
            for row in batch.to_pylist():
                if row["lang"] != "ben" or not row["n_id"] or not row["doc_id"]:
                    raise ValueError(f"invalid Bengali document identity: {path}")
                group = f"pralekha:{row['n_id']}"
                attribution = f"{ATTRIBUTION}; n_id={row['n_id']}; doc_id={row['doc_id']}"
                for original in row["text"].split("\n"):
                    text, reason = paragraph(original, "bn")
                    yield (
                        reason,
                        {
                            "text": text,
                            "bucket": "bangla",
                            "source": "pralekha-ben-alignable",
                            "group": group,
                            "attribution": attribution,
                            "split": "",
                        },
                    )


def reserved_records(paths: list[tuple[str, Path]]):
    import pyarrow.parquet as pq

    for split, path in paths:
        corpus_split = "validation" if split == "dev" else split
        parquet = pq.ParquetFile(path, pre_buffer=False)
        required = {"src_lang", "tgt_lang", "src_txt", "tgt_txt"}
        if set(parquet.schema_arrow.names) != required:
            raise ValueError(f"unexpected {split} schema: {path}")
        for batch in parquet.iter_batches(
            batch_size=128, columns=["src_lang", "tgt_lang", "tgt_txt"]
        ):
            for row in batch.to_pylist():
                if row["src_lang"] != "eng" or row["tgt_lang"] != "ben":
                    raise ValueError(f"unexpected language pair in {path}")
                full = normalize(row["tgt_txt"])
                group = f"pralekha-reserved:{split}:{hashlib.sha256(full.encode()).hexdigest()}"
                for original in row["tgt_txt"].split("\n"):
                    text = normalize(original)
                    if text:
                        yield {
                            "text": text,
                            "bucket": "bangla",
                            "source": f"pralekha-ben-{split}",
                            "group": group,
                            "attribution": ATTRIBUTION + "; official evaluation exclusion",
                            "split": corpus_split,
                        }


def convert(lock: dict, artifacts: dict[str, Path], output: Path) -> dict:
    candidate = output / "candidates/pralekha-ben-alignable.jsonl"
    reserved = output / "reserved/pralekha-ben-dev-test.jsonl"
    candidate.parent.mkdir(parents=True)
    reserved.parent.mkdir(parents=True)
    counts, characters, rejected = Counter(), Counter(), Counter()
    alignable = [
        artifacts[item["path"]]
        for item in lock["artifacts"]
        if item["path"].startswith("alignable/")
    ]
    holdouts = [
        (item["path"].split("/", 1)[0], artifacts[item["path"]])
        for item in lock["artifacts"]
        if item["path"].startswith(("dev/", "test/"))
    ]
    with candidate.open("w", encoding="utf-8") as stream:
        for reason, record in candidate_records(alignable):
            if reason:
                rejected[reason] += 1
                continue
            stream.write(serialize(record))
            counts["candidate_records"] += 1
            characters["candidate_normalized_characters"] += len(record["text"])
    with reserved.open("w", encoding="utf-8") as stream:
        for record in reserved_records(holdouts):
            stream.write(serialize(record))
            counts[f"reserved_{record['split']}_records"] += 1
            characters[f"reserved_{record['split']}_normalized_characters"] += len(record["text"])
    metadata = {
        "source": lock["source"],
        "repository": lock["repository"],
        "revision": lock["revision"],
        "license": lock["license"],
        "scope": lock["scope"],
        "lock_sha256": sha256(DEFAULT_LOCK),
        "counts": dict(sorted(counts.items())),
        "characters": dict(sorted(characters.items())),
        "rejected_candidates": dict(sorted(rejected.items())),
        "artifacts": {
            path: {"bytes": value.stat().st_size, "sha256": sha256(value)}
            for path, value in sorted(artifacts.items())
        },
        "files": {
            str(candidate.relative_to(output)): sha256(candidate),
            str(reserved.relative_to(output)): sha256(reserved),
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def load(lock_path: Path) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock_path.resolve() != DEFAULT_LOCK.resolve():
        raise ValueError("alternate lock files are not accepted for this frozen acquisition")
    if lock["scope"].find("unalignable Sangraha is prohibited") < 0:
        raise ValueError("source scope does not prohibit unalignable material")
    return lock


def artifact_paths(lock: dict, raw: Path, download: bool) -> dict[str, Path]:
    paths = {}
    for item in lock["artifacts"]:
        path = acquire(lock, item, raw) if download else raw / "artifacts" / item["path"]
        validate_artifact(path, item)
        paths[item["path"]] = path
    return paths


def build(lock: dict, raw: Path, output: Path) -> dict:
    destinations = [output / name for name in ("candidates", "reserved", "metadata.json")]
    if any(path.exists() for path in destinations):
        raise ValueError(f"converted output already exists: {output}")
    artifacts = artifact_paths(lock, raw, download=True)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pralekha-v2-", dir=output.parent) as temporary:
        root = Path(temporary)
        metadata = convert(lock, artifacts, root)
        shutil.move(root / "candidates", output / "candidates")
        shutil.move(root / "reserved", output / "reserved")
        shutil.move(root / "metadata.json", output / "metadata.json")
    return metadata


def verify(lock: dict, raw: Path, output: Path) -> dict:
    artifacts = artifact_paths(lock, raw, download=False)
    saved = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="pralekha-v2-verify-", dir=output.parent) as temporary:
        actual = convert(lock, artifacts, Path(temporary))
    if actual != saved:
        raise ValueError("Pralekha conversion differs from saved metadata")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/expanded-v2-pralekha"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/expanded-v2-pralekha"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    lock = load(args.lock)
    result = (
        verify(lock, args.raw, args.output) if args.verify else build(lock, args.raw, args.output)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

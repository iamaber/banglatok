"""Publish only the frozen tokenizer corpus (requires huggingface_hub to upload).

Run from any directory: python tools/publish_dataset.py [--dry-run OUTPUT] [--repo ID].
Dry runs use only the standard library and require an empty output directory.
"""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "iamaber/banglatok-training-corpus"
TRAIN_SHA256 = "bda48c8388fcd1cd706b4578619367a609ccf75a7f3aa659fede8c88aa198b39"
EXPECTED_ROWS = {"train": 563476, "validation": 14779, "test": 16712}
FIELDS = ("text", "bucket", "source", "group", "attribution", "split")
ATTRIBUTION_MARKERS = {
    "banglatlit": "https://github.com/farhanishmam/BanglaTLit",
    "tatoeba-ben": "tatoeba.org #",
    "tatoeba-fra": "tatoeba.org #",
    "wikipedia-bn-20231101": "https://bn.wikipedia.org/wiki/",
}
BUCKETS = {"bangla", "english", "banglish", "code-mixed"}
FILES = [
    "README.md",
    "manifest.json",
    "sources.lock.json",
    "corpus-stats.json",
    "LICENSES.md",
    "data/train.jsonl.gz",
    "data/validation.jsonl.gz",
    "data/test.jsonl.gz",
]
CAVEATS = "https://github.com/iamaber/banglatok/blob/main/data/CORPUS.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compress_split(split: str, output: Path) -> dict:
    source = ROOT / "data" / "processed" / f"{split}.jsonl"
    digest = hashlib.sha256()
    rows = size = 0
    by_source, by_bucket = Counter(), Counter()
    with source.open("rb") as original, output.open("wb") as compressed:
        with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as zipped:
            for line in original:
                rows += 1
                record = json.loads(line)
                if (
                    not isinstance(record, dict)
                    or set(record) != set(FIELDS)
                    or any(
                        not isinstance(record[key], str) or not record[key].strip()
                        for key in FIELDS
                    )
                ):
                    raise ValueError(f"{split}:{rows}: expected six nonempty string fields")
                if record["split"] != split or record["bucket"] not in BUCKETS:
                    raise ValueError(f"{split}:{rows}: invalid split or bucket")
                marker = ATTRIBUTION_MARKERS.get(record["source"])
                if marker is None or marker not in record["attribution"]:
                    raise ValueError(f"{split}:{rows}: missing source attribution")
                by_source[record["source"]] += 1
                by_bucket[record["bucket"]] += 1
                size += len(line)
                digest.update(line)
                zipped.write(line)  # Preserve the exact original JSONL bytes.
    if rows != EXPECTED_ROWS[split]:
        raise ValueError(f"{split}: expected {EXPECTED_ROWS[split]} rows, got {rows}")
    if split == "train" and digest.hexdigest() != TRAIN_SHA256:
        raise ValueError("training bytes differ from the frozen tokenizer training corpus")
    return {
        "path": f"data/{split}.jsonl.gz",
        "rows": rows,
        "uncompressed_bytes": size,
        "uncompressed_sha256": digest.hexdigest(),
        "compressed_bytes": output.stat().st_size,
        "compressed_sha256": sha256(output),
        "by_source": dict(sorted(by_source.items())),
        "by_bucket": dict(sorted(by_bucket.items())),
    }


def license_notices() -> str:
    lock = json.loads((ROOT / "data/sources.lock.json").read_text())
    required = {"banglatlit/LICENSE", "tatoeba/ben-eng.zip", "tatoeba/fra-eng.zip"}
    for entry in lock:
        if entry["path"] in required:
            path = ROOT / "data/raw" / entry["path"]
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"source notice checksum mismatch: {path}")
            required.remove(entry["path"])
    if required:
        raise ValueError(f"missing source notice locks: {sorted(required)}")
    text = f"""# Source-specific licenses and attribution

This is a collection, not a blanket relicensing of its contents. Preserve each
row's `source` and `attribution` fields and comply with the applicable source
terms. See the [dataset card](README.md#licenses-and-limitations) and
[corpus caveats]({CAVEATS}). No comprehensive rights or privacy audit has been
performed. Public upload was explicitly authorized with these limitations.

## Bengali Wikipedia

Original source: https://huggingface.co/datasets/wikimedia/wikipedia
(snapshot `20231101.bn`); original article URLs are retained per row. Article
histories at those URLs identify contributors. Applicable terms are
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and
[GFDL](https://www.gnu.org/licenses/fdl-1.3.html), as stated by the source card
and retained attribution notices. Corpus preparation normalizes and selects
paragraphs; preserve attribution and applicable share-alike obligations.

## Tatoeba via ManyThings

Original sources: https://tatoeba.org/ and https://www.manythings.org/anki/.
The processed attribution fields identify sentence IDs and contributors and
state CC BY 2.0 (France): https://creativecommons.org/licenses/by/2.0/fr/.
The original archives link the generic CC BY 2.0 terms below. Bengali/English
pairs and only the English side of French/English pairs are included.
The original `_about.txt` notices are retained verbatim below.
"""
    for language in ("ben", "fra"):
        with zipfile.ZipFile(ROOT / f"data/raw/tatoeba/{language}-eng.zip") as archive:
            names = [name for name in archive.namelist() if name.endswith("_about.txt")]
            if len(names) != 1:
                raise ValueError(f"expected one attribution notice for {language}")
            notice = archive.read(names[0]).decode("utf-8")
        text += f"\n### {language}-eng.zip / {names[0]}\n\n```text\n{notice}\n```\n"
    text += """
## BanglaTLit

Original source: https://github.com/farhanishmam/BanglaTLit/tree/f780fcdd5176ddfe6364ffe7948acb6a1c2ea58e

The exact upstream MIT notice follows. This repository-level notice does not
independently establish rights for every original third-party comment or
aggregated dataset. No blanket assertion that all source text is MIT is made.

```text
"""
    text += (ROOT / "data/raw/banglatlit/LICENSE").read_bytes().decode("utf-8")
    return text + "\n```\n"


def prepare(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    (output / "data").mkdir()
    splits = {
        name: compress_split(name, output / f"data/{name}.jsonl.gz") for name in EXPECTED_ROWS
    }
    shutil.copyfile(ROOT / "docs/DATASET.md", output / "README.md")
    shutil.copyfile(ROOT / "data/sources.lock.json", output / "sources.lock.json")
    shutil.copyfile(ROOT / "data/processed/stats.json", output / "corpus-stats.json")
    (output / "LICENSES.md").write_text(license_notices(), encoding="utf-8")
    # A Git revision plus hashes also identifies preprocessing in an uncommitted tree.
    preprocessing = sorted((ROOT / "corpus").glob("*.go"))
    preprocessing += sorted((ROOT / "tokenizer").glob("*.go"))
    preprocessing += [ROOT / "cmd/bntok/main.go", ROOT / "go.mod", ROOT / "go.sum"]
    preprocessing = [path for path in preprocessing if not path.name.endswith("_test.go")]
    manifest = {
        "manifest_version": 1,
        "schema": {field: "string" for field in FIELDS},
        "serialization": "Original UTF-8 JSONL bytes, gzip mtime=0 and empty filename",
        "splits": splits,
        "source_lock_sha256": sha256(output / "sources.lock.json"),
        "corpus_stats_sha256": sha256(output / "corpus-stats.json"),
        "preprocessing": {
            "repository": "https://github.com/iamaber/banglatok",
            "git_revision": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in preprocessing},
            "commands": ["go run ./cmd/bntok fetch", "go run ./cmd/bntok prepare"],
            "training_sha256_frozen": TRAIN_SHA256,
            "policy": (
                "NFC/whitespace normalization, quality/privacy heuristics, "
                "grouped canonical/SimHash deduplication, official heldout priority "
                "then deterministic 96/2/2 grouping"
            ),
        },
        "files": {
            name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
            for name in FILES
            if name != "manifest.json"
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="public destination dataset ID")
    parser.add_argument(
        "--dry-run", type=Path, metavar="PATH", help="prepare only, without uploading"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="banglatok-dataset-") as temporary:
        output = args.dry_run.resolve() if args.dry_run else Path(temporary)
        manifest = prepare(output)
        if args.dry_run:
            print(f"prepared {output}")
        else:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
            if api.repo_info(args.repo, repo_type="dataset").private:
                raise ValueError(
                    "destination exists but is private; refusing a nonpublic publication"
                )
            commit = api.upload_folder(
                repo_id=args.repo,
                repo_type="dataset",
                folder_path=str(output),
                allow_patterns=FILES,
                commit_message="Publish frozen banglatok tokenizer corpus with provenance",
            )
            print(f"uploaded https://huggingface.co/datasets/{args.repo}")
            print(f"commit {commit.oid}")
        print(json.dumps(manifest["splits"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

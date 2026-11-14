"""Publish a frozen v1 or explicitly selected expanded v2 tokenizer corpus.

Run: python tools/publish_dataset.py [--corpus-version v1|v2] [--dry-run OUTPUT].
Dry runs use only the standard library and require an empty output directory.
"""

import argparse
import gzip
import hashlib
import json
import shutil
import sqlite3
import subprocess
import tempfile
import zipfile
from collections import Counter
from contextlib import closing
from pathlib import Path

import prepare_corpus_v2 as corpus_v2

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
FROZEN_HASHES = dict(corpus_v2.FROZEN_HASHES)
V2_MIN_TRAIN_RECORDS = 2817380
V2_MIN_TRAIN_CHARACTERS = 329839300
V2_SPLITS = ("train", "validation", "test", "extra-validation", "extra-test")
V2_NOTICE_PATHS = (
    "README.md",
    "wikimedia/bnwikibooks-rightsinfo.json",
    "wikimedia/bnwikivoyage-rightsinfo.json",
)
V2_FILES = [
    "README.md",
    "v2/manifest.json",
    "v2/sources.lock.json",
    "v2/frozen-v1-sources.lock.json",
    "v2/domain-sources.lock.json",
    "v2/corpus-stats.json",
    "v2/audit.json",
    "v2/independent-audit.json",
    "v2/LICENSES.md",
    "v2/PROTOCOL.md",
    "v2/SOURCES.md",
    *(f"v2/notices/{name}" for name in V2_NOTICE_PATHS),
    *(f"v2/data/{split}.jsonl.gz" for split in V2_SPLITS),
]


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


def v2_source_notices(output: Path) -> dict:
    lock = json.loads((ROOT / "data/corpus-v2-sources.lock.json").read_text())
    domain_lock = json.loads((ROOT / "data/corpus-v2-domain-sources.lock.json").read_text())
    overlap = set(lock["source_declarations"]) & set(domain_lock["source_declarations"])
    if overlap:
        raise ValueError(f"duplicate v2 source declarations: {sorted(overlap)}")
    lock["source_declarations"].update(domain_lock["source_declarations"])
    entries = {entry["path"]: entry for entry in lock["artifacts"] if entry["source"] == "license"}
    for name in V2_NOTICE_PATHS:
        entry = entries.get(name)
        if entry is None:
            raise ValueError(f"missing v2 source notice lock: {name}")
        source = ROOT / "data/raw/expanded-v2/artifacts" / name
        if source.stat().st_size != entry["bytes"] or sha256(source) != entry["sha256"]:
            raise ValueError(f"source notice checksum mismatch: {name}")
        target = output / "v2/notices" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    notices = (
        license_notices().replace("(README.md#", "(../README.md#")
        + """

## Expanded Wikimedia sources

Bengali and English Wikipedia use the pinned 20231101 source card, reproduced
in [notices/README.md](notices/README.md): CC BY-SA 3.0 and GFDL where applicable,
not a blanket claim that every article is GFDL-available. Original article URLs
and contributor histories provide attribution; normalized/selected paragraphs
are modifications and retain applicable share-alike obligations.

Bengali Wikibooks and Wikivoyage use the 20261101 snapshots under CC BY-SA 4.0:
https://creativecommons.org/licenses/by-sa/4.0/. Their pinned rights API responses
are retained in notices/wikimedia/. Each row retains the original page URL,
revision URL, contributor-history URL, snapshot and modification notice.
Imported passages can have additional page-specific obligations.

## Pralekha Bengali government material

Scoped Bengali alignable documents use pinned Pralekha revision
`4c16a270141891dd5c3e98f6f4f207f11bd11236` under CC BY 4.0. Only
PIB/Mann Ki Baat alignable material is eligible; unalignable Sangraha is excluded.
Official Pralekha development and test Bengali text is exclusion-only. Rows retain
Pralekha `n_id`, `doc_id`, revision, license and transformation attribution. See
[domain-sources.lock.json](domain-sources.lock.json) for artifact hashes and the
first-party PM India and PIB reproduction policies.

See [SOURCES.md](SOURCES.md), [sources.lock.json](sources.lock.json), and
[frozen-v1-sources.lock.json](frozen-v1-sources.lock.json) for evidence and pins.
These notices do not establish complete copyright, privacy or consent clearance.
"""
    )
    (output / "v2/LICENSES.md").write_text(notices, encoding="utf-8")
    return lock


def compress_v2_split(
    split: str, source: Path, output: Path, db: sqlite3.Connection, declarations: dict
) -> dict:
    digest = hashlib.sha256()
    rows = size = characters = 0
    by_source, by_bucket = Counter(), Counter()
    declared = {item["record_source"]: item for item in declarations.values()}
    with source.open("rb") as original, output.open("wb") as compressed:
        with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as zipped:
            while line := original.readline(2 * 1024 * 1024):
                rows += 1
                if len(line) >= 2 * 1024 * 1024:
                    raise ValueError(f"{split}:{rows}: record exceeds 2 MiB")
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
                if (
                    record["split"] != split.removeprefix("extra-")
                    or record["bucket"] not in BUCKETS
                ):
                    raise ValueError(f"{split}:{rows}: invalid split or bucket")
                marker = ATTRIBUTION_MARKERS.get(record["source"])
                declaration = declared.get(record["source"])
                if marker is None and declaration is None:
                    raise ValueError(f"{split}:{rows}: unknown source attribution")
                if marker is not None and marker not in record["attribution"]:
                    raise ValueError(f"{split}:{rows}: missing source attribution")
                if declaration:
                    license_name = declaration["license"].split(";")[0]
                    if declaration.get("attribution_marker"):
                        mismatch = (
                            record["bucket"] != declaration["bucket"]
                            or not record["group"].startswith(declaration["group"].split("<")[0])
                            or declaration["attribution_marker"] not in record["attribution"]
                            or license_name not in record["attribution"]
                        )
                    else:
                        language = declaration["language"]
                        project = record["source"].split("-")[0]
                        url = f"https://{language}.{project}.org/wiki/"
                        mismatch = (
                            record["bucket"] != declaration["bucket"]
                            or not record["group"].startswith(declaration["group"].split("<")[0])
                            or url not in record["attribution"]
                            or f"{project.capitalize()} contributors" not in record["attribution"]
                            or license_name not in record["attribution"]
                        )
                    if mismatch:
                        raise ValueError(
                            f"{split}:{rows}: source language/group/attribution mismatch"
                        )
                    if (
                        not declaration.get("attribution_marker")
                        and project
                        in {
                            "wikibooks",
                            "wikivoyage",
                        }
                        and (
                            "oldid=" not in record["attribution"]
                            or "action=history" not in record["attribution"]
                        )
                    ):
                        raise ValueError(f"{split}:{rows}: missing revision/history attribution")
                if (
                    "banglishrev"
                    in (record["source"] + record["group"] + record["attribution"]).lower()
                ):
                    raise ValueError(f"{split}:{rows}: BanglishRev cannot be published")
                text = corpus_v2.normalize(record["text"])
                if split not in {"validation", "test"}:
                    baseline = (
                        split == "train"
                        and db.execute(
                            "SELECT 1 FROM original_train_keys WHERE key=?",
                            (corpus_v2.canonical(text),),
                        ).fetchone()
                        is not None
                    )
                    reason = corpus_v2.quality(text, record, baseline)
                    if text != record["text"] or reason:
                        raise ValueError(
                            f"{split}:{rows}: quality gate: {reason or 'not normalized'}"
                        )
                by_source[record["source"]] += 1
                by_bucket[record["bucket"]] += 1
                characters += len(text)
                size += len(line)
                digest.update(line)
                zipped.write(line)
    return {
        "path": f"v2/data/{split}.jsonl.gz",
        "rows": rows,
        "normalized_characters": characters,
        "uncompressed_bytes": size,
        "uncompressed_sha256": digest.hexdigest(),
        "compressed_bytes": output.stat().st_size,
        "compressed_sha256": sha256(output),
        "by_source": dict(sorted(by_source.items())),
        "by_bucket": dict(sorted(by_bucket.items())),
    }


def v2_dataset_card(splits: dict, stats: dict) -> str:
    configs = []
    for name, prefix, names in (
        ("default", "", ("train", "validation", "test")),
        ("v1", "", ("train", "validation", "test")),
        ("v2", "v2/", V2_SPLITS),
    ):
        configs.append(
            f"- config_name: {name}\n"
            + ("  default: true\n" if name == "default" else "")
            + "  data_files:\n"
        )
        for split in names:
            configs.append(
                f"  - split: {split.replace('-', '_')}\n    path: {prefix}data/{split}.jsonl.gz\n"
            )
    sizes = "\n".join(
        f"| `{split.replace('-', '_')}` | {entry['rows']:,} | {entry['normalized_characters']:,} | "
        f"{entry['uncompressed_bytes']:,} | {entry['compressed_bytes']:,} |"
        for split, entry in splits.items()
    )
    composition = "\n".join(
        f"| `{row['split'].replace('-', '_')}` | `{row['source']}` | "
        f"`{row['bucket']}` | {row['count']:,} |"
        for row in stats["source_bucket_records"]
    )
    frozen_card = (ROOT / "docs/DATASET.md").read_text(encoding="utf-8").split("---", 2)[2].strip()
    return (
        """---
language:
- bn
- en
license: other
license_name: source-specific-terms
license_link: https://huggingface.co/datasets/iamaber/banglatok-training-corpus/blob/main/v2/LICENSES.md
task_categories:
- text-generation
pretty_name: banglatok versioned tokenizer corpus
size_categories:
- 1M<n<10M
tags:
- bangla
- bengali
- banglish
- tokenizer
- provenance
configs:
"""
        + "".join(configs)
        + f"""---

# banglatok tokenizer corpus: frozen v1 and expanded v2

This repository preserves the original corpus at the root and adds an explicitly
versioned expansion under `v2/`. The default configuration is still frozen v1;
`v1` is an explicit name for the same root files. This is a **dataset release,
not a model promotion** or evidence that any tokenizer or downstream model improved.
Published v1 data, manifest and source notices are not replaced.

## Expanded v2

```python
from datasets import load_dataset

corpus = load_dataset("iamaber/banglatok-training-corpus", "v2")
train = load_dataset(
    "iamaber/banglatok-training-corpus", "v2", split="train", streaming=True,
    revision="<full dataset commit SHA>",
)
```

Pin a dataset commit for reproducibility. The six string fields remain `text`,
`bucket`, `source`, `group`, `attribution`, `split`. Extra files are exposed with
valid Hub split names `extra_validation` and `extra_test`; their row-level `split`
values remain `validation` and `test`, preserving emitted bytes. Frozen original
validation/test are byte-identical to v1, not newly sampled holdouts.

### Exact accepted sizes

| Split | Records | Normalized Unicode characters | JSONL bytes | Gzip bytes |
|---|---:|---:|---:|---:|
{sizes}

| Split | Source | Bucket | Records |
|---|---|---|---:|
{composition}

[v2/manifest.json](v2/manifest.json) records lossless compressed/uncompressed hashes
and exact per-source/bucket counts. [v2/corpus-stats.json](v2/corpus-stats.json)
records input distributions, rejection reasons and thresholds.
[v2/audit.json](v2/audit.json) retains the preparer's independent audit;
[v2/independent-audit.json](v2/independent-audit.json) records the publisher's
independent reread, rebuilt disk-backed indexes and quality/attribution validation.
Publication requires at least 2,817,380 canonically unique training rows and
329,839,300 normalized characters, with every leakage/growth gate passing.

### Sources and selection

The expansion retains eligible original BanglaTLit and Tatoeba material, reads all
Bengali Wikipedia articles from the pinned `20231101.bn` snapshot rather than the
old article sample, and adds English Wikipedia `20231101.en`, Bengali Wikibooks,
Bengali Wikivoyage (`20261101`) and licensed Bengali government/news-podcast
documents from pinned Pralekha. English uses a deterministic prefix of pinned
shards, not a uniform sample. Pralekha's unalignable Sangraha component is excluded,
and its official development/test Bengali text is leakage protection only. Real
paragraphs are selected, not synthetic transliterations or fragments chopped to
inflate growth. Bengali Wikibooks chapters share root-book groups.
[v2/SOURCES.md](v2/SOURCES.md) records acquisition settings, candidate supply
(not final accepted sizes), primary evidence and exclusions.
The exact [expanded source lock](v2/sources.lock.json),
[domain source lock](v2/domain-sources.lock.json) and
[original source lock](v2/frozen-v1-sources.lock.json) are retained separately.

NFC/Go-whitespace normalization preserves combining marks and joiners. Additional
filters reject control/replacement corruption, recognizable private information,
URL-heavy/boilerplate material, repetition and gross script/source mismatches.
New paragraphs require 40 characters and five words or 60 letters. Canonical
keys retain letters, numbers and marks; 64-bit word SimHash applies at 12 words,
radius three, with exhaustive band-pair candidate lookup. Connected source groups,
including links through rejected text, protect frozen holdouts and reserved
evaluation. New components use deterministic 96/2/2 assignment; original training
groups cannot become extra holdouts. Frozen holdouts are preserved, not refiltered.
Short texts receive canonical/group protection only. Approximate deduplication
does not guarantee semantic, translation or transliteration independence.

BanglishRev is exclusion-only local evaluation material: **no raw review data is
uploaded**. Research annotations, raw downloads and private-review files are not
in the upload allowlist. No newly acquired naturally authored Banglish/code-mixed
source met the rights/non-benchmark requirements; these registers remain limited.
Bucket labels are source/script proxies, not independently reviewed annotation.

### Licenses and limitations

This collection is not relicensed under a single license. Preserve row attribution.
Wikipedia retains the pinned historical CC BY-SA 3.0 / GFDL-where-applicable notice;
Wikibooks/Wikivoyage retain CC BY-SA 4.0, page revision and contributor-history
links. Pralekha's scoped government material retains CC BY 4.0, revision and
document IDs. Tatoeba retains CC BY 2.0 France sentence IDs and contributor credit.
The exact BanglaTLit MIT repository notice does **not** independently establish
rights for every original commenter or aggregated third-party passage.
[v2/LICENSES.md](v2/LICENSES.md) and [v2/notices/](v2/notices/) retain source evidence.
Article histories identify contributors; normalization/selection are modifications
and applicable share-alike and page-specific obligations remain.

Neither source availability nor these heuristic audits establishes comprehensive
copyright clearance, consent, anonymization, privacy or content safety. Residual
PII, offensive language, errors and domain/social biases can remain. Larger size
does not establish better linguistic quality, representativeness or downstream
accuracy. This is a tokenizer research corpus, not a validated embedding/LM corpus.

### Reproduce and verify v2

Use the source hashes and preprocessing revision/file hashes in the v2 manifest:

```sh
go run ./cmd/bntok fetch
go run ./cmd/bntok prepare
# Preserve verified frozen-v1 copies before expanded preparation.
.venv/corpus-v2/bin/python tools/acquire_corpus_v2.py --source all
.venv/corpus-v2/bin/python tools/acquire_corpus_v2.py --verify
.venv/corpus-v2/bin/python tools/acquire_pralekha_v2.py
.venv/corpus-v2/bin/python tools/acquire_pralekha_v2.py --verify
python tools/prepare_corpus_v2.py \
  --candidates 'data/raw/expanded-v2/candidates/*.jsonl' \
  --candidates 'data/raw/expanded-v2-extension/candidates/*.jsonl' \
  --candidates 'data/raw/expanded-v2-pralekha/candidates/*.jsonl' \
  --reserved-candidates 'data/raw/expanded-v2-pralekha/reserved/*.jsonl'
python tools/publish_dataset.py --corpus-version v2 --dry-run .cache/corpus-v2-publication
```

Acquisition dependencies and snapshot details are in [v2/SOURCES.md](v2/SOURCES.md).
Preparation requires the locally held exclusion-only BanglishRev input and pinned
Pralekha source holdouts; neither is redistributed as training data. Reproduction
therefore needs those verified local exclusions. [v2/PROTOCOL.md](v2/PROTOCOL.md)
freezes growth/quality gates and
future tokenizer-selection rules; publishing this dataset does not claim that
those model-selection rules were passed. The publisher independently reconstructs
audit facts against actual files and local exclusion input, verifies all frozen
hashes and source notices, checks quality/schema/attribution, and streams exact
JSONL bytes into deterministic gzip (`mtime=0`, empty filename). Use an empty
staging directory. Only the shared README and explicit `v2/` allowlist are eligible
for a v2 upload; no existing root data, manifests, locks or licenses are included.
Omitting `--corpus-version v2` retains the original v1 publisher behavior.

## Frozen v1 reference

The following original documentation describes only root/default/v1 files:

{frozen_card}
"""
    )


def prepare_v2(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    corpus = ROOT / "data/processed/expanded-v2"
    baseline = ROOT / "data/processed/frozen-v1"
    reserved = ROOT / corpus_v2.DEFAULT_RESERVED
    for split, expected in FROZEN_HASHES.items():
        if sha256(baseline / f"{split}.jsonl") != expected:
            raise ValueError(f"frozen {split} baseline hash mismatch")
        if split != "train" and sha256(corpus / f"{split}.jsonl") != expected:
            raise ValueError(f"frozen {split} output hash mismatch")
    audit_bytes = (corpus / "audit.json").read_bytes()
    stats_bytes = (corpus / "stats.json").read_bytes()
    audit = json.loads(audit_bytes)
    stats = json.loads(stats_bytes)
    gate_names = {"minimum_training_records", "minimum_training_characters", "no_leakage"}
    for name, report in (("audit", audit), ("stats", stats)):
        if set(report.get("gates", {})) != gate_names or any(
            report["gates"][key] is not True for key in gate_names
        ):
            raise ValueError(f"{name}: every audit/growth gate must pass")
        if report.get("frozen_sha256") != FROZEN_HASHES:
            raise ValueError(f"{name}: frozen hash mismatch")
    requirements = stats.get("growth_requirements", {})
    if (
        requirements.get("records", 0) < V2_MIN_TRAIN_RECORDS
        or requirements.get("normalized_characters", 0) < V2_MIN_TRAIN_CHARACTERS
    ):
        raise ValueError("preparation growth requirements are below the publication contract")
    (output / "v2/data").mkdir(parents=True)
    lock = v2_source_notices(output)
    with tempfile.TemporaryDirectory(prefix="banglatok-publish-audit-") as temporary:
        with closing(sqlite3.connect(Path(temporary) / "audit.sqlite3")) as db:
            db.execute("PRAGMA cache_size=-32768")
            actual = corpus_v2.audit(corpus, reserved, db, FROZEN_HASHES, baseline)
            splits = {
                split: compress_v2_split(
                    split,
                    corpus / f"{split}.jsonl",
                    output / f"v2/data/{split}.jsonl.gz",
                    db,
                    lock["source_declarations"],
                )
                for split in V2_SPLITS
            }
    if actual != {key: value for key, value in audit.items() if key != "gates"}:
        raise ValueError("independent audit/data mismatch")
    if any(actual["overlaps"].values()):
        raise ValueError("independent audit leakage gate failed")
    if (
        actual["accepted_unique_training_records"] < requirements["records"]
        or actual["accepted_training_normalized_characters"] < requirements["normalized_characters"]
    ):
        raise ValueError("independent growth gates failed")
    for split, entry in splits.items():
        count = sum(
            row["count"] for row in actual["source_bucket_records"] if row["split"] == split
        )
        characters = sum(
            row["count"] for row in actual["source_bucket_characters"] if row["split"] == split
        )
        if (
            entry["uncompressed_sha256"] != actual["split_sha256"][split]
            or entry["rows"] != count
            or entry["normalized_characters"] != characters
        ):
            raise ValueError(f"{split}: compression/audit data mismatch")
    for key, audit_key in (
        ("output_source_bucket_records", "source_bucket_records"),
        ("output_source_bucket_characters", "source_bucket_characters"),
    ):
        expected = [row for row in actual[audit_key] if row["split"] not in {"validation", "test"}]
        if stats.get(key) != expected:
            raise ValueError(f"stats/audit distribution mismatch: {key}")
    if stats.get("near_duplicate_method") != actual["near_duplicate_method"]:
        raise ValueError("stats/audit near-duplicate policy mismatch")
    for source, target in (
        ("data/corpus-v2-sources.lock.json", "sources.lock.json"),
        ("data/corpus-v2-domain-sources.lock.json", "domain-sources.lock.json"),
        ("data/sources.lock.json", "frozen-v1-sources.lock.json"),
        ("research/reports/CORPUS_V2_PROTOCOL.md", "PROTOCOL.md"),
        ("research/reports/CORPUS_V2_SOURCES.md", "SOURCES.md"),
    ):
        shutil.copyfile(ROOT / source, output / "v2" / target)
    (output / "v2/audit.json").write_bytes(audit_bytes)
    (output / "v2/corpus-stats.json").write_bytes(stats_bytes)
    actual["gates"] = audit["gates"]
    actual["publication_validation"] = {
        "schema_attribution_and_normalization": (
            "all rows checked; frozen heldouts retained unchanged"
        ),
        "quality": "all training/extra rows checked with preparer quality policy",
        "snapshot_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "snapshot_stats_sha256": hashlib.sha256(stats_bytes).hexdigest(),
    }
    (output / "v2/independent-audit.json").write_text(
        json.dumps(actual, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(v2_dataset_card(splits, actual), encoding="utf-8")
    preprocessing = [
        ROOT / "tools" / name
        for name in (
            "acquire_corpus_v2.py",
            "acquire_pralekha_v2.py",
            "prepare_corpus_v2.py",
            "publish_dataset.py",
        )
    ]
    manifest = {
        "manifest_version": 2,
        "corpus_version": "v2",
        "schema": {field: "string" for field in FIELDS},
        "serialization": "Original UTF-8 JSONL bytes, gzip mtime=0 and empty filename",
        "hf_config": "v2",
        "hf_split_names": {split: split.replace("-", "_") for split in V2_SPLITS},
        "splits": splits,
        "frozen_sha256": FROZEN_HASHES,
        "preprocessing": {
            "repository": "https://github.com/iamaber/banglatok",
            "git_revision": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in preprocessing},
            "candidate_source_sha256": stats.get("candidate_source_sha256", {}),
            "commands": [
                "python tools/acquire_corpus_v2.py --source all",
                "python tools/acquire_corpus_v2.py --verify",
                "python tools/acquire_pralekha_v2.py",
                "python tools/acquire_pralekha_v2.py --verify",
                "python tools/prepare_corpus_v2.py "
                "--candidates 'data/raw/expanded-v2/candidates/*.jsonl' "
                "--candidates 'data/raw/expanded-v2-extension/candidates/*.jsonl' "
                "--candidates 'data/raw/expanded-v2-pralekha/candidates/*.jsonl' "
                "--reserved-candidates "
                "'data/raw/expanded-v2-pralekha/reserved/*.jsonl'",
                "python tools/publish_dataset.py --corpus-version v2 --dry-run OUTPUT",
            ],
        },
        "files": {
            name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
            for name in V2_FILES
            if name != "v2/manifest.json"
        },
    }
    (output / "v2/manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="public destination dataset ID")
    parser.add_argument(
        "--corpus-version",
        choices=("v1", "v2"),
        default="v1",
        help="explicit corpus snapshot; default retains frozen v1",
    )
    parser.add_argument(
        "--dry-run", type=Path, metavar="PATH", help="prepare only, without uploading"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="banglatok-dataset-") as temporary:
        output = args.dry_run.resolve() if args.dry_run else Path(temporary)
        manifest = prepare_v2(output) if args.corpus_version == "v2" else prepare(output)
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
                allow_patterns=V2_FILES if args.corpus_version == "v2" else FILES,
                commit_message=(
                    "Add versioned banglatok v2 corpus without changing frozen v1"
                    if args.corpus_version == "v2"
                    else "Publish frozen banglatok tokenizer corpus with provenance"
                ),
            )
            print(f"uploaded https://huggingface.co/datasets/{args.repo}")
            print(f"commit {commit.oid}")
        print(json.dumps(manifest["splits"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

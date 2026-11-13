---
language:
- bn
- en
license: other
license_name: source-specific-terms
license_link: https://huggingface.co/datasets/iamaber/banglatok-training-corpus/blob/main/LICENSES.md
task_categories:
- text-generation
pretty_name: banglatok frozen tokenizer training corpus
size_categories:
- 100K<n<1M
tags:
- bangla
- bengali
- banglish
- code-mixed
- tokenizer
- provenance
configs:
- config_name: default
  default: true
  data_files:
  - split: train
    path: data/train.jsonl.gz
  - split: validation
    path: data/validation.jsonl.gz
  - split: test
    path: data/test.jsonl.gz
---

# banglatok tokenizer training corpus

The exact processed corpus used for both
[bn-bpe-58k](https://huggingface.co/iamaber/banglatok-bn-bpe-58k) and
[bn-bpe-28k](https://huggingface.co/iamaber/banglatok-bn-bpe-28k): naturally sourced
Bangla, English, Banglish (romanized Bengali), and a small code-mixed script-proxy
bucket. **Only `train` was used to learn tokenizer vocabularies and merges.**
Validation supported tokenizer screening; the frozen test split is held out and
was checked for split integrity, not used for tokenizer learning. These are
grouped in-domain splits, not an unseen-source benchmark or an embedding model
training corpus. No unseen BanglishRev benchmark text is included.

The compressed files preserve the original processed JSONL bytes exactly;
records have not been reserialized, filtered, or augmented for this upload.
The uncompressed training SHA-256 is
`bda48c8388fcd1cd706b4578619367a609ccf75a7f3aa659fede8c88aa198b39`.
[manifest.json](manifest.json) records both compressed and uncompressed byte
counts and hashes for every split, row counts, source/bucket counts, the
source-lock hash, and versioned preprocessing provenance.

## Load

Install the optional Hugging Face `datasets` library in your own environment:

```sh
uv pip install datasets
```

`pip install datasets` works too.

```python
from datasets import load_dataset

corpus = load_dataset("iamaber/banglatok-training-corpus")
print(corpus["train"][0])
```

Stream without downloading the entire corpus first:

```python
from datasets import load_dataset

train = load_dataset(
    "iamaber/banglatok-training-corpus", split="train", streaming=True
)
row = next(iter(train))
print(row["text"], row["source"], row["attribution"])
```

For reproducible consumption, pass `revision="<full dataset commit SHA>"` to
`load_dataset`. Retain attribution and group metadata when making derivatives;
using the text alone does not remove the source-specific obligations below.

## Composition

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| BanglaTLit | 257,678 | 8,538 | 10,416 |
| Tatoeba Bengali–English | 10,658 | 245 | 216 |
| Tatoeba French–English (English only) | 162,026 | 3,429 | 3,420 |
| Bengali Wikipedia `20231101.bn` | 133,114 | 2,567 | 2,660 |
| **Total** | **563,476** | **14,779** | **16,712** |

| Bucket | Train | Validation | Test |
|---|---:|---:|---:|
| `bangla` | 173,108 | 4,925 | 5,955 |
| `english` | 166,936 | 3,538 | 3,511 |
| `banglish` | 218,723 | 6,160 | 7,020 |
| `code-mixed` | 4,709 | 156 | 226 |

Training has 65,967,860 normalized Unicode characters. Bucket labels are
preparation categories, not independently annotated language or code-switching
ground truth. The code-mixed category is a script proxy and is underrepresented.
[corpus-stats.json](corpus-stats.json) preserves the preparation statistics:
691,979 input records, 9,011 quality rejections, 88,001 duplicates removed, and
1,072 near-duplicate links.

### Original sources

- [BanglaTLit](https://github.com/farhanishmam/BanglaTLit/tree/f780fcdd5176ddfe6364ffe7948acb6a1c2ea58e):
  pinned upstream commit; Romanized records and nonempty paired Bengali fields.
  The official validation/test CSVs receive held-out priority. The pretraining
  text file is not double-counted beside the training CSV.
- [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia):
  Bengali snapshot `20231101.bn`, both source shards; a deterministic roughly
  20% article sample, at most ten substantial paragraphs per article. Each row
  retains the original article URL and contributor/license attribution.
- [Tatoeba](https://tatoeba.org/) through
  [ManyThings](https://www.manythings.org/anki/): Bengali–English sentences and
  only English sentences from the French–English archive. Sentence IDs and
  contributor names remain in each row's attribution. French text is not used.

[sources.lock.json](sources.lock.json) records exact upstream URLs, byte counts,
and SHA-256 hashes. It is provenance metadata, not a list of extra uploaded
corpora: raw downloads, research annotations, and other processed datasets are
not distributed here. No synthetic transliteration is generated. BanglaDual is
excluded because overlap and natural/synthetic provenance need further audit.

## Row schema

Each JSONL row has exactly six string fields:

| Field | Meaning |
|---|---|
| `text` | Retained NFC- and whitespace-normalized text. |
| `bucket` | `bangla`, `english`, `banglish`, or `code-mixed`. |
| `source` | `banglatlit`, `tatoeba-ben`, `tatoeba-fra`, or `wikipedia-bn-20231101`. |
| `group` | Stable preparation group identifier used to keep linked records together. |
| `attribution` | Source URL/license notice, or sentence contributor names and IDs. |
| `split` | `train`, `validation`, or `test`, matching the file containing the row. |

For Tatoeba sentence IDs in attribution, the original sentence page is
`https://tatoeba.org/en/sentences/show/<ID>`. Wikipedia URLs point directly to
original articles; their page histories identify contributors. BanglaTLit
attribution points to the original upstream repository, not a new claim of
individual contributor clearance.

## Preprocessing and leakage controls

Preparation normalizes NFC and collapses/trims whitespace. It rejects text
outside 12–2,000 normalized runes, replacement-character corruption,
recognizable email addresses, Bangladeshi phone patterns, and simple credential
assignments. Punctuation, case, emoji, joiners, spelling variation, and meaningful
combining marks are preserved in retained text. **These are conservative
heuristics, not complete PII removal.**

Paired sentences and article paragraphs form groups before splitting. Canonical
duplicate keys ignore case, spacing, and punctuation only for deduplication and
grouping; they do not replace the published text. Texts with at least twelve
whitespace words additionally use 64-bit bag-of-words SimHash, linking Hamming
distances at most three. This approximate technique can miss variants and
over-group common wording; there is no claim of semantic or transliteration
leakage elimination.

Official test membership wins over validation membership; both win over
deterministic 96/2/2 group hashing. Connected records remain in a single split,
including links through rejected text. Official split sizes consequently change
through leakage exclusions and additional held-out groups. These controls reduce
in-domain leakage but do not establish generalization to new sources.

## Reproduce and verify

From a checkout of [banglatok](https://github.com/iamaber/banglatok), use the
preprocessing revision and source hashes in `manifest.json`:

```sh
go run ./cmd/bntok fetch
go run ./cmd/bntok prepare
python tools/publish_dataset.py --dry-run .cache/corpus-publication
```

`fetch` verifies checksum-locked downloads (and rejects changed upstream bytes);
`prepare` verifies the raw source manifest before reading data. Upstream URLs
may disappear or change; the lock describes this frozen snapshot rather than a
promise that moving downloads will always remain available. The publisher
requires the exact frozen training hash and all split row counts, validates
schema/split/attribution, and losslessly gzip-compresses only the three specified
JSONL splits with deterministic `mtime=0` and no embedded local filename. Use an
empty output directory. Upload requires `huggingface_hub` and an authenticated
account; `python tools/publish_dataset.py` publishes this prepared allowlist to
the public dataset repo. `--repo` overrides the destination.

The uploaded files are `README.md`, the three `data/*.jsonl.gz` files,
`manifest.json`, `sources.lock.json`, `corpus-stats.json`, and `LICENSES.md`.
Gzip decompression must reproduce each `uncompressed_sha256` in the manifest.
Source hashes alongside the Git revision identify preprocessing even when the
publication checkout includes uncommitted changes.

## Licenses and limitations

**No blanket CC BY-SA 4.0 or MIT license is asserted for this corpus.** The card
uses `license: other` because the collection retains source-specific terms:

- Wikipedia: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) /
  [GFDL](https://www.gnu.org/licenses/fdl-1.3.html), as stated by the original
  dataset card and row notices. Preserve attribution and applicable share-alike
  requirements for normalized/selected paragraphs and further adaptations.
- Tatoeba: [CC BY 2.0 France](https://creativecommons.org/licenses/by/2.0/fr/)
  as identified in the retained per-row notices, with contributor names and
  sentence IDs. Original archive notices link CC BY 2.0 and credit ManyThings
  and Tatoeba; those notices are preserved in [LICENSES.md](LICENSES.md).
- BanglaTLit: the exact upstream MIT notice is retained in
  [LICENSES.md](LICENSES.md). **MIT repository metadata does not independently
  establish rights for every original commenter or aggregated third-party
  source.** It is not a comprehensive corpus rights audit.

This corpus has **not undergone a comprehensive rights or privacy review**.
Public source availability is not a guarantee of permission for every intended reuse. Review
source terms, attribution, and your applicable legal obligations independently.
See the [corpus caveats](https://github.com/iamaber/banglatok/blob/main/data/CORPUS.md)
and original sources before redistribution or commercial use.

Public web text and comments may contain residual personal information,
profanity, offensive language, slurs, errors, and social/demographic bias.
Normalization and privacy heuristics do not imply consent, anonymization,
content safety, or comprehensive profanity filtering. Do not use this snapshot
as identity ground truth or assume it is safe for sensitive downstream uses.
It is designed to document tokenizer learning, not to establish sufficient or
representative training data for a final embedding or language model.

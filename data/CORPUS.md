# Training corpora

## Published v1 snapshot

Raw files are physically present under `data/raw/`; processed splits are under `data/processed/`. They are ignored by Git to avoid embedding large third-party corpora in source history. `sources.lock.json` records every download URL, exact byte count, and SHA-256; `bntok fetch` recreates matching files or rejects changed upstream content. `bntok prepare` verifies the manifest before reading data.

| Source | Included material | Attribution/license evidence |
|---|---|---|
| BanglaTLit, pinned Git commit | Romanized records plus nonempty Bengali paired fields; official held-out CSVs reserved through grouped splitting | Upstream repository includes MIT notice, retained at `raw/banglatlit/LICENSE`; source content aggregates third-party comments/datasets. MIT repository metadata is not an independent audit of every original contributor's rights. |
| Wikimedia Wikipedia, `20231101.bn` | Both raw shards downloaded; deterministic approximately 20% article sample, up to ten substantial paragraphs per article used | [Dataset card](https://huggingface.co/datasets/wikimedia/wikipedia): CC BY-SA 3.0 / GFDL. Original article URLs and contributor/license notice retained per record. Preserve applicable attribution/share-alike terms when distributing adapted corpus text. |
| Tatoeba via ManyThings | Bengali–English sentences and English side of French–English sentences | [Download page](https://www.manythings.org/anki/); each archive includes `_about.txt` identifying CC BY 2.0 and per-row attribution retained in every processed record. French text is not used. |

The processed training, validation and test splits are published at [iamaber/banglatok-training-corpus](https://huggingface.co/datasets/iamaber/banglatok-training-corpus), with per-row attribution and a hash manifest. Raw snapshots remain local. The text has not undergone a comprehensive rights or privacy audit. Do not treat the repository's software license as a blanket license for corpus redistribution; the dataset card retains source-specific terms and the limitations above.

Preparation applies NFC and whitespace normalization; excludes text outside 12–2,000 normalized runes, replacement-character corruption, recognizable email addresses, Bangladeshi phone-number patterns, and simple credential assignments. These are conservative heuristic filters, **not complete PII removal**. It preserves punctuation, case, emoji, joiners, spelling variation, and meaningful combining marks in retained text.

Before splitting, paired records and paragraphs of an article form groups. Canonical duplicate keys ignore case, spacing, and punctuation solely for grouping/deduplication. Long texts (at least twelve whitespace words) additionally use 64-bit bag-of-words SimHash with Hamming distance at most three. This approximate heuristic can both miss variants and over-group common wording; there is no semantic/transliteration deduplication claim. Official test membership wins over validation membership; both win over deterministic 96/2/2 group hashing. Consequently official split sizes change through leakage exclusions and additional held-out groups.

No synthetic transliteration is generated. BanglaDual is excluded because its upstream sources overlap and natural/synthetic provenance needs further audit. `BanglaTLit-PT.txt` is not double-counted beside its training CSV. Raw licenses and attribution remain in the checksum-locked source files.

| Training bucket | Records | Normalized characters |
|---|---:|---:|
| Bangla | 173,108 | 43,368,936 |
| English | 166,936 | 5,252,680 |
| Banglish | 218,723 | 16,946,172 |
| Code-mixed script proxy | 4,709 | 400,072 |
| Total | 563,476 | 65,967,860 |

Validation contains 14,779 records; the frozen test split contains 16,712. See `research/reports/corpus-stats.json` and `research/reports/corpus-audit.json` for counts, source composition, and hashes. Test text is checked only for split integrity; reported tokenizer screening uses validation. These are grouped in-domain splits, not fully unseen-source evaluations.

The small code-mixed bucket is oversampled by the proposed weighted mixture. Additional naturally authored code-mixed and target-domain records are the highest-priority corpus expansion. Corpus sufficiency for final embeddings must be established by learning curves and downstream evaluation, not by the presence of a 32k vocabulary.

## Corpus v2 expansion

The v2 acquisition provides 4,063,101 candidate passages: 863,101 Bengali and 3,200,000 English. It reads the full pinned Bengali Wikipedia snapshot and adds English Wikipedia, Bengali Wikibooks and Bengali Wikivoyage. These are **candidate counts, not accepted training counts**. The [source report](../research/reports/CORPUS_V2_SOURCES.md) records exact snapshots, licenses, extraction settings, checksums, exclusions and research-only dependencies; [`corpus-v2-sources.lock.json`](corpus-v2-sources.lock.json) pins the downloads.

No new naturally authored Banglish/code-mixed source passed the provenance and non-benchmark requirements. Those buckets are retained, not synthetically expanded. English Wikipedia supplies much of the volume; that is not evidence of improved Banglish, conversational coverage or downstream accuracy.

The [preregistered protocol](../research/reports/CORPUS_V2_PROTOCOL.md) requires at least 2,817,380 canonically unique training records and 329,839,300 normalized characters: five times both v1 measures after filtering, deduplication and leakage exclusions. New material passes stronger corruption, script, privacy-pattern, boilerplate and repetition checks. These remain heuristics, not a comprehensive rights, privacy or linguistic-quality audit.

Frozen v1 validation/test bytes and original article groups are preserved. Canonical duplicates and radius-three word-SimHash links protect entire connected source groups, including late-arriving or quality-rejected matches. New groups receive separate extra-validation/extra-test assignments. Near matching is exhaustive for that signature radius on texts with at least twelve words; it is not semantic or transliteration deduplication. BanglishRev is exclusion-only and is never published as training material.

From a clone with the original locked corpus prepared, preserve verified v1 copies once:

```sh
mkdir data/processed/frozen-v1 &&
  cp data/processed/{train,validation,test}.jsonl data/processed/frozen-v1/
```

Then use the optional acquisition environment described in the source report:

```sh
.venv/corpus-v2/bin/python tools/acquire_corpus_v2.py --source all
.venv/corpus-v2/bin/python tools/acquire_corpus_v2.py --verify
python tools/prepare_corpus_v2.py \
  --candidates 'data/raw/expanded-v2/candidates/*.jsonl' \
  --baseline-dir data/processed/frozen-v1 \
  --output data/processed/expanded-v2 \
  --work-db data/processed/corpus-v2-work.sqlite3
```

Preparation requires the separately acquired local BanglishRev exclusion input. Use fresh output/database paths; existing corpus files are not overwritten. It emits per-source/bucket counts and characters, rejection reasons, hashes, growth gates and an independently rebuilt split audit. Nonzero exit means a growth or integrity gate failed; it does not certify a fivefold expansion.

Train only on `expanded-v2/train.jsonl`. Select candidates on frozen validation and extra-validation with `benchmarks/select_corpus_v2.py` before opening either test split. Old model IDs remain immutable; larger data alone does not promote a new default.

# Corpus v2 tokenizer refresh: results

**Decision: keep `bn-bpe-58k` and `bn-bpe-28k` as the published models.** The expanded corpus passed its growth and leakage gates, but no tokenizer trained on it passed the promotion gates in the [protocol](CORPUS_V2_PROTOCOL.md). Every candidate used fewer tokens on new Wikipedia-heavy text and more tokens on the conversational, Banglish and review text that the current models serve.

## Corpus

The audited training split has **3,047,265 unique records and 719,996,744 normalized characters** (5.4× and 10.9× v1), with no overlap with frozen v1 holdouts, the new extra holdouts, BanglishRev or Pralekha dev/test. Composition and source details are in [CORPUS_V2_SOURCES.md](CORPUS_V2_SOURCES.md).

| Bucket | v1 training rows | v2 training rows | Main v2 additions |
|---|---:|---:|---|
| Bangla | 173,108 | 936,929 | full Bengali Wikipedia, Pralekha government text, Wikibooks, Wikivoyage |
| English | 166,936 | 1,890,061 | 1,723,311 English Wikipedia paragraphs |
| Banglish | 218,723 | 216,332 | none |
| Code-mixed | 4,709 | 3,943 | none |

v1 English was conversational Tatoeba text; v1 had no English Wikipedia. No natural Banglish or code-mixed source passed the redistribution-rights review ([CORPUS_V2_BANGLISH_SOURCES.md](CORPUS_V2_BANGLISH_SOURCES.md)), and synthetic transliteration was not used.

## Full-corpus candidates

Four native candidates were trained on the full v2 split: base 16k and 32k, plain and bucket-balanced, then fused toward the 27,760 and 57,925 budgets. Fusion only adds word-initial pieces seen at least twice, and this base had too few to fill the budgets: the candidates have 25,147–25,150 and 55,150–56,081 IDs. Each was compared with the existing model of the next size up, so a candidate never wins on extra vocabulary.

Ratios to the same-tier published model on frozen v1 validation (fertility) and extra-validation (total tokens); above 1 is worse:

| Candidate | IDs | Bangla | English | Banglish | Code-mixed | Extra-validation |
|---|---:|---:|---:|---:|---:|---:|
| 58k plain | 56,081 | 1.033 | 1.123 | 1.039 | 1.034 | **0.917** |
| 58k balanced | 55,150 | 1.081 | 1.129 | 1.034 | 1.024 | 0.936 |
| 28k plain | 25,147 | 1.110 | 1.242 | 1.103 | 1.094 | 0.989 |
| 28k balanced | 25,150 | 1.161 | 1.231 | 1.081 | 1.051 | 1.002 |

All four fail the frozen-validation gates (aggregate no-regression; at most 1% per bucket, 2% code-mixed). Throughput was 0.93–1.27× the baselines, within the 0.8× gate.

## User-text-weighted mixtures

A follow-up, recorded in the protocol before training, removed most English Wikipedia. **m1** keeps every v2 row except English Wikipedia plus 167,000 English Wikipedia rows (1,490,954 rows). **m2** also cuts Bengali Wikipedia to 266,228 rows (1,057,230 rows). These are deterministic, group-preserving subsets of the audited split made with `tools/subset_corpus_v2.py`.

| Candidate | IDs | Bangla | English | Banglish | Code-mixed | Extra-validation |
|---|---:|---:|---:|---:|---:|---:|
| 58k plain m1 | 57,252 | 1.015 | 1.032 | 1.023 | 1.027 | 0.924 |
| 58k plain m2 | 57,442 | 1.017 | 1.027 | 1.018 | 1.020 | **0.920** |
| 58k balanced m2 | 56,160 | 1.064 | 1.032 | 1.016 | 1.008 | 0.938 |
| 28k plain m2 | 26,978 | 1.031 | 1.057 | 1.030 | 1.019 | 0.942 |

Removing English Wikipedia cuts the regression from 3–24% to 1.5–9%, but none of the eight mixture candidates passes the 1% bucket limits. The balanced variants were worse on Bangla in every comparison.

## Test results after selection

Tokens per word on held-out test data, scored only after selection had rejected every candidate. Lower is better.

| Model | Frozen v1 test | BanglishRev reviews 0–10k | BanglishRev reviews 10k–20k | Extra-test |
|---|---:|---:|---:|---:|
| **bn-bpe-58k (published)** | **1.361** | **1.379** | **1.375** | 1.657 |
| v2 58k plain | 1.416 | 1.457 | 1.454 | 1.521 |
| v2 58k plain m2 | 1.386 | 1.413 | — | 1.526 |
| SentencePiece Unigram 57,925, v2 data | 1.480 | 1.463 | 1.459 | 1.404 |
| SentencePiece BPE 57,925, v2 data | 1.491 | 1.467 | 1.462 | 1.415 |
| **bn-bpe-28k (published)** | **1.543** | **1.587** | **1.582** | 1.906 |
| v2 28k plain | 1.723 | 1.750 | 1.744 | 1.890 |
| v2 28k plain m2 | 1.593 | 1.614 | — | 1.798 |
| SentencePiece Unigram 27,760, v2 data | 1.697 | 1.677 | 1.674 | 1.577 |

The extra-test split is drawn from the new Wikipedia-heavy material, so it favours models trained on it. On real e-commerce reviews from an unseen source (BanglishRev), every v2 model needs more tokens than the published one. Same-size SentencePiece trained on the v2 data shows the same pattern, so the loss comes from the data mix rather than the training algorithm.

## What this means

- A larger licensed corpus did not improve the tokenizer for this project's users. Compression on informal and Banglish text depends on having that kind of text in training, and the available rights-cleared sources are encyclopedic or governmental.
- Lower token counts on extra-test reflect a vocabulary shift toward formal prose, not a better general tokenizer.
- The matched downstream encoder gate was not run because no candidate passed the intrinsic gates first.
- A future attempt needs rights-cleared natural Banglish/code-mixed text (for example data the project owns or is licensed to redistribute). More Wikipedia will not close this gap.

## Evidence

- Validation and selection: [corpus-v2-validation.json](../../benchmarks/results/corpus-v2-validation.json), [corpus-v2-selection.json](../../benchmarks/results/corpus-v2-selection.json), [mixtures validation](../../benchmarks/results/corpus-v2-mixtures-validation.json), [mixtures selection](../../benchmarks/results/corpus-v2-mixtures-selection.json).
- Tests after selection: [corpus-v2-final-test.json](../../benchmarks/results/corpus-v2-final-test.json), [corpus-v2-disjoint-reviews.json](../../benchmarks/results/corpus-v2-disjoint-reviews.json), [mixtures final test](../../benchmarks/results/corpus-v2-mixtures-final-test.json).
- Timings: Apple M5, single thread, five shuffled repeats, batches of 64.

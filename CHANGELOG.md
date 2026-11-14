# Changelog

## Unreleased

### Research

- Added reproducible corpus-v2 acquisition from pinned Bengali/English Wikipedia, Bengali Wikibooks and Wikivoyage, with source-specific attribution and rights limitations.
- Added stronger quality filtering, disk-backed canonical/near deduplication, immutable v1 evaluation files, separate extra holdouts and independent growth/leakage audits. Fivefold growth counts only accepted unique training records and normalized characters.
- Added validation-only promotion gates and versioned Hugging Face corpus/model publishing. Existing model IDs and v1 dataset artifacts remain immutable; changing the default requires passing the promotion gates. Default Hub publishing targets v1 only; versioned models require explicit paths and their own provenance.
- Streamed fusion training and made SentencePiece training sequential by default for larger corpora. Benchmarks can compare local candidates, extra holdouts and disjoint review samples.
- Run research tooling tests separately from inference wheel tests across the source-CI platform/Python matrix, keeping research scripts and provenance inputs out of distributions.
- Built and audited corpus v2: 3,047,265 unique training records and 720M normalized characters (5.4× and 10.9× v1) with no holdout leakage, adding scoped CC BY 4.0 Pralekha Bengali government text. Sources without upstream redistribution rights, including Vashantor, were excluded.
- Tokenizers trained on corpus v2, and on two user-text-weighted subsets of it, failed the promotion gates: they compressed Wikipedia-style text better but needed more tokens on frozen v1 test text and BanglishRev reviews (+1.8% to +5.7% at 58k, up to +11.7% at 28k). `bn-bpe-58k` and `bn-bpe-28k` remain the defaults. See `research/reports/CORPUS_V2_RESULTS.md`.
- Added a restricted training-only corpus workflow for directly authorized
  natural text that may produce public model artifacts but may not be
  redistributed. Rights evidence, commercial/model grants, platform terms,
  privacy review and retention/deletion attestations are mandatory; marked
  private corpora are rejected by the public dataset publisher.

## 0.2.2 — 2026-11-13

- Added the full usage guide to both Hugging Face model repositories, with links from their model cards.
- Simplified the introductory examples in the README and guide.
- Published the attributed training, validation and test corpus on Hugging Face.
- Added the training-data sources and measured token-count/speed comparisons with popular tokenizers to both Hugging Face model cards.

## 0.2.1 — 2026-11-13

### Documentation

- Added `docs/USAGE.md`: installation, model selection, normalization, Python API, CLI, Hugging Face, multiprocessing, troubleshooting, and guides for token budgets, Datasets, PyTorch batches and training a language model.
- Recommended uv for installation while retaining pip instructions. Linked guides from the README and PyPI documentation metadata.
- Standardized the package name (`banglatok`) and benchmark model labels (`banglatok bn-bpe-58k` / `banglatok bn-bpe-28k`). README links work on both GitHub and PyPI.

### Fixed

- Calls in a child forked after the Go library was loaded now raise an actionable error instead of hanging. Use `spawn` or `forkserver` and create the tokenizer in each worker.

### Internal

- Removed unused native JSON encode operations; shared corpus manifest/hash helpers; moved training and report output helpers into the Go CLI. Model files and token IDs are unchanged.
- Updated the corpus CLI's parquet-go dependency to 0.32.0; source builds now require Go 1.24.9 or newer. The native wheel's Unicode dependency remains golang.org/x/text 0.28.0.

## 0.2.0 — 2026-11-13

First public release.

### Added

- Two pretrained models bundled in the wheel and loaded offline with `Tokenizer.from_pretrained()`. `bn-bpe-58k` (default) and `bn-bpe-28k` are space-fused BPE vocabularies, documented in `docs/MODELS.md` and licensed CC BY-SA 4.0.
- `tokenize`, `count_tokens`, `decode_batch`, `id_to_token`, `token_to_id`, `get_vocab`, the `algorithm` property, `available_models()` and the control-ID constants.
- `tokenizer(texts, add_special_tokens=..., max_length=..., padding=...)` returns encoder-ready `input_ids` and `attention_mask`. `decode(..., skip_special_tokens=True)` drops the control IDs again.
- A `banglatok` command and `python -m banglatok`.
- `benchmarks/compare_popular.py` and `benchmarks/POPULAR_COMPARISON.md` compare the models with SentencePiece/BNLP, tiktoken, BanglaBERT, MuRIL, mBERT, XLM-R and Qwen3 on unseen and in-domain text.
- `benchmarks/build_fused_model.py` reproducibly builds fused models from a base BPE model.
- `Tokenizer.save_huggingface(directory)` writes `tokenizer.json` and `tokenizer_config.json` for Hugging Face `tokenizers` and `transformers`, with the same IDs as the native tokenizer (checked on 24,799 validation, review and stress texts for both models). `tools/publish_huggingface.py` uploads both models to the Hub.
- Linux aarch64 and Windows x86_64 wheels in the release workflow.
- `benchmarks/train_sentencepiece_baselines.py` trains SentencePiece Unigram and BPE baselines with the same vocabulary sizes as the bundled models; the comparison now includes them, and a Linux x86_64 run.

### Changed

- The project is now `banglatok` (was `bn-tokenizers-embedding`, never published to PyPI). The import is `banglatok` and the command is `banglatok`.
- Encoding is about 2× faster with identical IDs. Normalization skips work for text that is already NFC. BPE merges recompute pair ranks only around merged tokens. Single texts are encoded straight into a reusable per-thread buffer, and batch results come back in one aligned buffer. Every model and about 94,000 validation, test and stress texts produce the same IDs as before.
- Go: `Tokenizer.AppendEncode`, `Tokenizer.Vocabulary`, `Tokenizer.Algorithm` and `Tokenizer.Merges`.
- The `banglatok` command reads and writes UTF-8 regardless of the console code page.
- Research reports, experiments and scripts moved under `research/`.
- License expression: `MIT AND BSD-3-Clause AND CC-BY-SA-4.0`.

## 0.1.0a1 — unreleased preview

- Prepared the typed Python/Go tokenizer API for a public alpha, with an offline toy vocabulary and a self-contained training example.
- Added MIT licensing, retained BSD-3-Clause native notices, installed-distribution CI, archive validation and opt-in platform-wheel/PyPI workflows.

The earlier `0.1.0` artifacts were local development builds and were never published to PyPI.

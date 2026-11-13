# Changelog

## 0.2.0 — 2026-11-10

First public release candidate.

### Added

- Two pretrained models bundled in the wheel and loaded offline with `Tokenizer.from_pretrained()`. `bn-bpe-58k` (default) and `bn-bpe-28k` are space-fused BPE vocabularies, documented in `docs/MODELS.md` and licensed CC BY-SA 4.0.
- `tokenize`, `count_tokens`, `decode_batch`, `id_to_token`, `token_to_id`, `get_vocab`, the `algorithm` property, `available_models()` and the control-ID constants.
- `tokenizer(texts, add_special_tokens=..., max_length=..., padding=...)` returns encoder-ready `input_ids` and `attention_mask`. `decode(..., skip_special_tokens=True)` drops the control IDs again.
- A `banglatok` command and `python -m banglatok`.
- `benchmarks/compare_popular.py` and `benchmarks/POPULAR_COMPARISON.md` compare the models with SentencePiece/BNLP, tiktoken, BanglaBERT, MuRIL, mBERT, XLM-R and Qwen3 on unseen and in-domain text.
- `benchmarks/build_fused_model.py` reproducibly builds fused models from a base BPE model.
- Linux aarch64 wheels in the release workflow.

### Changed

- The project is now `banglatok` (was `bn-tokenizers-embedding`, never published to PyPI). The import is `banglatok` and the command is `banglatok`.
- Encoding is about 2× faster with identical IDs. Normalization skips work for text that is already NFC. BPE merges recompute pair ranks only around merged tokens. Single texts are encoded straight into a reusable per-thread buffer, and batch results come back in one aligned buffer. Every model and about 94,000 validation, test and stress texts produce the same IDs as before.
- Go: `Tokenizer.AppendEncode`, `Tokenizer.Vocabulary` and `Tokenizer.Algorithm`.
- License expression: `MIT AND BSD-3-Clause AND CC-BY-SA-4.0`.

## 0.1.0a1 — unreleased preview

- Prepared the typed Python/Go tokenizer API for a public alpha, with an offline toy vocabulary and a self-contained training example.
- Added MIT licensing, retained BSD-3-Clause native notices, installed-distribution CI, archive validation and opt-in platform-wheel/PyPI workflows.

The earlier `0.1.0` artifacts were local development builds and were never published to PyPI.

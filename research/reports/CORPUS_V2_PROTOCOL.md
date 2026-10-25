# Corpus v2 and tokenizer refresh protocol

This protocol was recorded before inspecting new candidate scores. The primary task refreshes tokenizers; matched encoder evaluation is now the final promotion gate because token count alone does not establish downstream quality.

## Immutable baseline

- Package/model baseline: banglatok 0.2.2.
- Training: 563,476 records, 65,967,860 normalized Unicode characters.
- Training SHA-256: `bda48c8388fcd1cd706b4578619367a609ccf75a7f3aa659fede8c88aa198b39`.
- Validation: 14,779 records; SHA-256 `047e0acb94d621508467fcd0e7b3369204f7e040d4b9144b58614faac9f1cc2d`.
- Test: 16,712 records; SHA-256 `3f39a8faf7ecc280ec32ce231d379cec788eb79a52d9a6e66917073500879834`.
- Existing model hashes: `bn-bpe-58k` = `f045ed2c69e8c10d718272ae40291d51ca7453cc6abd26ef58eedddcaaaa13b9`; `bn-bpe-28k` = `b5b97c3a9472798e22da0d32182c5bc4e9a4b89a74f0a6af7bc301c1a0985cbf`.
- Local copies: `data/processed/frozen-v1/` and `models/baselines/banglatok-0.2.2/`.

## Growth and data-quality gates

The final training split must contain at least **2,817,380 canonically distinct records** and **329,839,300 normalized Unicode characters**: five times both baseline measures. Repeating rows, synthetic transliteration and chopping short fragments merely to inflate counts do not count as growth.

Retain attribution, source licenses, original article/sentence groups and explicit source language. Acquire real text from documented redistributable sources. Exclude NC/ND material, unlicensed general crawls and every BanglishRev source used for evaluation. Existing uncertain BanglaTLit provenance remains disclosed rather than relabeled as newly cleared.

Use bounded-memory preparation, stronger corruption/PII/repetition/boilerplate filters, canonical and approximate deduplication, and split/group exclusion. Report rejection reasons, source/bucket counts and character totals before and after filtering. Larger size is not itself evidence of improved linguistic quality. Language buckets are source labels or disclosed script proxies, not independently reviewed annotations.

Preserve original validation/test bytes. New source groups receive deterministic held-out assignments into separate extra-validation/extra-test files. Exclude frozen heldout groups and exact/near text matches from training, independent of input ordering. Report remaining limitations of approximate deduplication.

### Source priority

Acquisition priority is natural Banglish/code-mixed text, then Bengali text from target domains, then additional general Bengali, and only then English volume. A source is eligible only when the exact artifact is revision-pinnable and its first-party terms permit transforming and publicly redistributing the retained text with the required attribution. Repository code licenses do not automatically license collected user content. Reject research-only, noncommercial, no-derivatives, ambiguous upstream-rights and synthetic-transliteration sources.

Small, high-value Banglish/code-mixed sources need not move the fivefold size totals materially. Measure and report their accepted rows separately, retain source groups, and give each language bucket equal sentence mass according to the declared training mixtures rather than duplicating minority records.

## Candidate training

Retain the original `bn-bpe-58k` and `bn-bpe-28k` artifacts and IDs. New models get new explicit identifiers.

Compare two native BPE training mixtures already supported by the CLI: 40/25/25/10 Bangla/English/Banglish/code-mixed and 25/25/25/25. Each sentence receives equal mass within its bucket; minority rows are not physically duplicated. Train base vocabularies of 16,000 and 32,000 IDs and fit space fusion/caches on expanded training text only. Aim at the existing total budgets of 27,760 and 57,925 IDs; record actual sizes if a fusion budget cannot be filled. Do not conflate extra vocabulary with better learning.

## Selection before final testing

Use frozen original validation and new extra-validation only for candidate selection. Record per-bucket tokens per whitespace word, byte-fallback rate, sequence lengths/truncation, unknown-token count, round-trip failures and encode throughput.

Promotion to default requires:

1. Zero unknown outputs and 100% normalized-text round trips, including stress cases.
2. No increase in overall original-validation token count; no more than 1% regression in Bangla, English or Banglish fertility and 2% in the small code-mixed bucket.
3. At least 1% fewer aggregate tokens on extra-validation than the same-size existing model.
4. Median native throughput no worse than 20% below the existing same-size model on the same inputs and machine.

`benchmarks/select_corpus_v2.py` applies these gates and records every rejected candidate, checking both single/batch throughput on both validation corpora. Native byte-fallback metrics count unmerged byte IDs 4–259, including ordinary ASCII punctuation and spaces; they are not unknown-token counts or a directly comparable metric for other tokenizer families.

If no candidate passes, retain the published default and report the negative result; do not describe a larger corpus or a retrained candidate as a demonstrated quality improvement.

## Matched encoder promotion gate

An intrinsic winner is eligible for an experimental v2 model ID, but it does not replace the package default until a matched encoder comparison also passes. Reuse the repository's pinned `intfloat/multilingual-e5-small` body, reviewed alignment data, hard negatives, objectives, seeds 17/29/43, equal-data and equal-compute budgets, and source-backed retrieval evaluator. Compare the candidate with the original E5 tokenizer under identical training exposure; do not compare only against unchanged E5 or an old custom-tokenizer checkpoint.

The primary downstream metric remains Banglish→Bengali nDCG@10. Promotion requires at least +0.03 absolute development gain over the matched adapted-original control, a paired 95% interval above zero, no more than 0.01 English-query mean nDCG loss, and no more than 1.25× latency or peak memory in every seed under both budgets. Freeze candidate identity, tokenizer hash, initialization method, corpus/alignment hashes and thresholds before scoring it.

The existing source-backed benchmark is agent-authored and has been used for earlier model decisions. A new candidate has not been scored on it, but its reserved split is not a never-observed benchmark. Report that limitation and treat the result as project evidence, not independent natural-traffic validation. If the encoder gate fails, retain the current package default even when the v2 tokenizer has better intrinsic compression; the experimental model may still be published with that negative downstream result.

## Final benchmarks

After selection, evaluate original test, extra-test, the original 10,000 BanglishRev reviews, and a disjoint additional review sample. No test set selects model settings. Keep all BanglishRev text out of acquisition/training.

Compare existing models, selected candidates, same-data/same-size SentencePiece Unigram and BPE, and the existing popular tokenizer baselines. Keep input normalization, versions, vocabulary accounting and special-token conventions explicit. Timings use five shuffled repeats, single calls and batches of 64, single thread. Existing reports are historical; write new result files. Report vocabulary, compression, round-trip rate, fallback, latency/throughput, source limitations and measured tradeoffs. Tokenizer metrics do not establish downstream accuracy.

## Publication

Publish expanded corpus and new model IDs separately from immutable v1 artifacts. Retain checksums, source locks, original notices, preparation/training parameters, selection results and independent heldout results. Update package/docs/Hub cards only with observed results. A default vocabulary change requires a new minor package release and an explicit compatibility note.

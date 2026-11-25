# Bundled models

banglatok bundles three models which load offline with
`Tokenizer.from_pretrained(name)`. The package, Python import and command are
all named `banglatok`; the `bn-bpe-*` names select its models. See the
[user guide](USAGE.md#choosing-a-model) for examples.

| Name | Vocabulary | Base BPE | Fused IDs | Use it when |
|---|---:|---:|---:|---|
| `bn-bpe-58k` (default) | 57,925 | 32,000 | 25,925 | You want the shortest sequences |
| `bn-bpe-28k-v2` | 27,760 | 16,000 | 11,760 | You want the best current 28k model |
| `bn-bpe-28k` | 27,760 | 16,000 | 11,760 | You need the original 28k token IDs |

The vocabulary counts include 4 control IDs and 256 byte-fallback IDs.
`bn-bpe-28k-v2` deliberately has different IDs from `bn-bpe-28k`; model
embeddings trained with one are not compatible with the other.

The original `bn-bpe-58k` and `bn-bpe-28k` use the published
[`iamaber/banglatok-training-corpus`](https://huggingface.co/datasets/iamaber/banglatok-training-corpus).
Only its 563,476-record training split was used for model learning.

`bn-bpe-28k-v2` uses that frozen training split plus licensed Global Voices
Bengali and Storybooks Bangladesh text. Leakage-aware canonical, group and
SimHash deduplication produced 613,096 training records and 79,133,165
normalized characters, including 61,351 Global Voices records and 40
Storybooks records. No YouTube text, viewer comments, BanglishRev records or
synthetic transliterations were used. Dataset text keeps its source-specific
terms, separate from the tokenizer files' CC BY-SA 4.0 license.

## How they were built

1. `tools/prepare_corpus_v2.py` deduplicated the added sources against the
   frozen training, validation, test and BanglishRev exclusion sets.
2. `tools/train_corpus_v2.py --skip-sentencepiece` learned each base BPE model
   and called `benchmarks/build_fused_model.py`. Fusion adds IDs for a space
   followed by a frequent word-initial base ID; the 64,000 most frequent
   non-piece training words become load-time cache entries without changing
   token IDs.
3. `go run ./tools/compile_model src/banglatok/pretrained/*.json` wrote each
   `.bntok` sidecar. It records the SHA-256 of the exact JSON and compiled
   payload; `go test ./tokenizer` rejects stale sidecars.

```sh
python tools/train_corpus_v2.py --training <prepared-public-train.jsonl> \
  --output <output> --skip-sentencepiece
go run ./tools/compile_model src/banglatok/pretrained/*.json
```

| Artifact | SHA-256 |
|---|---|
| `bn-bpe-58k.json` | `f045ed2c69e8c10d718272ae40291d51ca7453cc6abd26ef58eedddcaaaa13b9` |
| `bn-bpe-28k.json` | `b5b97c3a9472798e22da0d32182c5bc4e9a4b89a74f0a6af7bc301c1a0985cbf` |
| `bn-bpe-28k-v2.json` | `6fcb410a3799d0f59d81db8776cc0312e9322b8d509c8fb45377dcff418e6b0d` |
| `bn-bpe-28k-v2.bntok` | `7d5c2c05709e7673d7ac21d79aac87bf823a28e6090daec193654bda5ed6a4ac` |
| v2 base `bpe-16000-v2.json` | `8f890eb3bea552ad1454b336a4c290d91ae92ed04e6ecc470d27c80671da7d23` |
| v2 training split | `0695ca6c02c04d2ec074d9498ea807dfeee2201adfbd5c76b6e032dec3fa929b` |
| Original base `bpe-32000.json` | `b3fcc85abb34e27648fc297dfc8052cf871f7c86482a4d4e91626a72513a0bc3` |
| Original base `bpe-16000.json` | `77ac7594a514f76eb78b22c95b173c96fb83b08f0e7538af60261bebfa28306` |
| Frozen v1 training split | `bda48c8388fcd1cd706b4578619367a609ccf75a7f3aa659fede8c88aa198b39` |

The base `bpe-*.json` files and training splits are not distributed in the
Python package. Their hashes identify the exact local training inputs.

## Behavior

- Lossless: `decode(encode(text)) == normalize(text)` for any string. Normalization is NFC plus whitespace collapse and trim.
- No unknown tokens. Characters not in the vocabulary fall back to their UTF-8 bytes.
- Case, Bengali marks, ZWJ/ZWNJ, digits, punctuation and emoji are preserved. Nothing is transliterated, lowercased or stemmed.
- Control IDs are `0 <pad>`, `1 <unk>` (never emitted), `2 <s>` and `3 </s>`. `encode` never adds them. Use `tokenizer(texts, add_special_tokens=True)` when you need them.
- They export to Hugging Face format with `save_huggingface(directory)`, but their vocabulary IDs differ from pretrained encoders such as BanglaBERT or XLM-R. Using one with such an encoder means re-initializing or adapting its embeddings.

## Limitations

- The training data is Wikipedia-heavy Bangla, sentence-level English, and Banglish from social-media comments. Domains far from these (legal, medical, code, other scripts) will fragment more.
- The code-mixed bucket is small: 4,709 training records and 156 validation records.
- Better compression and speed do not by themselves mean better downstream accuracy. See [research/reports/ROUND2_ORIGINAL_COMPARISON.md](../research/reports/ROUND2_ORIGINAL_COMPARISON.md) for the encoder experiments.

### `bn-bpe-28k-v2` evaluation

We compared the released 0.3.0 model with the unchanged original 28k model.
Lower tokens per whitespace word is better.

| Evaluation corpus | `bn-bpe-28k` | `bn-bpe-28k-v2` | Change |
|---|---:|---:|---:|
| Unseen-source BanglishRev reviews, 10,000 texts | 1.5874 | 1.5699 | -1.10% |
| Frozen v1 validation, 14,779 texts | 1.5628 | 1.5532 | -0.61% |
| New-source public holdout | 1.6202 | 1.5136 | -6.58% |
| YouTube ASR holdout, 16 videos | 1.5224 | 1.5053 | -1.12% |

On the BanglishRev reviews, `bn-bpe-28k-v2` measured 102.5 MB/s for individual
calls, 123.5 MB/s for batches of 64 on one Apple M5 thread, and 192.8 MB/s for
batches of 1,024 on all 10 cores. All round trips were exact and no unknown
token was emitted. The full same-machine comparison is in
[POPULAR_COMPARISON.md](../benchmarks/POPULAR_COMPARISON.md). Download the
standard Hugging Face files from
[`iamaber/banglatok-bn-bpe-28k-v2`](https://huggingface.co/iamaber/banglatok-bn-bpe-28k-v2).

The candidate passed every 28k promotion gate. The source-matched public
holdout overstates general-domain confidence; BanglishRev improved by only
1.1%, and the 156-text frozen code-mixed bucket regressed by 0.75%, within the
2% gate. Agent-generated synthetic Banglish was tested separately and rejected
because it worsened BanglishRev; it is not in this model.

## License of the model files

The software is MIT-licensed. The vocabularies and cached word lists are
statistics learned from attributed text. To respect the source terms
conservatively, all bundled model files are distributed under **CC BY-SA
4.0**, with attribution to:

- Wikipedia contributors, Bengali Wikipedia dump `20231101.bn` (via `wikimedia/wikipedia`).
- Tatoeba contributors, via ManyThings.org Anki exports (CC BY 2.0).
- The BanglaTLit authors (MIT), https://github.com/farhanishmam/BanglaTLit.
- Global Voices Bengali authors and translators (CC BY 3.0),
  https://bn.globalvoices.org/ and its
  [attribution policy](https://globalvoices.org/about/global-voices-attribution-policy/).
- Storybooks Bangladesh writers and translators for the included CC BY 3.0 or
  CC BY 4.0 stories,
  https://global-asp.github.io/storybooks-bangladesh/stories/bn/.

See [MODEL_LICENSE.md](../MODEL_LICENSE.md).

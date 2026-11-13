# Bundled models

Both models ship inside the wheel and load offline with `Tokenizer.from_pretrained(name)`.

| Name | Vocabulary | Base BPE | Fused IDs | Use it when |
|---|---:|---:|---:|---|
| `bn-bpe-58k` (default) | 57,925 | 32,000 | 25,925 | You want the shortest sequences |
| `bn-bpe-28k` | 27,760 | 16,000 | 11,760 | You want a smaller embedding table |

The vocabulary counts include 4 control IDs and 256 byte-fallback IDs.

## How they were built

1. `bntok train -algorithm bpe` learned the base vocabulary from the training split described in [data/CORPUS.md](../data/CORPUS.md). That split has 563,476 records: Bengali Wikipedia, Tatoeba sentence pairs, and BanglaTLit romanized and Bengali text. Its buckets are Bangla, English, Banglish and code-mixed.
2. `benchmarks/build_fused_model.py` added the space-fusion IDs. Each one stands for a space followed by one base ID. Candidates are the base IDs that most often start a word in the training split, with a minimum count of 2.
3. The same script stored the 64,000 most frequent training words that are not already pieces as `cache_words`. Their segmentation is computed once when the model loads. This only makes encoding faster and never changes IDs.

```sh
python benchmarks/build_fused_model.py --base models/bpe-32000.json --budget 64000 \
  --output models/release/fused-32000-64k.json   # → bn-bpe-58k
python benchmarks/build_fused_model.py --base models/bpe-16000.json --budget 32000 \
  --output models/release/fused-16000-32k.json   # → bn-bpe-28k
```

| Artifact | SHA-256 |
|---|---|
| `bn-bpe-58k.json` | `f045ed2c69e8c10d718272ae40291d51ca7453cc6abd26ef58eedddcaaaa13b9` |
| `bn-bpe-28k.json` | `b5b97c3a9472798e22da0d32182c5bc4e9a4b89a74f0a6af7bc301c1a0985cbf` |
| Base `bpe-32000.json` | `b3fcc85abb34e27648fc297dfc8052cf871f7c86482a4d4e91626a72513a0bc3` |
| Base `bpe-16000.json` | `77ac7594a514f76eb78b22c95b173c96fb83b08f0e75338af60261bebfa28306` |
| Training split | `bda48c8388fcd1cd706b4578619367a609ccf75a7f3aa659fede8c88aa198b39` |

## Behavior

- Lossless: `decode(encode(text)) == normalize(text)` for any string. Normalization is NFC plus whitespace collapse and trim.
- No unknown tokens. Characters not in the vocabulary fall back to their UTF-8 bytes.
- Case, Bengali marks, ZWJ/ZWNJ, digits, punctuation and emoji are preserved. Nothing is transliterated, lowercased or stemmed.
- Control IDs are `0 <pad>`, `1 <unk>` (never emitted), `2 <s>` and `3 </s>`. `encode` never adds them. Use `tokenizer(texts, add_special_tokens=True)` when you need them.
- These are not SentencePiece or Hugging Face vocabularies. Using one with a pretrained encoder means re-initializing or adapting that encoder's embeddings.

## Limitations

- The training data is Wikipedia-heavy Bangla, sentence-level English, and Banglish from social-media comments. Domains far from these (legal, medical, code, other scripts) will fragment more.
- The code-mixed bucket is small: 4,709 training records and 156 validation records.
- Better compression and speed do not by themselves mean better downstream accuracy. See [research/reports/ROUND2_ORIGINAL_COMPARISON.md](../research/reports/ROUND2_ORIGINAL_COMPARISON.md) for the encoder experiments.

## License of the model files

The software is MIT-licensed. The vocabularies and cached word lists are statistics learned from text, including Wikipedia (CC BY-SA 3.0 / GFDL) and Tatoeba (CC BY 2.0). To respect those terms conservatively, the two model JSON files are distributed under **CC BY-SA 4.0**, with attribution to:

- Wikipedia contributors, Bengali Wikipedia dump `20231101.bn` (via `wikimedia/wikipedia`).
- Tatoeba contributors, via ManyThings.org Anki exports (CC BY 2.0).
- The BanglaTLit authors (MIT), https://github.com/farhanishmam/BanglaTLit.

See [MODEL_LICENSE.md](../MODEL_LICENSE.md).

# Comparison with popular Bangla-capable tokenizers

Measured 10 November 2026 on an Apple M5 (arm64), macOS 27.0.1, Python 3.12.13, with bn-tokenizers-embedding 0.2.0. Raw results, timing samples and hashes are in [popular-comparison.json](popular-comparison.json). The harness is [benchmarks/compare_popular.py](../benchmarks/compare_popular.py).

**Summary.** On text from a source it never trained on, the default `bn-bpe-58k` model:

- produces the fewest tokens of every tokenizer tested;
- is faster in single-call and batch encoding than every other tokenizer tested (only our own `bn-bpe-28k` is quicker, by about 2%);
- reproduces every input exactly. Besides our own 28k model, only the SentencePiece model we trained on our data, the two tiktoken encodings and Qwen3 also do that, and they need 1.10× to 2.92× as many tokens on this corpus.

The compact `bn-bpe-28k` model is just as lossless and fast and still beats every published Bangla tokenizer on lossless compression, with a 28k-entry vocabulary.

## Unseen-source corpus: BanglishRev product reviews

These are 10,000 unique, naturally written e-commerce reviews (12–2,000 characters) from [BanglishRev](https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset) (CC BY-NC-SA 4.0). They were sampled deterministically from the first 64 MiB of `reviews v1.json` (SHA-256 `553b8e1e…`), which is 1,297,966 UTF-8 bytes. None of this source is in our training data. Buckets are by script: Bengali only, Latin only (English and Banglish), and mixed.

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| **bn-bpe-58k (this package)** | 57,925 | **1.379** | **100.00%** | **0** | **61.1** | **71.0** |
| bn-bpe-28k (this package) | 27,760 | 1.587 | 100.00% | 0 | 62.1 | 72.8 |
| BNLP bn_spm (SentencePiece) | 50,000 | 1.825 | 80.14% | 1,932 | 40.5 | 39.0 |
| SentencePiece Unigram 32k, our training data | 32,000 | 1.518 | 100.00% | 0 | 45.7 | 43.6 |
| tiktoken o200k_base (GPT-4o) | 200,019 | 1.780 | 100.00% | 0 | 37.4 | 12.0 |
| tiktoken cl100k_base (GPT-4) | 100,277 | 4.027 | 100.00% | 0 | 24.3 | 9.8 |
| BanglaBERT (csebuetnlp) | 32,000 | 1.771 | 51.10% | 1,110 | 9.2 | 9.2 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.507 | 0.32% | 7,410 | 6.9 | 6.9 |
| MuRIL (Google) | 197,258 | 1.498 | 47.94% | 1,617 | 8.8 | 8.8 |
| mBERT cased | 119,547 | 2.122 | 47.97% | 1,706 | 8.1 | 8.1 |
| XLM-R / multilingual-E5 / BGE-M3 | 250,002 | 1.740 | 98.16% | 109 | 6.5 | 6.5 |
| Qwen3 | 151,669 | 3.588 | 100.00% | 0 | 6.0 | 6.0 |

| Tokenizer | Bengali script | Latin script | Mixed script |
|---|---:|---:|---:|
| **bn-bpe-58k** | 1.381 | 1.373 | **1.418** |
| bn-bpe-28k | 1.603 | 1.570 | 1.633 |
| BNLP bn_spm | 1.396 | 2.157 | 1.545 |
| SentencePiece 32k, our data | 1.493 | 1.531 | 1.543 |
| tiktoken o200k_base | 2.254 | 1.420 | 2.050 |
| BanglaBERT | **1.343** | 2.099 | 1.515 |
| Bangla-BERT (sagorsarker) | 1.866 | **1.234** | 1.705 |
| MuRIL | 1.499 | 1.499 | 1.486 |
| XLM-R / mE5 / BGE-M3 | 2.024 | 1.520 | 1.927 |
| Qwen3 | 6.282 | 1.504 | 5.323 |

Two tokenizers beat ours on a single script, and both lose text. BanglaBERT is 2.8% shorter on Bengali-only text, but it reproduces only half of all inputs. Sagorsarker's Bangla-BERT is shorter on Latin-script text because it lowercases and strips accents; it reproduces 0.32% of inputs and emits `[UNK]` in 74% of them. Among lossless tokenizers, ours is the most compact on every script.

## In-domain corpus: frozen validation split

These are the 14,779 validation records from [data/CORPUS.md](../data/CORPUS.md): 3,013,257 bytes in Bangla, Banglish, code-mixed and English buckets. Our models were trained on the training split of the same sources, so this corpus favors them. The unseen-source table above is the fairer comparison.

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| **bn-bpe-58k** | 57,925 | **1.373** | **100.00%** | **0** | 77.4 | 88.3 |
| bn-bpe-28k | 27,760 | 1.563 | 100.00% | 0 | **78.9** | **89.7** |
| BNLP bn_spm | 50,000 | 1.861 | 87.95% | 647 | 42.8 | 41.7 |
| SentencePiece 32k, our data | 32,000 | 1.459 | 100.00% | 0 | 49.1 | 47.7 |
| tiktoken o200k_base | 200,019 | 2.153 | 100.00% | 0 | 33.0 | 15.0 |
| tiktoken cl100k_base | 100,277 | 5.430 | 100.00% | 0 | 24.1 | 12.3 |
| BanglaBERT | 32,000 | 1.791 | 50.35% | 1,579 | 10.6 | 10.6 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.698 | 0.64% | 11,971 | 7.9 | 7.9 |
| MuRIL | 197,258 | 1.629 | 50.19% | 352 | 10.1 | 10.0 |
| mBERT cased | 119,547 | 2.481 | 49.89% | 1,641 | 8.9 | 8.8 |
| XLM-R / mE5 / BGE-M3 | 250,002 | 2.035 | 91.60% | 20 | 7.7 | 7.6 |
| Qwen3 | 151,669 | 4.852 | 100.00% | 0 | 6.6 | 6.5 |

On 16 authored stress cases (emoji, ZWJ/ZWNJ, NUL, combining marks, other scripts, literal `<unk>`/`▁` text, URLs and an empty string), both of our models, both tiktoken encodings and Qwen3 reproduce all 16. SentencePiece reproduces 15, XLM-R 11, BNLP 8, mBERT 6, BanglaBERT and MuRIL 5, and sagorsarker's Bangla-BERT 1.

## Method

- **Same input for everyone.** Text is NFC- and whitespace-normalized once, before any candidate sees it. "Lossless round trip" means `decode(encode(text)) == text` on that normalized text. Tokens per word divides all emitted IDs, without special tokens, by whitespace-separated words.
- **Timing.** Single-threaded throughout: `TOKENIZERS_PARALLELISM=false`, `RAYON_NUM_THREADS=1`, `num_threads=1` for SentencePiece and tiktoken batches. Each value is the median of five full passes, with candidate and mode order shuffled on every pass. Timed work runs from the public encode call to a Python list of integer IDs. Hugging Face models use their Rust `tokenizers` backend directly, the fastest path they offer. Batches hold 64 texts. Model loading and quality checks are not timed.
- **Candidates.** BNLP's `SentencepieceTokenizer` wraps exactly this `bn_spm.model`; an earlier run measured the wrapper itself with identical IDs ([FRAMEWORK_COMPARISON.md](FRAMEWORK_COMPARISON.md)). We trained "SentencePiece Unigram 32k, our data" on our training split to show what the same data gives an established library. It is not a published package. Multilingual-E5 and BGE-M3 use the XLM-R tokenizer unchanged. Versions: sentencepiece 0.2.3, tiktoken 0.14.0, tokenizers 0.23.3, transformers 5.22.2.

## Limitations

- One machine, warm caches, five repeats. Other CPUs will give other absolute numbers, and the ratios may move too.
- Fewer tokens means shorter sequences and lower compute or API cost per text. It does not by itself mean better downstream accuracy. Our vocabularies only help an encoder or LLM that was trained or adapted to use them; see [ROUND2_ORIGINAL_COMPARISON.md](ROUND2_ORIGINAL_COMPARISON.md).
- Script buckets on BanglishRev do not separate English from Banglish within Latin-script text.
- BanglishRev is review text. Formal domains such as news and law, and other dialects, may behave differently.

## Reproduce

```sh
python -m pip install . -r benchmarks/popular-requirements.txt
# validation split: see data/CORPUS.md (bntok fetch && bntok prepare)
mkdir -p data/raw/banglishrev
curl -L -r 0-67108863 -o data/raw/banglishrev/reviews-v1-head64m.json \
  "https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset/resolve/main/reviews%20v1.json"
curl -L -o models/comparison/bnlp.model \
  https://raw.githubusercontent.com/sagorbrur/bnlp/master/model/bn_spm.model
TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 python benchmarks/compare_popular.py \
  --sentencepiece models/comparison/bnlp.model \
  --sentencepiece models/comparison/sentencepiece-32000.model --cpu-model "your CPU"
```

Either corpus may be omitted; the harness evaluates whichever files exist. `benchmarks/train_comparison.py` builds the same-data SentencePiece model.

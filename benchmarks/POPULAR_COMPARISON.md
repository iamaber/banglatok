# Comparison with popular Bangla-capable tokenizers

Measured 18 November 2026 on an Apple M5 (arm64, macOS 27), Python 3.12, with banglatok from `main` at commit `d93d39d`. That commit is the 0.2.3 code plus the unreleased encoder, boundary and load changes listed in the [changelog](../CHANGELOG.md). The raw results are in [results/popular-comparison.json](results/popular-comparison.json), and the harness is [compare_popular.py](compare_popular.py). Every timing is **single-threaded**.

An earlier run measured banglatok 0.2.0 on both this M5 and an AMD EPYC 9V74 GitHub-hosted Linux x86_64 VM. Its Linux results are in [results/popular-comparison-linux-x86_64.json](results/popular-comparison-linux-x86_64.json). That run predates the encoder changes, so its banglatok speeds are not comparable with the tables below.

**Summary.** On text from a source it never trained on:

- Against tokenizers people use for Bangla today, `bn-bpe-58k` gives the shortest lossless sequences. It emits 22% fewer tokens than BanglaBERT and `o200k_base`, 21% fewer than XLM-R, and 8% fewer than MuRIL.
- Against SentencePiece trained on **the same data at the same vocabulary size**, compression is a tie. `bn-bpe-58k` is about 2% shorter, and `bn-bpe-28k` is about 1% longer. Most of the advantage above therefore comes from the training data and vocabulary size, not from the algorithm.
- It is the fastest tokenizer tested. Across single calls and batches of 64, it is 2.3–3.0× as fast as SentencePiece Unigram at the same size, 5.7–7.4× as fast as SentencePiece BPE, 2.7–12× as fast as tiktoken, and 11–21× as fast as Hugging Face `tokenizers`. On single calls alone, the lead over tiktoken is 2.7–4.2×.
- Both models reproduce every input exactly.

## Unseen-source BanglishRev product reviews

These are 10,000 unique, naturally written e-commerce reviews (12–2,000 characters) from [BanglishRev](https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset) (CC BY-NC-SA 4.0). They were sampled deterministically from the first 64 MiB of `reviews v1.json` (SHA-256 `553b8e1e…`), which is 1,297,966 UTF-8 bytes. None of this source is in our training data. Buckets are by script: Bengali only, Latin only (English and Banglish), and mixed.

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| **banglatok `bn-bpe-58k`** | 57,925 | **1.379** | **100.00%** | **0** | 98.2 | **122.0** |
| banglatok `bn-bpe-28k` | 27,760 | 1.587 | 100.00% | 0 | **101.1** | 120.4 |
| SentencePiece Unigram 57,925, same training data | 57,925 | 1.410 | 100.00% | 0 | 42.3 | 40.9 |
| SentencePiece BPE 57,925, same training data | 57,925 | 1.407 | 100.00% | 0 | 17.3 | 16.5 |
| SentencePiece Unigram 27,760, same training data | 27,760 | 1.572 | 100.00% | 0 | 43.7 | 43.4 |
| SentencePiece BPE 27,760, same training data | 27,760 | 1.578 | 100.00% | 0 | 16.2 | 16.3 |
| BNLP bn_spm (SentencePiece) | 50,000 | 1.825 | 80.14% | 1,932 | 39.4 | 38.2 |
| tiktoken o200k_base (GPT-4o) | 200,019 | 1.780 | 100.00% | 0 | 36.6 | 11.7 |
| tiktoken cl100k_base (GPT-4) | 100,277 | 4.027 | 100.00% | 0 | 24.1 | 9.8 |
| BanglaBERT (csebuetnlp) | 32,000 | 1.771 | 51.10% | 1,110 | 9.0 | 9.0 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.507 | 0.32% | 7,410 | 6.8 | 6.8 |
| MuRIL (Google) | 197,258 | 1.498 | 47.94% | 1,617 | 8.6 | 8.7 |
| mBERT cased | 119,547 | 2.122 | 47.97% | 1,706 | 7.9 | 7.9 |
| XLM-R / multilingual-E5 / BGE-M3 | 250,002 | 1.740 | 98.16% | 109 | 6.4 | 6.4 |
| Qwen3 | 151,669 | 3.588 | 100.00% | 0 | 6.0 | 5.9 |

| Tokenizer | Bengali script | Latin script | Mixed script |
|---|---:|---:|---:|
| **banglatok `bn-bpe-58k`** | 1.381 | **1.373** | **1.418** |
| banglatok `bn-bpe-28k` | 1.603 | 1.570 | 1.633 |
| SentencePiece Unigram 57,925, our data | 1.397 | 1.414 | 1.440 |
| SentencePiece BPE 57,925, our data | 1.392 | 1.415 | 1.428 |
| SentencePiece Unigram 27,760, our data | 1.571 | 1.565 | 1.614 |
| SentencePiece BPE 27,760, our data | 1.569 | 1.580 | 1.610 |
| BNLP bn_spm | 1.396 | 2.157 | 1.545 |
| tiktoken o200k_base | 2.254 | 1.420 | 2.050 |
| BanglaBERT | **1.343** | 2.099 | 1.515 |
| Bangla-BERT (sagorsarker) | 1.866 | 1.234 | 1.705 |
| MuRIL | 1.499 | 1.499 | 1.486 |
| XLM-R / mE5 / BGE-M3 | 2.024 | 1.520 | 1.927 |
| Qwen3 | 6.282 | 1.504 | 5.323 |

Two tokenizers beat ours on a single script, and both lose text. BanglaBERT is 2.8% shorter on Bengali-only text, but it reproduces only half of all inputs. Sagorsarker's Bangla-BERT is shorter on Latin-script text because it lowercases and strips accents; it reproduces 0.32% of inputs and emits `[UNK]` in 74% of them. Among lossless tokenizers, `bn-bpe-58k` is the most compact on every script, but only by 1–3% over same-size SentencePiece.

## In-domain frozen validation split

These are the 14,779 validation records from [data/CORPUS.md](../data/CORPUS.md): 3,013,257 bytes in Bangla, Banglish, code-mixed and English buckets. All models trained on our data saw the training split of the same sources, so this corpus favors them over the published tokenizers. The unseen-source tables above are the fairer comparison.

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| **banglatok `bn-bpe-58k`** | 57,925 | **1.373** | **100.00%** | **0** | **134.8** | **154.3** |
| banglatok `bn-bpe-28k` | 27,760 | 1.563 | 100.00% | 0 | 133.4 | 151.0 |
| SentencePiece Unigram 57,925, our data | 57,925 | 1.384 | 100.00% | 0 | 44.8 | 43.5 |
| SentencePiece BPE 57,925, our data | 57,925 | 1.381 | 100.00% | 0 | 19.0 | 18.9 |
| SentencePiece Unigram 27,760, our data | 27,760 | 1.495 | 100.00% | 0 | 47.9 | 47.0 |
| SentencePiece BPE 27,760, our data | 27,760 | 1.512 | 100.00% | 0 | 18.0 | 17.9 |
| BNLP bn_spm | 50,000 | 1.861 | 87.95% | 647 | 42.2 | 41.4 |
| tiktoken o200k_base | 200,019 | 2.153 | 100.00% | 0 | 32.4 | 14.7 |
| tiktoken cl100k_base | 100,277 | 5.430 | 100.00% | 0 | 23.9 | 11.6 |
| BanglaBERT | 32,000 | 1.791 | 50.35% | 1,579 | 10.3 | 10.2 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.698 | 0.64% | 11,971 | 7.8 | 7.7 |
| MuRIL | 197,258 | 1.629 | 50.19% | 352 | 9.9 | 9.8 |
| mBERT cased | 119,547 | 2.481 | 49.89% | 1,641 | 8.7 | 8.7 |
| XLM-R / mE5 / BGE-M3 | 250,002 | 2.035 | 91.60% | 20 | 7.5 | 7.2 |
| Qwen3 | 151,669 | 4.852 | 100.00% | 0 | 6.4 | 6.3 |

In-domain, `bn-bpe-58k` is 0.6–0.8% shorter than same-size SentencePiece, while same-size SentencePiece is 3.4–4.5% shorter than `bn-bpe-28k`.

On 16 authored stress cases (emoji, ZWJ/ZWNJ, NUL, combining marks, other scripts, literal `<unk>`/`▁` text, URLs and an empty string), both of our models, both tiktoken encodings and Qwen3 reproduce all 16. The SentencePiece models trained on our data reproduce 15, XLM-R 11, BNLP 8, mBERT 6, BanglaBERT and MuRIL 5, and sagorsarker's Bangla-BERT 1.

## Method

- **Same input for everyone.** Text is NFC- and whitespace-normalized once, before any candidate sees it. "Lossless round trip" means `decode(encode(text)) == text` on that normalized text. Tokens per word divides all emitted IDs, without special tokens, by whitespace-separated words.
- **Timing.** Single-threaded throughout: `TOKENIZERS_PARALLELISM=false`, `RAYON_NUM_THREADS=1`, `num_threads=1` for SentencePiece and tiktoken batches. Each value is the median of five full passes, with candidate and mode order shuffled on every pass. Timed work runs from the public encode call to a Python list of integer IDs. Hugging Face models use their Rust `tokenizers` backend directly, the fastest path they offer. Batches hold 64 texts. Model loading and quality checks are not timed. Multi-threaded batch encoding is not measured.
- **Same-data baselines.** [train_sentencepiece_baselines.py](train_sentencepiece_baselines.py) trains SentencePiece Unigram and BPE on our normalized training split with byte fallback, `character_coverage=1.0`, identity normalization, and a total vocabulary equal to each bundled model's (27,760 and 57,925). Settings and model hashes are in [results/sentencepiece-baselines.json](results/sentencepiece-baselines.json). They are not published packages.
- **Candidates.** BNLP's `SentencepieceTokenizer` wraps exactly this `bn_spm.model`; an earlier run measured the wrapper itself with identical IDs ([FRAMEWORK_COMPARISON.md](../research/reports/FRAMEWORK_COMPARISON.md)). Multilingual-E5 and BGE-M3 use the XLM-R tokenizer unchanged. Versions: sentencepiece 0.2.3, tiktoken 0.14.0, tokenizers 0.23.3, transformers 5.22.2.
- **Repeat run.** A second full run on the same machine gave banglatok speeds within 4% of the table (unseen-source single calls 98.0 and 101.9 MB/s; batches 119.4 and 119.9 MB/s). Tokens per word and losslessness are deterministic and match the 0.2.0 run exactly.

## Limitations

- One machine, warm caches, five repeats per run. Ratios differ by CPU: with 0.2.0, the lead over SentencePiece Unigram was 1.3–1.7× on the M5 but 1.1–1.4× on the Linux VM. Linux has not been rerun with the current encoder.
- Fewer tokens means shorter sequences and lower compute or API cost per text. It does not by itself mean better downstream accuracy. Our vocabularies only help an encoder or LLM that was trained or adapted to use them; see [ROUND2_ORIGINAL_COMPARISON.md](../research/reports/ROUND2_ORIGINAL_COMPARISON.md).
- Script buckets on BanglishRev do not separate English from Banglish within Latin-script text.
- BanglishRev is review text. Formal domains such as news and law, and other dialects, may behave differently.

## Reproduce

```sh
uv pip install . -r benchmarks/popular-requirements.txt     # or: python -m pip install ...
# training and validation splits: see data/CORPUS.md (bntok fetch && bntok prepare)
python benchmarks/train_sentencepiece_baselines.py
mkdir -p data/raw/banglishrev
curl -L -r 0-67108863 -o data/raw/banglishrev/reviews-v1-head64m.json \
  "https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset/resolve/main/reviews%20v1.json"
curl -L -o models/comparison/bnlp.model \
  https://raw.githubusercontent.com/sagorbrur/bnlp/master/model/bn_spm.model
TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 python benchmarks/compare_popular.py \
  --sentencepiece models/comparison/bnlp.model \
  --sentencepiece models/comparison/sentencepiece-unigram-57925.model \
  --sentencepiece models/comparison/sentencepiece-bpe-57925.model \
  --sentencepiece models/comparison/sentencepiece-unigram-27760.model \
  --sentencepiece models/comparison/sentencepiece-bpe-27760.model --cpu-model "your CPU"
```

Either corpus may be omitted; the harness evaluates whichever files exist.

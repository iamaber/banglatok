# Comparison with popular Bangla-capable tokenizers

Measured 13 November 2026 with banglatok 0.2.0, Python 3.12, on two machines:

- an Apple M5 (arm64), macOS 27.0.1: [results/popular-comparison.json](results/popular-comparison.json), both corpora;
- an AMD EPYC 9V74 GitHub-hosted `ubuntu-latest` VM (x86_64, 4 vCPUs): [results/popular-comparison-linux-x86_64.json](results/popular-comparison-linux-x86_64.json), unseen-source corpus only, because the in-domain split is not distributed.

The harness is [compare_popular.py](compare_popular.py). Every timing is **single-threaded**.

**Summary.** On text from a source it never trained on:

- Against tokenizers people use for Bangla today, `bn-bpe-58k` gives the shortest lossless sequences: 22% fewer tokens than BanglaBERT and `o200k_base`, 21% fewer than XLM-R, 8% fewer than MuRIL.
- Against SentencePiece trained on **the same data at the same vocabulary size**, compression is a tie: `bn-bpe-58k` is about 2% shorter, `bn-bpe-28k` about 1% longer. Most of the advantage above therefore comes from the training data and vocabulary size, not from the algorithm.
- It is the fastest tokenizer tested on both machines: 1.1–1.7× SentencePiece Unigram, 2.6–4.4× SentencePiece BPE, 1.3–5.9× tiktoken and 5–12× Hugging Face `tokenizers`, all single-threaded.
- Both models reproduce every input exactly.

## Unseen-source corpus: BanglishRev product reviews

These are 10,000 unique, naturally written e-commerce reviews (12–2,000 characters) from [BanglishRev](https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset) (CC BY-NC-SA 4.0). They were sampled deterministically from the first 64 MiB of `reviews v1.json` (SHA-256 `553b8e1e…`), which is 1,297,966 UTF-8 bytes. None of this source is in our training data. Buckets are by script: Bengali only, Latin only (English and Banglish), and mixed.

Tokens per word and losslessness are identical on both machines. Speed is MB/s for single calls / batches of 64.

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | MB/s, Apple M5 | MB/s, Linux x86_64 |
|---|---:|---:|---:|---:|---:|---:|
| **banglatok `bn-bpe-58k`** | 57,925 | **1.379** | **100.00%** | **0** | 58.5 / 71.6 | 24.4 / 29.7 |
| banglatok `bn-bpe-28k` | 27,760 | 1.587 | 100.00% | 0 | **61.6 / 72.9** | **26.1 / 33.6** |
| SentencePiece Unigram 57,925, same training data | 57,925 | 1.410 | 100.00% | 0 | 42.4 / 41.7 | 20.5 / 20.9 |
| SentencePiece BPE 57,925, same training data | 57,925 | 1.407 | 100.00% | 0 | 17.7 / 17.4 | 9.5 / 9.8 |
| SentencePiece Unigram 27,760, same training data | 27,760 | 1.572 | 100.00% | 0 | 46.4 / 43.2 | 22.8 / 23.4 |
| SentencePiece BPE 27,760, same training data | 27,760 | 1.578 | 100.00% | 0 | 16.6 / 16.4 | 9.8 / 9.8 |
| BNLP bn_spm (SentencePiece) | 50,000 | 1.825 | 80.14% | 1,932 | 40.1 / 39.2 | 19.3 / 19.8 |
| tiktoken o200k_base (GPT-4o) | 200,019 | 1.780 | 100.00% | 0 | 37.3 / 12.1 | 19.4 / 5.4 |
| tiktoken cl100k_base (GPT-4) | 100,277 | 4.027 | 100.00% | 0 | 24.5 / 9.7 | 12.3 / 4.9 |
| BanglaBERT (csebuetnlp) | 32,000 | 1.771 | 51.10% | 1,110 | 9.1 / 9.2 | 4.8 / 4.6 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.507 | 0.32% | 7,410 | 6.9 / 6.9 | 3.7 / 3.5 |
| MuRIL (Google) | 197,258 | 1.498 | 47.94% | 1,617 | 8.8 / 8.9 | 4.6 / 4.4 |
| mBERT cased | 119,547 | 2.122 | 47.97% | 1,706 | 8.1 / 8.1 | 4.2 / 4.1 |
| XLM-R / multilingual-E5 / BGE-M3 | 250,002 | 1.740 | 98.16% | 109 | 6.5 / 6.4 | 3.7 / 3.5 |
| Qwen3 | 151,669 | 3.588 | 100.00% | 0 | 6.0 / 6.0 | 3.3 / 3.1 |

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

## In-domain corpus: frozen validation split (Apple M5)

These are the 14,779 validation records from [data/CORPUS.md](../data/CORPUS.md): 3,013,257 bytes in Bangla, Banglish, code-mixed and English buckets. All models trained on our data saw the training split of the same sources, so this corpus favors them over the published tokenizers. The unseen-source tables above are the fairer comparison.

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| **banglatok `bn-bpe-58k`** | 57,925 | **1.373** | **100.00%** | **0** | 77.5 | 88.2 |
| banglatok `bn-bpe-28k` | 27,760 | 1.563 | 100.00% | 0 | **78.9** | **90.8** |
| SentencePiece Unigram 57,925, our data | 57,925 | 1.384 | 100.00% | 0 | 45.5 | 44.6 |
| SentencePiece BPE 57,925, our data | 57,925 | 1.381 | 100.00% | 0 | 19.5 | 19.2 |
| SentencePiece Unigram 27,760, our data | 27,760 | 1.495 | 100.00% | 0 | 49.1 | 47.9 |
| SentencePiece BPE 27,760, our data | 27,760 | 1.512 | 100.00% | 0 | 18.3 | 18.2 |
| BNLP bn_spm | 50,000 | 1.861 | 87.95% | 647 | 42.6 | 42.2 |
| tiktoken o200k_base | 200,019 | 2.153 | 100.00% | 0 | 33.0 | 14.9 |
| tiktoken cl100k_base | 100,277 | 5.430 | 100.00% | 0 | 24.3 | 12.0 |
| BanglaBERT | 32,000 | 1.791 | 50.35% | 1,579 | 10.6 | 10.6 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.698 | 0.64% | 11,971 | 7.9 | 7.8 |
| MuRIL | 197,258 | 1.629 | 50.19% | 352 | 10.1 | 10.0 |
| mBERT cased | 119,547 | 2.481 | 49.89% | 1,641 | 8.9 | 8.8 |
| XLM-R / mE5 / BGE-M3 | 250,002 | 2.035 | 91.60% | 20 | 7.5 | 7.6 |
| Qwen3 | 151,669 | 4.852 | 100.00% | 0 | 6.6 | 6.5 |

In-domain, `bn-bpe-58k` is 0.6–0.8% shorter than same-size SentencePiece, while same-size SentencePiece is 3.4–4.5% shorter than `bn-bpe-28k`.

On 16 authored stress cases (emoji, ZWJ/ZWNJ, NUL, combining marks, other scripts, literal `<unk>`/`▁` text, URLs and an empty string), both of our models, both tiktoken encodings and Qwen3 reproduce all 16. The SentencePiece models trained on our data reproduce 15, XLM-R 11, BNLP 8, mBERT 6, BanglaBERT and MuRIL 5, and sagorsarker's Bangla-BERT 1.

## Method

- **Same input for everyone.** Text is NFC- and whitespace-normalized once, before any candidate sees it. "Lossless round trip" means `decode(encode(text)) == text` on that normalized text. Tokens per word divides all emitted IDs, without special tokens, by whitespace-separated words.
- **Timing.** Single-threaded throughout: `TOKENIZERS_PARALLELISM=false`, `RAYON_NUM_THREADS=1`, `num_threads=1` for SentencePiece and tiktoken batches. Each value is the median of five full passes, with candidate and mode order shuffled on every pass. Timed work runs from the public encode call to a Python list of integer IDs. Hugging Face models use their Rust `tokenizers` backend directly, the fastest path they offer. Batches hold 64 texts. Model loading and quality checks are not timed. Multi-threaded batch encoding is not measured.
- **Same-data baselines.** [train_sentencepiece_baselines.py](train_sentencepiece_baselines.py) trains SentencePiece Unigram and BPE on our normalized training split with byte fallback, `character_coverage=1.0`, identity normalization, and a total vocabulary equal to each bundled model's (27,760 and 57,925). Settings and model hashes are in [results/sentencepiece-baselines.json](results/sentencepiece-baselines.json). They are not published packages.
- **Candidates.** BNLP's `SentencepieceTokenizer` wraps exactly this `bn_spm.model`; an earlier run measured the wrapper itself with identical IDs ([FRAMEWORK_COMPARISON.md](../research/reports/FRAMEWORK_COMPARISON.md)). Multilingual-E5 and BGE-M3 use the XLM-R tokenizer unchanged. Versions: sentencepiece 0.2.3, tiktoken 0.14.0, tokenizers 0.23.3, transformers 5.22.2.
- **Linux run.** A one-off GitHub Actions job installed the package from source, downloaded the same BanglishRev prefix and BNLP model, and used the same SentencePiece baseline files (identical SHA-256).

## Limitations

- Two machines, warm caches, five repeats. The Linux machine is a shared cloud VM, so its absolute numbers are noisier. Ratios differ by CPU: against SentencePiece Unigram the lead is 1.3–1.7× on the M5 but 1.1–1.4× on the Linux VM.
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

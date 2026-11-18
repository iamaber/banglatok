# Comparison with popular Bangla-capable tokenizers

Measured 18 November 2026 with banglatok 0.2.4 and Python 3.12 on two machines:

- an Apple M5 (arm64, 10 cores), macOS 27: [results/popular-comparison.json](results/popular-comparison.json);
- an AMD EPYC 9V74 GitHub-hosted `ubuntu-latest` VM (x86_64, 4 vCPUs): [results/popular-comparison-linux-x86_64.json](results/popular-comparison-linux-x86_64.json).

Both machines read byte-identical corpora and model files. The Linux job downloaded the published training and validation splits from Hugging Face, retrained the same-data SentencePiece baselines and checked that their SHA-256 values match [results/sentencepiece-baselines.json](results/sentencepiece-baselines.json). The harness is [compare_popular.py](compare_popular.py). Every timing in the tables is **single-threaded**. Multi-core banglatok batches are reported separately.

**Summary.** On text from a source it never trained on:

- Against tokenizers people use for Bangla today, `bn-bpe-58k` gives the shortest lossless sequences. It emits 22% fewer tokens than BanglaBERT and `o200k_base`, 21% fewer than XLM-R, and 8% fewer than MuRIL.
- Against SentencePiece trained on **the same data at the same vocabulary size**, compression is a tie. `bn-bpe-58k` is about 2% shorter, and `bn-bpe-28k` is about 1% longer. Most of the advantage above therefore comes from the training data and vocabulary size, not from the algorithm.
- It is the fastest tokenizer tested on both machines. Across single calls and batches of 64, it is 1.8–2.9× as fast as SentencePiece Unigram at the same size, 4.1–7.3× as fast as SentencePiece BPE, 2.0–12× as fast as tiktoken, and 8–20× as fast as Hugging Face `tokenizers`. On single calls alone, the lead over tiktoken is 2.0–4.2×.
- Both models reproduce every input exactly.

## Unseen-source BanglishRev product reviews

These are 10,000 unique, naturally written e-commerce reviews (12–2,000 characters) from [BanglishRev](https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset) (CC BY-NC-SA 4.0). They were sampled deterministically from the first 64 MiB of `reviews v1.json` (SHA-256 `553b8e1e…`), which is 1,297,966 UTF-8 bytes. None of this source is in our training data. Buckets are by script: Bengali only, Latin only (English and Banglish), and mixed.

Speed is MB/s for single calls / batches of 64. Tokens per word and losslessness are identical on both machines.

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | MB/s, Apple M5 | MB/s, Linux x86_64 |
|---|---:|---:|---:|---:|---:|---:|
| **banglatok `bn-bpe-58k`** | 57,925 | **1.379** | **100.00%** | **0** | 99.4 / **119.4** | 40.9 / **55.4** |
| banglatok `bn-bpe-28k` | 27,760 | 1.587 | 100.00% | 0 | **101.1** / 118.6 | **41.6** / 54.2 |
| BNLP bn_spm (SentencePiece) | 50,000 | 1.825 | 80.14% | 1,932 | 39.0 / 38.0 | 20.4 / 20.9 |
| SentencePiece Unigram 57,925, same training data | 57,925 | 1.410 | 100.00% | 0 | 43.0 / 40.9 | 21.5 / 22.0 |
| SentencePiece BPE 57,925, same training data | 57,925 | 1.407 | 100.00% | 0 | 17.4 / 17.1 | 10.1 / 10.2 |
| SentencePiece Unigram 27,760, same training data | 27,760 | 1.572 | 100.00% | 0 | 45.1 / 43.1 | 23.3 / 24.0 |
| SentencePiece BPE 27,760, same training data | 27,760 | 1.578 | 100.00% | 0 | 16.4 / 16.3 | 10.1 / 10.2 |
| tiktoken o200k_base (GPT-4o) | 200,019 | 1.780 | 100.00% | 0 | 36.6 / 12.4 | 20.5 / 5.6 |
| tiktoken cl100k_base (GPT-4) | 100,277 | 4.027 | 100.00% | 0 | 24.0 / 9.6 | 12.7 / 5.0 |
| BanglaBERT (csebuetnlp) | 32,000 | 1.771 | 51.10% | 1,110 | 8.9 / 9.0 | 4.9 / 4.7 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.507 | 0.32% | 7,410 | 6.7 / 6.8 | 3.8 / 3.6 |
| MuRIL (Google) | 197,258 | 1.498 | 47.94% | 1,617 | 8.5 / 8.7 | 4.8 / 4.6 |
| mBERT cased | 119,547 | 2.122 | 47.97% | 1,706 | 7.9 / 8.0 | 4.4 / 4.1 |
| XLM-R / multilingual-E5 / BGE-M3 | 250,002 | 1.740 | 98.16% | 109 | 6.3 / 6.3 | 3.2 / 3.0 |
| Qwen3 | 151,669 | 3.588 | 100.00% | 0 | 5.9 / 5.9 | 3.2 / 3.0 |

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

These are the 14,779 validation records from [data/CORPUS.md](../data/CORPUS.md): 3,013,257 bytes in Bangla, Banglish, code-mixed and English buckets. All models trained on our data saw the training split of the same sources, so this corpus favors them over the published tokenizers. The unseen-source tables above are the fairer comparison. Speed is MB/s for single calls / batches of 64.

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | MB/s, Apple M5 | MB/s, Linux x86_64 |
|---|---:|---:|---:|---:|---:|---:|
| **banglatok `bn-bpe-58k`** | 57,925 | **1.373** | **100.00%** | **0** | **133.2 / 152.4** | 54.5 / **70.4** |
| banglatok `bn-bpe-28k` | 27,760 | 1.563 | 100.00% | 0 | 131.7 / 147.5 | **54.7** / 70.3 |
| BNLP bn_spm | 50,000 | 1.861 | 87.95% | 647 | 42.0 / 40.7 | 21.8 / 22.2 |
| SentencePiece Unigram 57,925, our data | 57,925 | 1.384 | 100.00% | 0 | 45.0 / 43.8 | 23.2 / 23.3 |
| SentencePiece BPE 57,925, our data | 57,925 | 1.381 | 100.00% | 0 | 18.9 / 18.7 | 11.0 / 11.1 |
| SentencePiece Unigram 27,760, our data | 27,760 | 1.495 | 100.00% | 0 | 48.2 / 47.3 | 25.9 / 26.3 |
| SentencePiece BPE 27,760, our data | 27,760 | 1.512 | 100.00% | 0 | 17.9 / 17.8 | 11.1 / 11.2 |
| tiktoken o200k_base | 200,019 | 2.153 | 100.00% | 0 | 32.0 / 14.4 | 19.3 / 7.4 |
| tiktoken cl100k_base | 100,277 | 5.430 | 100.00% | 0 | 23.7 / 11.8 | 12.7 / 5.8 |
| BanglaBERT | 32,000 | 1.791 | 50.35% | 1,579 | 10.3 / 10.2 | 5.7 / 5.5 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.698 | 0.64% | 11,971 | 7.5 / 7.5 | 4.3 / 4.1 |
| MuRIL | 197,258 | 1.629 | 50.19% | 352 | 9.6 / 9.5 | 5.4 / 5.2 |
| mBERT cased | 119,547 | 2.481 | 49.89% | 1,641 | 8.5 / 8.6 | 4.7 / 4.6 |
| XLM-R / mE5 / BGE-M3 | 250,002 | 2.035 | 91.60% | 20 | 7.2 / 7.2 | 3.8 / 3.7 |
| Qwen3 | 151,669 | 4.852 | 100.00% | 0 | 6.3 / 6.2 | 3.4 / 3.3 |

In-domain, `bn-bpe-58k` is 0.6–0.8% shorter than same-size SentencePiece, while same-size SentencePiece is 3.4–4.5% shorter than `bn-bpe-28k`.

On 16 authored stress cases (emoji, ZWJ/ZWNJ, NUL, combining marks, other scripts, literal `<unk>`/`▁` text, URLs and an empty string), both of our models, both tiktoken encodings and Qwen3 reproduce all 16. The SentencePiece models trained on our data reproduce 15, XLM-R 11, BNLP 8, mBERT 6, BanglaBERT and MuRIL 5, and sagorsarker's Bangla-BERT 1.

## banglatok batches on every core

`encode_batch` spreads large batches across native worker threads. The harness times it separately, on every CPU of each machine, in batches of 1,024 texts. Before timing, it checks that the multi-threaded IDs equal a one-thread encode of the same texts. No other tokenizer was measured this way, so these numbers are not a comparison.

| Corpus | Model | Apple M5, 10 threads | Linux x86_64, 4 threads |
|---|---|---:|---:|
| Unseen-source reviews | `bn-bpe-58k` | 228.6 MB/s | 93.6 MB/s |
| Unseen-source reviews | `bn-bpe-28k` | 222.4 MB/s | 90.3 MB/s |
| In-domain validation | `bn-bpe-58k` | 304.5 MB/s | 117.4 MB/s |
| In-domain validation | `bn-bpe-28k` | 309.0 MB/s | 114.1 MB/s |

On the M5, ten threads give about twice the one-thread batch speed, not ten times. The Go encoder scales across cores, but building the Python ID lists holds the GIL and runs on one core. The 10,000 reviews take only about 6 ms per pass at this speed, so treat the review numbers as approximate.

## Method

- **Same input for everyone.** Text is NFC- and whitespace-normalized once, before any candidate sees it. "Lossless round trip" means `decode(encode(text)) == text` on that normalized text. Tokens per word divides all emitted IDs, without special tokens, by whitespace-separated words.
- **Timing.** The comparison tables are single-threaded: `TOKENIZERS_PARALLELISM=false`, `RAYON_NUM_THREADS=1`, `num_threads=1` for banglatok, SentencePiece and tiktoken batches. Each value is the median of five full passes, with candidate and mode order shuffled on every pass. Timed work runs from the public encode call to a Python list of integer IDs. Hugging Face models use their Rust `tokenizers` backend directly, the fastest path they offer. Batches hold 64 texts. Model loading and quality checks are not timed.
- **Same-data baselines.** [train_sentencepiece_baselines.py](train_sentencepiece_baselines.py) trains SentencePiece Unigram and BPE on our normalized training split with byte fallback, `character_coverage=1.0`, identity normalization, and a total vocabulary equal to each bundled model's (27,760 and 57,925). Settings and model hashes are in [results/sentencepiece-baselines.json](results/sentencepiece-baselines.json). They are not published packages.
- **Candidates.** BNLP's `SentencepieceTokenizer` wraps exactly this `bn_spm.model`; an earlier run measured the wrapper itself with identical IDs ([FRAMEWORK_COMPARISON.md](../research/reports/FRAMEWORK_COMPARISON.md)). Multilingual-E5 and BGE-M3 use the XLM-R tokenizer unchanged. Versions: sentencepiece 0.2.3, tiktoken 0.14.0, tokenizers 0.23.3, transformers 5.22.2.
- **Repeat runs.** Three full runs on the M5 gave banglatok speeds within 4% of each other. Tokens per word, losslessness and UNK counts are deterministic: they match the 0.2.0 run exactly and are identical on both machines.

## Limitations

- Two machines, warm caches, five repeats per run. The Linux machine is a shared cloud VM, so its absolute numbers are noisier. Ratios differ by CPU. Against SentencePiece Unigram the lead is 2.2–2.9× on the M5 and 1.8–2.5× on the Linux VM on unseen text; with 0.2.0 it was 1.3–1.7× and 1.1–1.4×.
- Fewer tokens means shorter sequences and lower compute or API cost per text. It does not by itself mean better downstream accuracy. Our vocabularies only help an encoder or LLM that was trained or adapted to use them; see [ROUND2_ORIGINAL_COMPARISON.md](../research/reports/ROUND2_ORIGINAL_COMPARISON.md).
- Script buckets on BanglishRev do not separate English from Banglish within Latin-script text.
- BanglishRev is review text. Formal domains such as news and law, and other dialects, may behave differently.

## Reproduce

```sh
uv pip install . -r benchmarks/popular-requirements.txt     # or: python -m pip install ...
# training and validation splits: data/CORPUS.md (bntok fetch && bntok prepare), or the published
# splits at https://huggingface.co/datasets/iamaber/banglatok-training-corpus
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

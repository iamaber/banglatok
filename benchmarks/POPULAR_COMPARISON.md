# Comparison with popular Bangla-capable tokenizers

The 25 November 2026 Apple M5 run used banglatok 0.3.0 and Python 3.12.

- Apple M5 (arm64, 10 cores), macOS 27: [0.3.0 result summary](results/popular-comparison-0.3.0.md).
- Linux x86_64 values are the 18 November 2026 banglatok 0.2.4 run on an AMD
  EPYC 9V74 GitHub-hosted `ubuntu-latest` VM (4 vCPUs):
  [results/popular-comparison-linux-x86_64.json](results/popular-comparison-linux-x86_64.json).
  `bn-bpe-28k-v2` was not measured on that machine.

Both runs used byte-identical evaluation corpora and original model files. The Linux job downloaded the published training and validation splits from Hugging Face. It retrained the same-data SentencePiece baselines, then checked their SHA-256 values against [results/sentencepiece-baselines.json](results/sentencepiece-baselines.json). [compare_popular.py](compare_popular.py) runs the comparison. Every timing in the tables is **single-threaded**. A later section reports multi-core banglatok batches.

**Summary.** On text from a source it never trained on:

- `bn-bpe-58k` gives the shortest lossless sequences among tokenizers people use for Bangla today. It emits 22% fewer tokens than BanglaBERT and `o200k_base`, 21% fewer than XLM-R, and 8% fewer than MuRIL.
- `bn-bpe-28k-v2` uses 1.1% fewer tokens than the original 28k model on unseen-source reviews and 0.6% fewer on frozen validation. It is 0.1% shorter than the original-data SentencePiece Unigram 28k baseline on those reviews.
- Compression is otherwise close to SentencePiece trained on the same data at the same vocabulary size. Most of the advantage over popular tokenizers therefore comes from the training data and vocabulary size, not from the algorithm.
- banglatok is the fastest tokenizer tested. Across single calls and batches of 64 on the M5, it is 2.2 to 3.0× as fast as SentencePiece Unigram at the same size, 4.4 to 8.4× as fast as SentencePiece BPE, 2.8 to 10× as fast as tiktoken, and 11 to 21× as fast as Hugging Face `tokenizers`.
- All three bundled models reproduce every input exactly.

## Unseen-source BanglishRev product reviews

These are 10,000 unique, naturally written e-commerce reviews (12–2,000 characters) from [BanglishRev](https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset) (CC BY-NC-SA 4.0). They were sampled deterministically from the first 64 MiB of `reviews v1.json` (SHA-256 `553b8e1e…`), which is 1,297,966 UTF-8 bytes. None of this source is in our training data. Buckets are by script: Bengali only, Latin only (English and Banglish), and mixed.

Speed is MB/s for single calls / batches of 64. Tokens per word and losslessness are identical on both machines.

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | MB/s, Apple M5 | MB/s, Linux x86_64 |
|---|---:|---:|---:|---:|---:|---:|
| **banglatok `bn-bpe-58k`** | 57,925 | **1.379** | **100.00%** | **0** | 102.9 / **124.0** | 40.9 / **55.4** |
| banglatok `bn-bpe-28k-v2` | 27,760 | 1.570 | 100.00% | 0 | 102.5 / 123.5 | not measured |
| banglatok `bn-bpe-28k` | 27,760 | 1.587 | 100.00% | 0 | **103.1** / 121.1 | **41.6** / 54.2 |
| BNLP bn_spm (SentencePiece) | 50,000 | 1.825 | 80.14% | 1,932 | 40.5 / 39.4 | 20.4 / 20.9 |
| SentencePiece Unigram 57,925, same training data | 57,925 | 1.410 | 100.00% | 0 | 43.6 / 41.3 | 21.5 / 22.0 |
| SentencePiece BPE 57,925, same training data | 57,925 | 1.407 | 100.00% | 0 | 17.5 / 17.6 | 10.1 / 10.2 |
| SentencePiece Unigram 27,760, same training data | 27,760 | 1.572 | 100.00% | 0 | 45.6 / 44.1 | 23.3 / 24.0 |
| SentencePiece BPE 27,760, same training data | 27,760 | 1.578 | 100.00% | 0 | 16.6 / 16.6 | 10.1 / 10.2 |
| tiktoken o200k_base (GPT-4o) | 200,019 | 1.780 | 100.00% | 0 | 36.9 / 12.2 | 20.5 / 5.6 |
| tiktoken cl100k_base (GPT-4) | 100,277 | 4.027 | 100.00% | 0 | 24.6 / 10.0 | 12.7 / 5.0 |
| BanglaBERT (csebuetnlp) | 32,000 | 1.771 | 51.10% | 1,110 | 9.0 / 9.1 | 4.9 / 4.7 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.507 | 0.32% | 7,410 | 6.8 / 6.8 | 3.8 / 3.6 |
| MuRIL (Google) | 197,258 | 1.498 | 47.94% | 1,617 | 8.7 / 8.8 | 4.8 / 4.6 |
| mBERT cased | 119,547 | 2.122 | 47.97% | 1,706 | 8.0 / 8.1 | 4.4 / 4.1 |
| XLM-R / multilingual-E5 / BGE-M3 | 250,002 | 1.740 | 98.16% | 109 | 6.5 / 6.5 | 3.2 / 3.0 |
| Qwen3 | 151,669 | 3.588 | 100.00% | 0 | 6.0 / 6.0 | 3.2 / 3.0 |

| Tokenizer | Bengali script | Latin script | Mixed script |
|---|---:|---:|---:|
| **banglatok `bn-bpe-58k`** | 1.381 | **1.373** | **1.418** |
| banglatok `bn-bpe-28k-v2` | 1.595 | 1.544 | 1.627 |
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
| **banglatok `bn-bpe-58k`** | 57,925 | **1.373** | **100.00%** | **0** | **137.2 / 159.3** | 54.5 / **70.4** |
| banglatok `bn-bpe-28k-v2` | 27,760 | 1.553 | 100.00% | 0 | 135.9 / 154.6 | not measured |
| banglatok `bn-bpe-28k` | 27,760 | 1.563 | 100.00% | 0 | 135.7 / 153.0 | **54.7** / 70.3 |
| BNLP bn_spm | 50,000 | 1.861 | 87.95% | 647 | 43.0 / 42.9 | 21.8 / 22.2 |
| SentencePiece Unigram 57,925, our data | 57,925 | 1.384 | 100.00% | 0 | 45.4 / 45.1 | 23.2 / 23.3 |
| SentencePiece BPE 57,925, our data | 57,925 | 1.381 | 100.00% | 0 | 19.5 / 19.6 | 11.0 / 11.1 |
| SentencePiece Unigram 27,760, our data | 27,760 | 1.495 | 100.00% | 0 | 48.5 / 48.3 | 25.9 / 26.3 |
| SentencePiece BPE 27,760, our data | 27,760 | 1.512 | 100.00% | 0 | 18.4 / 18.6 | 11.1 / 11.2 |
| tiktoken o200k_base | 200,019 | 2.153 | 100.00% | 0 | 33.0 / 15.7 | 19.3 / 7.4 |
| tiktoken cl100k_base | 100,277 | 5.430 | 100.00% | 0 | 24.2 / 12.9 | 12.7 / 5.8 |
| BanglaBERT | 32,000 | 1.791 | 50.35% | 1,579 | 10.5 / 10.6 | 5.7 / 5.5 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.698 | 0.64% | 11,971 | 7.8 / 7.8 | 4.3 / 4.1 |
| MuRIL | 197,258 | 1.629 | 50.19% | 352 | 9.9 / 10.1 | 5.4 / 5.2 |
| mBERT cased | 119,547 | 2.481 | 49.89% | 1,641 | 8.8 / 8.9 | 4.7 / 4.6 |
| XLM-R / mE5 / BGE-M3 | 250,002 | 2.035 | 91.60% | 20 | 7.7 / 7.8 | 3.8 / 3.7 |
| Qwen3 | 151,669 | 4.852 | 100.00% | 0 | 6.6 / 6.5 | 3.4 / 3.3 |

In-domain, `bn-bpe-58k` is 0.6–0.8% shorter than same-size SentencePiece.
The original-data 28k SentencePiece baselines are 2.6–3.7% shorter than
`bn-bpe-28k-v2`, while the new model is 0.6% shorter than `bn-bpe-28k`.

On 16 authored stress cases (emoji, ZWJ/ZWNJ, NUL, combining marks, other scripts, literal `<unk>`/`▁` text, URLs and an empty string), all three bundled models, both tiktoken encodings and Qwen3 reproduce all 16. The SentencePiece models trained on our data reproduce 15, XLM-R 11, BNLP 8, mBERT 6, BanglaBERT and MuRIL 5, and sagorsarker's Bangla-BERT 1.

## banglatok batches on every core

`encode_batch` spreads large batches across native worker threads. The harness times it separately, on every CPU of each machine, in batches of 1,024 texts. Before timing, it checks that the multi-threaded IDs equal a one-thread encode of the same texts. No other tokenizer was measured this way, so these numbers are not a comparison.

| Corpus | Model | Apple M5, 10 threads | Linux x86_64, 4 threads |
|---|---|---:|---:|
| Unseen-source reviews | `bn-bpe-58k` | 218.7 MB/s | 93.6 MB/s |
| Unseen-source reviews | `bn-bpe-28k-v2` | 192.8 MB/s | not measured |
| Unseen-source reviews | `bn-bpe-28k` | 206.8 MB/s | 90.3 MB/s |
| In-domain validation | `bn-bpe-58k` | 325.9 MB/s | 117.4 MB/s |
| In-domain validation | `bn-bpe-28k-v2` | 309.3 MB/s | not measured |
| In-domain validation | `bn-bpe-28k` | 308.7 MB/s | 114.1 MB/s |

On the M5, ten threads give about twice the one-thread batch speed, not ten times. The Go encoder scales across cores, but building the Python ID lists holds the GIL and runs on one core. The 10,000 reviews take about 6 ms per pass at this speed, so treat the review numbers as approximate.

## Method

- **Same input for everyone.** Text is NFC- and whitespace-normalized once, before any candidate sees it. "Lossless round trip" means `decode(encode(text)) == text` on that normalized text. Tokens per word divides all emitted IDs, without special tokens, by whitespace-separated words.
- **Timing.** The comparison tables are single-threaded: `TOKENIZERS_PARALLELISM=false`, `RAYON_NUM_THREADS=1`, `num_threads=1` for banglatok, SentencePiece and tiktoken batches. Each value is the median of five full passes, with candidate and mode order shuffled on every pass. Timed work runs from the public encode call to a Python list of integer IDs. Hugging Face models use their Rust `tokenizers` backend directly, the fastest path they offer. Batches hold 64 texts. Model loading and quality checks are not timed.
- **Same-data baselines.** [train_sentencepiece_baselines.py](train_sentencepiece_baselines.py) trains SentencePiece Unigram and BPE on our normalized training split with byte fallback, `character_coverage=1.0`, identity normalization, and a total vocabulary equal to each bundled model's (27,760 and 57,925). Settings and model hashes are in [results/sentencepiece-baselines.json](results/sentencepiece-baselines.json). They are not published packages.
- **Candidates.** BNLP's `SentencepieceTokenizer` wraps exactly this `bn_spm.model`; an earlier run measured the wrapper itself with identical IDs ([FRAMEWORK_COMPARISON.md](../research/reports/FRAMEWORK_COMPARISON.md)). Multilingual-E5 and BGE-M3 use the XLM-R tokenizer unchanged. Versions: banglatok 0.3.0, sentencepiece 0.2.3, tiktoken 0.14.0, tokenizers 0.23.3, transformers 5.22.2.
- **Repeat runs.** The fresh native-only run and full comparison produced identical quality metrics. Three earlier full runs on the M5 put banglatok speeds within 4% of each other. Tokens per word, losslessness and UNK counts are deterministic.

## Limitations

- The benchmark used two machines, warm caches and five repeats per run. The Linux machine is a shared cloud VM, so its absolute numbers are noisier. We have not timed `bn-bpe-28k-v2` there. Against SentencePiece Unigram, the original models lead by 2.2 to 3.0× on the M5 and 1.8 to 2.5× on the Linux VM on unseen text.
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

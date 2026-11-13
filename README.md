# banglatok

**Fast, lossless subword tokenizers for Bangla, English and Banglish (romanized Bengali), including code-mixed text.**

[![CI](https://github.com/iamaber/banglatok/actions/workflows/ci.yml/badge.svg)](https://github.com/iamaber/banglatok/actions/workflows/ci.yml)
![Python 3.10–3.15](https://img.shields.io/badge/python-3.10%E2%80%933.15-blue)
![Runtime dependencies: none](https://img.shields.io/badge/runtime%20deps-none-brightgreen)
[![License: MIT (code), CC BY-SA 4.0 (models)](https://img.shields.io/badge/license-MIT%20%7C%20CC%20BY--SA%204.0-lightgrey)](#license)

```python
from banglatok import Tokenizer

tok = Tokenizer.from_pretrained()          # bundled model; no download, no network
tok.tokenize("meetingটা ১০টায়, don't be late 🙂")
# ['meeting', 'টা', ' ১০', 'টায়', ',', " don't", ' be', ' late', ' 🙂']
```

- **Short sequences.** On unseen product reviews `bn-bpe-58k` produced 22% fewer tokens than BanglaBERT and GPT-4o's `o200k_base`, 21% fewer than XLM-R/mE5/BGE-M3, 8% fewer than MuRIL, and 62% fewer than Qwen3. Most of that comes from training on Bangla and Banglish text at this vocabulary size: SentencePiece trained on the same data with the same vocabulary size comes within about 2% (see [Benchmarks](#benchmarks)).
- **Lossless.** `decode(encode(text))` gives back the normalized text exactly, for every input. There is no `[UNK]` and no lowercasing, and Bengali marks, ZWJ/ZWNJ, emoji and mixed scripts are kept.
- **Fast.** Single-threaded, on unseen text, it was 1.1–1.7× faster than SentencePiece Unigram and 2.6–4.4× faster than SentencePiece BPE at the same vocabulary size, 1.3–5.9× faster than tiktoken, and 5–12× faster than Hugging Face `tokenizers`, on both an Apple M5 and a Linux x86_64 server CPU.
- **Tiny footprint.** The engine is native Go behind `ctypes`, with zero Python runtime dependencies and typed APIs. It is thread-safe, two pretrained models ship inside the wheel, and they export to Hugging Face `tokenizers`/`transformers` with identical IDs.

The vocabularies are meant for training or adapting encoders and LLMs on Bengali text, and for counting or segmenting such text.

## Benchmarks

Here are 10,000 naturally written e-commerce reviews from [BanglishRev](https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset), in Bengali, English, Banglish and mixed script. This source was never seen in training. Every tokenizer runs **single-threaded** (`RAYON_NUM_THREADS=1`, `num_threads=1`), on an Apple M5 (macOS) and on an AMD EPYC 9V74 GitHub-hosted Linux x86_64 VM, Python 3.12. Speed is MB/s for single calls / batches of 64.

| Tokenizer | Vocab | Tokens / word ↓ | Lossless round trip ↑ | Texts with UNK ↓ | MB/s, Apple M5 ↑ | MB/s, Linux x86_64 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **bn-bpe-58k (this package)** | 57,925 | **1.379** | **100%** | **0** | 58.5 / 71.6 | 24.4 / 29.7 |
| **bn-bpe-28k (this package)** | 27,760 | 1.587 | **100%** | **0** | **61.6 / 72.9** | **26.1 / 33.6** |
| SentencePiece Unigram, same data and size as 58k | 57,925 | 1.410 | 100% | 0 | 42.4 / 41.7 | 20.5 / 20.9 |
| SentencePiece BPE, same data and size as 58k | 57,925 | 1.407 | 100% | 0 | 17.7 / 17.4 | 9.5 / 9.8 |
| SentencePiece Unigram, same data and size as 28k | 27,760 | 1.572 | 100% | 0 | 46.4 / 43.2 | 22.8 / 23.4 |
| SentencePiece BPE, same data and size as 28k | 27,760 | 1.578 | 100% | 0 | 16.6 / 16.4 | 9.8 / 9.8 |
| BNLP `bn_spm` (SentencePiece) | 50,000 | 1.825 | 80.1% | 1,932 | 40.1 / 39.2 | 19.3 / 19.8 |
| tiktoken `o200k_base` (GPT-4o) | 200,019 | 1.780 | 100% | 0 | 37.3 / 12.1 | 19.4 / 5.4 |
| BanglaBERT (csebuetnlp) | 32,000 | 1.771 | 51.1% | 1,110 | 9.1 / 9.2 | 4.8 / 4.6 |
| MuRIL (Google) | 197,258 | 1.498 | 47.9% | 1,617 | 8.8 / 8.9 | 4.6 / 4.4 |
| XLM-R / multilingual-E5 / BGE-M3 | 250,002 | 1.740 | 98.2% | 109 | 6.5 / 6.4 | 3.7 / 3.5 |
| mBERT cased | 119,547 | 2.122 | 48.0% | 1,706 | 8.1 / 8.1 | 4.2 / 4.1 |
| Qwen3 | 151,669 | 3.588 | 100% | 0 | 6.0 / 6.0 | 3.3 / 3.1 |

What this shows:

- **Against SentencePiece with the same data and vocabulary size, compression is a tie.** `bn-bpe-58k` is about 2% shorter than both SentencePiece models; `bn-bpe-28k` is about 1% longer. On the in-domain validation split, SentencePiece at the 28k size is 3–4.5% shorter. The gap to BanglaBERT, XLM-R and tiktoken comes from the training data and the vocabulary size, not from a better algorithm.
- **The speed lead is real but depends on the CPU.** Against SentencePiece Unigram it is 1.3–1.7× on the M5 and 1.1–1.4× on the Linux VM. Hugging Face `tokenizers` was limited to one thread like everything else; its batch API can use all cores, which this table does not measure.
- BanglaBERT is about 3% shorter on Bengali-only text but lossy. Among lossless tokenizers, `bn-bpe-58k` is the most compact on Bengali, Latin and mixed script.

[Full report](benchmarks/POPULAR_COMPARISON.md): per-script breakdown, the in-domain validation split, more tokenizers, method, limitations, and the commands to reproduce it. Shorter sequences cut compute and context use, but they do not by themselves make a downstream model more accurate.

## Install

```sh
pip install banglatok
```

Prebuilt wheels target Linux (x86_64, aarch64), macOS (x86_64, arm64) and Windows (x86_64) for CPython 3.10–3.15, and need no compiler. On other platforms pip builds from source, which needs Go 1.24+ and a C compiler:

```sh
pip install git+https://github.com/iamaber/banglatok
```

## Usage

```python
from banglatok import Tokenizer

with Tokenizer.from_pretrained("bn-bpe-58k") as tok:
    ids = tok.encode("আমি বাংলায় গান গাই। ami banglay gaan gai")
    tok.decode(ids)                       # 'আমি বাংলায় গান গাই। ami banglay gaan gai'
    tok.encode_batch(["আমি ভালো আছি", "kemon acho?"])   # one native call
    tok.count_tokens("অনেক লম্বা লেখা…")    # fast length, no list built

    # Encoder-ready inputs: <s> … </s>, truncation and padding with <pad>.
    batch = tok(["আমি ভালো আছি", "hi"], max_length=128, padding=True)
    batch["input_ids"], batch["attention_mask"]
```

Load a tokenizer once and reuse it. Calls from many threads at once are safe, because the native call releases the GIL. In worker processes, create a fresh tokenizer inside each worker.

### Hugging Face

```python
tok.save_huggingface("bn-bpe-58k-hf")      # tokenizer.json + tokenizer_config.json

from transformers import AutoTokenizer
hf = AutoTokenizer.from_pretrained("bn-bpe-58k-hf")   # same IDs as tok.encode / tok(...)
```

The exported models are also on the Hub as [`iamaber/banglatok-bn-bpe-58k`](https://huggingface.co/iamaber/banglatok-bn-bpe-58k) and [`iamaber/banglatok-bn-bpe-28k`](https://huggingface.co/iamaber/banglatok-bn-bpe-28k). Only BPE models can be exported. With the `tokenizers` library alone, set `encode_special_tokens = True` so that literal `<s>` in text stays text; `transformers` does this from the saved configuration.

### Command line

```sh
$ banglatok "আমি বাংলা বলি"
আমি |  বাংলা |  বলি
$ echo "ami bangla boli" | banglatok --ids
4405 33616 34749
$ banglatok --json "ami valo"
{"ids": [4405, 32047], "tokens": ["ami", " valo"]}
$ banglatok --count --model bn-bpe-28k < lines.txt    # one count per line
```

### API

| Member | Description |
|---|---|
| `Tokenizer.from_pretrained(name="bn-bpe-58k")` | Load a bundled model, offline |
| `Tokenizer(path)` | Load your own JSON model, e.g. one trained with `bntok train` |
| `encode(text)` / `encode_batch(texts)` | Token IDs, without control tokens |
| `decode(ids, skip_special_tokens=False)` / `decode_batch(batch)` | Normalized text |
| `tokenize(text)` | Token strings: fused IDs start with `" "`, fallback bytes look like `"<0xF0>"` |
| `count_tokens(text)` | `len(encode(text))`, without building the list |
| `save_huggingface(directory)` | Write a Hugging Face `tokenizer.json` with identical IDs (BPE models) |
| `tok(texts, add_special_tokens=True, max_length=None, padding=False)` | `{"input_ids", "attention_mask"}` |
| `id_to_token`, `token_to_id`, `get_vocab()`, `vocab_size`, `algorithm` | Vocabulary access |
| `normalize(text)` | The shared normalization: NFC plus whitespace collapse and trim |
| `available_models()`, `PAD_ID=0`, `UNK_ID=1`, `BOS_ID=2`, `EOS_ID=3` | Constants |

## Models

| Name | Vocab | Best for |
|---|---:|---|
| `bn-bpe-58k` (default) | 57,925 | Shortest sequences |
| `bn-bpe-28k` | 27,760 | Smaller embedding tables, same speed |

Both are BPE vocabularies with *space fusion*: a space and the token after it can share a single ID. That removes most standalone space tokens without overloading a literal marker such as `▁`. They were learned from 563k records of Bengali Wikipedia, Tatoeba and BanglaTLit. [docs/MODELS.md](docs/MODELS.md) covers the build commands, checksums, behavior and limitations. They are not SentencePiece or Hugging Face vocabularies, so pairing one with a pretrained encoder means adapting that encoder's embeddings.

## Train your own

The Go CLI trains Unigram or BPE models on your JSONL corpus, and `Tokenizer(path)` loads the result:

```sh
go run ./cmd/bntok train -algorithm bpe -vocab 384 \
  -input examples/training-records.jsonl -out models/toy-bpe.json
go run ./cmd/bntok encode -model models/toy-bpe.json -text 'আমি বাংলা বলি। ami bangla boli'
python benchmarks/build_fused_model.py --base models/my-bpe.json --budget 64000 \
  --output models/my-fused.json        # optional: add space fusion
```

See [docs/TRAINING_AND_RESEARCH.md](docs/TRAINING_AND_RESEARCH.md) for the corpus format, the full research pipeline and the model contract.

## Research

This repository also holds the research behind the models, under [research/](research/). [data/CORPUS.md](data/CORPUS.md) documents the corpus construction, provenance and leakage controls. [research/reports/RESULTS.md](research/reports/RESULTS.md) has the intrinsic evaluation. The retrieval experiments are in [research/experiments/retrieval/](research/experiments/retrieval/README.md), and [research/reports/ROUND2_ORIGINAL_COMPARISON.md](research/reports/ROUND2_ORIGINAL_COMPARISON.md) has the encoder-adaptation results. In those experiments, swapping a custom vocabulary into multilingual-E5 did not beat the unchanged model. Treat these vocabularies as a strong starting point for training, not as a drop-in accuracy gain for existing encoders.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) explains how to work on the project, [SECURITY.md](SECURITY.md) how to report vulnerabilities, and [docs/RELEASING.md](docs/RELEASING.md) how releases are made. Platform reports, regression cases and independent benchmark runs are especially welcome.

## Citation

If you use this library or its models, please cite it with the metadata in [CITATION.cff](CITATION.cff).

## License

- Code, documentation and the demo vocabulary: [MIT](LICENSE).
- Bundled model files (`bn-bpe-58k`, `bn-bpe-28k`): [CC BY-SA 4.0](MODEL_LICENSE.md), because they are learned from Wikipedia and Tatoeba text.
- Native wheels include BSD-3-Clause Go components ([notices](THIRD_PARTY_NOTICES.md)).

The SPDX expression is `MIT AND BSD-3-Clause AND CC-BY-SA-4.0`. Research corpora keep their own terms and are not distributed.

# banglatok

**Fast, lossless subword tokenizers for Bangla, English and Banglish (romanized Bengali), including code-mixed text.**

[![PyPI](https://img.shields.io/pypi/v/banglatok)](https://pypi.org/project/banglatok/)
[![CI](https://github.com/iamaber/banglatok/actions/workflows/ci.yml/badge.svg)](https://github.com/iamaber/banglatok/actions/workflows/ci.yml)
![Python 3.10–3.15](https://img.shields.io/badge/python-3.10%E2%80%933.15-blue)
![Runtime dependencies: none](https://img.shields.io/badge/runtime%20deps-none-brightgreen)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-iamaber%2Fbanglatok--bn--bpe--58k-yellow)](https://huggingface.co/iamaber/banglatok-bn-bpe-58k)
[![License: MIT (code), CC BY-SA 4.0 (models)](https://img.shields.io/badge/license-MIT%20%7C%20CC%20BY--SA%204.0-lightgrey)](#license)

```python
from banglatok import Tokenizer

tok = Tokenizer.from_pretrained()          # bundled model; no download, no network
tok.tokenize("meetingটা ১০টায়, don't be late 🙂")
# ['meeting', 'টা', ' ১০', 'টায়', ',', " don't", ' be', ' late', ' 🙂']
```

- **Short sequences.** On unseen product reviews banglatok's default model, `bn-bpe-58k`, produced 22% fewer tokens than BanglaBERT and GPT-4o's `o200k_base`, 21% fewer than XLM-R/mE5/BGE-M3, 8% fewer than MuRIL, and 62% fewer than Qwen3. Most of that comes from training on Bangla and Banglish text at this vocabulary size: SentencePiece trained on the same data with the same vocabulary size comes within about 2% (see [Benchmarks](#benchmarks)).
- **Lossless.** `decode(encode(text))` gives back the normalized text exactly, for every input. There is no `[UNK]` and no lowercasing, and Bengali marks, ZWJ/ZWNJ, emoji and mixed scripts are kept.
- **Fast.** Single-threaded, on unseen text, it was 1.1–1.7× faster than SentencePiece Unigram and 2.6–4.4× faster than SentencePiece BPE at the same vocabulary size, 1.3–5.9× faster than tiktoken, and 5–12× faster than Hugging Face `tokenizers`, on both an Apple M5 and a Linux x86_64 server CPU.
- **Easy to adopt.** The engine is native Go behind `ctypes`, with zero Python runtime dependencies and typed APIs. It is thread-safe, two pretrained models ship inside the wheel, and they are on the Hugging Face Hub with identical IDs.

The vocabularies are meant for training or adapting encoders and LLMs on Bengali text, and for counting or segmenting such text.

## Install

We recommend [uv](https://docs.astral.sh/uv/):

```sh
uv add banglatok                     # add to a uv project
uv pip install banglatok             # or install into the active virtual environment
uvx banglatok "আমি বাংলা বলি"          # or run the command once without installing
```

pip works too: `pip install banglatok`.

Prebuilt wheels cover CPython 3.10–3.15 on Linux (x86_64, aarch64), macOS (x86_64, arm64) and Windows (x86_64), and need no compiler. Other platforms build from source, which needs Go 1.24+ and a C compiler. See the [install guide](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md#install) for details.

## Usage

```python
from banglatok import Tokenizer

tok = Tokenizer.from_pretrained()                    # "bn-bpe-58k"; or from_pretrained("bn-bpe-28k")
ids = tok.encode("আমি বাংলায় গান গাই। ami banglay gaan gai")
tok.decode(ids)                                      # 'আমি বাংলায় গান গাই। ami banglay gaan gai'
tok.encode_batch(["আমি ভালো আছি", "kemon acho?"])     # many texts, one native call
tok.count_tokens("অনেক লম্বা লেখা…")                  # length without building the list

# Inputs for a model: <s> … </s>, truncation, padding with <pad>.
batch = tok(["আমি ভালো আছি", "hi"], max_length=128, padding=True)
batch["input_ids"], batch["attention_mask"]
```

```sh
$ banglatok "আমি বাংলা বলি"
আমি |  বাংলা |  বলি
$ echo "ami bangla boli" | banglatok --ids
4405 33616 34749
$ banglatok --count --model bn-bpe-28k < lines.txt    # one count per line
```

The **[user guide](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md)** covers every method, normalization, token strings, the command line, threads and processes, troubleshooting, and the full API reference.

## Guides

Short recipes; each links to a complete, tested version in the user guide.

**Use it from 🤗 Transformers.** Both models are on the Hub and need only `transformers`:

```python
from transformers import AutoTokenizer

hf = AutoTokenizer.from_pretrained("iamaber/banglatok-bn-bpe-58k")   # same IDs as banglatok
```

`tok.save_huggingface(directory)` exports any banglatok BPE model the same way. [More](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md#hugging-face)

**Tokenize a dataset.** The tokenizer returns plain lists, which `datasets` accepts directly:

```python
tokenized = dataset.map(lambda batch: tok(batch["text"], max_length=512), batched=True)
```

[More](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md#tokenize-a-dataset-with--datasets)

**Train a model from scratch.** A model needs only the vocabulary size and the control-token IDs:

```python
from transformers import GPT2Config, GPT2LMHeadModel
from banglatok import BOS_ID, EOS_ID, PAD_ID

model = GPT2LMHeadModel(GPT2Config(vocab_size=tok.vocab_size, n_embd=256, n_layer=4,
                                   n_head=4, bos_token_id=BOS_ID, eos_token_id=EOS_ID,
                                   pad_token_id=PAD_ID))
```

[More](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md#train-a-small-language-model-from-scratch)

**Count tokens against a budget**: [guide](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md#count-tokens-and-stay-within-a-budget). **Feed a PyTorch DataLoader**: [guide](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md#feed-a-pytorch-dataloader).

**Use several processes.** Threads can share one tokenizer. Processes must use the `spawn` or `forkserver` start method, because the Go runtime does not survive `fork()`. [More](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md#threads-and-processes)

## Benchmarks

Here are 10,000 naturally written e-commerce reviews from [BanglishRev](https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset), in Bengali, English, Banglish and mixed script. This source was never seen in training. Every tokenizer runs **single-threaded** (`RAYON_NUM_THREADS=1`, `num_threads=1`), on an Apple M5 (macOS) and on an AMD EPYC 9V74 GitHub-hosted Linux x86_64 VM, Python 3.12. Speed is MB/s for single calls / batches of 64.

| Tokenizer | Vocab | Tokens / word ↓ | Lossless round trip ↑ | Texts with UNK ↓ | MB/s, Apple M5 ↑ | MB/s, Linux x86_64 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **banglatok `bn-bpe-58k`** | 57,925 | **1.379** | **100%** | **0** | 58.5 / 71.6 | 24.4 / 29.7 |
| **banglatok `bn-bpe-28k`** | 27,760 | 1.587 | **100%** | **0** | **61.6 / 72.9** | **26.1 / 33.6** |
| SentencePiece Unigram, same data and size as `bn-bpe-58k` | 57,925 | 1.410 | 100% | 0 | 42.4 / 41.7 | 20.5 / 20.9 |
| SentencePiece BPE, same data and size as `bn-bpe-58k` | 57,925 | 1.407 | 100% | 0 | 17.7 / 17.4 | 9.5 / 9.8 |
| SentencePiece Unigram, same data and size as `bn-bpe-28k` | 27,760 | 1.572 | 100% | 0 | 46.4 / 43.2 | 22.8 / 23.4 |
| SentencePiece BPE, same data and size as `bn-bpe-28k` | 27,760 | 1.578 | 100% | 0 | 16.6 / 16.4 | 9.8 / 9.8 |
| BNLP `bn_spm` (SentencePiece) | 50,000 | 1.825 | 80.1% | 1,932 | 40.1 / 39.2 | 19.3 / 19.8 |
| tiktoken `o200k_base` (GPT-4o) | 200,019 | 1.780 | 100% | 0 | 37.3 / 12.1 | 19.4 / 5.4 |
| BanglaBERT (csebuetnlp) | 32,000 | 1.771 | 51.1% | 1,110 | 9.1 / 9.2 | 4.8 / 4.6 |
| MuRIL (Google) | 197,258 | 1.498 | 47.9% | 1,617 | 8.8 / 8.9 | 4.6 / 4.4 |
| XLM-R / multilingual-E5 / BGE-M3 | 250,002 | 1.740 | 98.2% | 109 | 6.5 / 6.4 | 3.7 / 3.5 |
| mBERT cased | 119,547 | 2.122 | 48.0% | 1,706 | 8.1 / 8.1 | 4.2 / 4.1 |
| Qwen3 | 151,669 | 3.588 | 100% | 0 | 6.0 / 6.0 | 3.3 / 3.1 |

What this shows:

- **Against SentencePiece with the same data and vocabulary size, compression is a tie.** `bn-bpe-58k` is about 2% shorter than both SentencePiece models; `bn-bpe-28k` is about 1% longer. On the in-domain validation split, SentencePiece at the `bn-bpe-28k` size is 3–4.5% shorter. The gap to BanglaBERT, XLM-R and tiktoken comes from the training data and the vocabulary size, not from a better algorithm.
- **The speed lead is real but depends on the CPU.** Against SentencePiece Unigram it is 1.3–1.7× on the M5 and 1.1–1.4× on the Linux VM. Hugging Face `tokenizers` was limited to one thread like everything else; its batch API can use all cores, which this table does not measure.
- BanglaBERT is about 3% shorter on Bengali-only text but lossy. Among lossless tokenizers, `bn-bpe-58k` is the most compact on Bengali, Latin and mixed script.

The [full report](https://github.com/iamaber/banglatok/blob/main/benchmarks/POPULAR_COMPARISON.md) has the per-script breakdown, the in-domain validation split, more tokenizers, the method, limitations, and the commands to reproduce it. Shorter sequences cut compute and context use, but they do not by themselves make a downstream model more accurate.

## Models

| Name | Vocab | Best for |
|---|---:|---|
| `bn-bpe-58k` (default) | 57,925 | Shortest sequences |
| `bn-bpe-28k` | 27,760 | Smaller embedding tables, same speed |

Both are BPE vocabularies with *space fusion*: a space and the token after it can share a single ID. That removes most standalone space tokens without overloading a literal marker such as `▁`. They were learned from 563k records of Bengali Wikipedia, Tatoeba and BanglaTLit. On the Hub they are [`iamaber/banglatok-bn-bpe-58k`](https://huggingface.co/iamaber/banglatok-bn-bpe-58k) and [`iamaber/banglatok-bn-bpe-28k`](https://huggingface.co/iamaber/banglatok-bn-bpe-28k). [docs/MODELS.md](https://github.com/iamaber/banglatok/blob/main/docs/MODELS.md) covers the build commands, checksums, behavior and limitations. Their vocabulary IDs differ from those of pretrained encoders such as BanglaBERT or XLM-R, so pairing one with such an encoder means adapting its embeddings.

## Documentation

| Document | Contents |
|---|---|
| [User guide](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md) | Install, API, command line, Hugging Face, guides, troubleshooting |
| [Models](https://github.com/iamaber/banglatok/blob/main/docs/MODELS.md) | How the bundled models were built, checksums, limitations |
| [Training and research](https://github.com/iamaber/banglatok/blob/main/docs/TRAINING_AND_RESEARCH.md) | Training your own models, corpus format, model contract |
| [Benchmarks](https://github.com/iamaber/banglatok/blob/main/benchmarks/POPULAR_COMPARISON.md) | Full comparison with popular tokenizers |
| [Changelog](https://github.com/iamaber/banglatok/blob/main/CHANGELOG.md) | Release notes |

## Train your own

The Go `bntok` command trains Unigram or BPE models on your JSONL corpus, and `Tokenizer(path)` loads the result:

```sh
go run ./cmd/bntok train -algorithm bpe -vocab 384 \
  -input examples/training-records.jsonl -out models/toy-bpe.json
go run ./cmd/bntok encode -model models/toy-bpe.json -text 'আমি বাংলা বলি। ami bangla boli'
python benchmarks/build_fused_model.py --base models/toy-bpe.json --budget 512 \
  --training examples/training-records.jsonl --output models/toy-fused.json
```

See [docs/TRAINING_AND_RESEARCH.md](https://github.com/iamaber/banglatok/blob/main/docs/TRAINING_AND_RESEARCH.md) for the corpus format, the full research pipeline and the model contract.

## Research

This repository also holds the research behind the models, under [research/](https://github.com/iamaber/banglatok/tree/main/research). [data/CORPUS.md](https://github.com/iamaber/banglatok/blob/main/data/CORPUS.md) documents the corpus construction, provenance and leakage controls. [RESULTS.md](https://github.com/iamaber/banglatok/blob/main/research/reports/RESULTS.md) has the intrinsic evaluation. The retrieval experiments are in [research/experiments/retrieval/](https://github.com/iamaber/banglatok/blob/main/research/experiments/retrieval/README.md), and [ROUND2_ORIGINAL_COMPARISON.md](https://github.com/iamaber/banglatok/blob/main/research/reports/ROUND2_ORIGINAL_COMPARISON.md) has the encoder-adaptation results. In those experiments, swapping a custom vocabulary into multilingual-E5 did not beat the unchanged model. Treat these vocabularies as a strong starting point for training, not as a drop-in accuracy gain for existing encoders.

## Contributing

[CONTRIBUTING.md](https://github.com/iamaber/banglatok/blob/main/CONTRIBUTING.md) explains how to work on the project, [SECURITY.md](https://github.com/iamaber/banglatok/blob/main/SECURITY.md) how to report vulnerabilities, and [docs/RELEASING.md](https://github.com/iamaber/banglatok/blob/main/docs/RELEASING.md) how releases are made. Platform reports, regression cases and independent benchmark runs are especially welcome.

## Citation

If you use banglatok or its models, please cite it with the metadata in [CITATION.cff](https://github.com/iamaber/banglatok/blob/main/CITATION.cff).

## License

- Code, documentation and the demo vocabulary: [MIT](https://github.com/iamaber/banglatok/blob/main/LICENSE).
- Bundled model files (`bn-bpe-58k`, `bn-bpe-28k`): [CC BY-SA 4.0](https://github.com/iamaber/banglatok/blob/main/MODEL_LICENSE.md), because they are learned from Wikipedia and Tatoeba text.
- Native wheels include BSD-3-Clause Go components ([notices](https://github.com/iamaber/banglatok/blob/main/THIRD_PARTY_NOTICES.md)).

The SPDX expression is `MIT AND BSD-3-Clause AND CC-BY-SA-4.0`. Research corpora keep their own terms and are not distributed.

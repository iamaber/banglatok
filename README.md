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

tok = Tokenizer.from_pretrained()
tok.tokenize("meetingটা ১০টায়, don't be late 🙂")
# ['meeting', 'টা', ' ১০', 'টায়', ',', " don't", ' be', ' late', ' 🙂']
```

banglatok splits real Bangla text, as people type it, into short and reversible token sequences. Use it to train or adapt encoders and LLMs on Bengali, to count tokens against a budget, or to segment text.

- **Fewer tokens.** On product reviews from a source it never trained on, the default model `bn-bpe-58k` emits 22% fewer tokens than BanglaBERT and GPT-4o's `o200k_base`, 21% fewer than XLM-R, mE5 and BGE-M3, 8% fewer than MuRIL, and 62% fewer than Qwen3.
- **Nothing lost.** `decode(encode(text))` returns the normalized text exactly, for every input. There is no `[UNK]` token and no lowercasing. Bengali combining marks, ZWJ and ZWNJ, emoji and mixed scripts all survive the round trip.
- **Fast on one core.** On one Apple M5 thread the three bundled models encode these reviews at about 103 MB/s. Batches of 64 run at 121 to 124 MB/s. That is 2.2 to 3.0× the throughput of SentencePiece Unigram and 2.8 to 10× the throughput of tiktoken `o200k_base`. Batches of 1,024 texts reached 193 to 219 MB/s on all 10 cores in the release run.
- **No dependencies.** A Go engine runs behind `ctypes`, so the wheel needs no other Python package and no compiler. The API is typed and thread-safe. All bundled models ship in the wheel with identical native and JSON behavior.

<img src="https://raw.githubusercontent.com/iamaber/banglatok/main/assets/benchmark.svg" alt="Tokens per word and single-thread MB/s for banglatok, same-data SentencePiece Unigram, tiktoken o200k_base, MuRIL, BanglaBERT, XLM-R and Qwen3 on 10,000 unseen BanglishRev reviews" width="100%">

Here is one product review ("The delivery came very late, but the product is good."), split by banglatok and by GPT-4o's tokenizer:

```text
text                        ডেলিভারি অনেক দেরিতে এসেছে কিন্তু প্রোডাক্ট ভালো

banglatok bn-bpe-58k (10)   ডেলি|ভারি| অনেক| দের|িতে| এসেছে| কিন্তু| প্রো|ডাক্ট| ভালো
tiktoken o200k_base  (16)   ড|েল|িভ|ারি| অনেক| দ|ের|িতে| এসেছে| কিন্তু| প্র|ো|ড|াক|্ট| ভালো
```

The same holds for romanized Bengali. `ami kal office e jabo, tumi asbe?` ("I'll go to the office tomorrow, will you come?") is 8 tokens in banglatok and 12 in `o200k_base`.

## Install

We recommend [uv](https://docs.astral.sh/uv/):

```sh
uv add banglatok                  # add it to a uv project
uv pip install banglatok          # or install it into the active virtual environment
uvx banglatok "আমি বাংলা বলি"       # or run the command line once without installing
```

pip works too:

```sh
pip install banglatok
```

Prebuilt wheels cover CPython 3.10–3.15 on Linux (x86_64, aarch64), macOS (x86_64, arm64) and Windows (x86_64). Other platforms build from source, which needs Go 1.24.9 or later and a C compiler. The [install guide](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md#install) has the details.

## Usage

```python
from banglatok import Tokenizer

tok = Tokenizer.from_pretrained()                    # default "bn-bpe-58k"; try "bn-bpe-28k-v2"
ids = tok.encode("আমি বাংলায় গান গাই। ami banglay gaan gai")
tok.decode(ids)                                      # 'আমি বাংলায় গান গাই। ami banglay gaan gai'
tok.encode_batch(["আমি ভালো আছি", "kemon acho?"])     # many texts, one native call
tok.count_tokens("অনেক লম্বা লেখা…")                  # length without building the list

# Inputs for a model: <s> … </s>, truncation, padding with <pad>.
batch = tok(["আমি ভালো আছি", "hi"], max_length=128, padding=True)
batch["input_ids"], batch["attention_mask"]
```

The same tokenizer works from the shell:

```sh
$ banglatok "আমি বাংলা বলি"
আমি |  বাংলা |  বলি
$ echo "ami bangla boli" | banglatok --ids
4405 33616 34749
$ banglatok --count --model bn-bpe-28k < lines.txt    # one count per line
```

The [user guide](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md) covers every method, normalization, token strings, the command line, threads and processes, and troubleshooting. It ends with the full API reference.

## Recipes

Each recipe links to a complete, tested version in the user guide.

### Load it in Transformers

The original models are on the Hub, and `transformers` alone can load them.

```python
from transformers import AutoTokenizer

hf = AutoTokenizer.from_pretrained("iamaber/banglatok-bn-bpe-58k")   # same IDs as banglatok
```

To export your own BPE model the same way, call `tok.save_huggingface(directory)`. [Hugging Face guide](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md#hugging-face)

### Tokenize a dataset

The tokenizer returns plain lists, and `datasets` accepts them as they are.

```python
tokenized = dataset.map(lambda batch: tok(batch["text"], max_length=512), batched=True)
```

[Dataset guide](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md#tokenize-a-dataset-with-hugging-face-datasets)

### Train a model from scratch

A new model needs only the vocabulary size and the IDs of the control tokens.

```python
from transformers import GPT2Config, GPT2LMHeadModel
from banglatok import BOS_ID, EOS_ID, PAD_ID

model = GPT2LMHeadModel(GPT2Config(vocab_size=tok.vocab_size, n_embd=256, n_layer=4,
                                   n_head=4, bos_token_id=BOS_ID, eos_token_id=EOS_ID,
                                   pad_token_id=PAD_ID))
```

[Training guide](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md#train-a-small-language-model-from-scratch)

The user guide also shows how to [count tokens against a budget](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md#count-tokens-and-stay-within-a-budget) and how to [feed a PyTorch DataLoader](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md#feed-a-pytorch-dataloader).

### Use several processes

Threads can share one tokenizer. Processes must start with the `spawn` or `forkserver` method, because the Go runtime does not survive `fork()`. [Threads and processes](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md#threads-and-processes)

## Benchmarks

The test set is 10,000 e-commerce reviews from [BanglishRev](https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset), written by real users in Bengali, English, Banglish and mixed script. None of this source is in the training data. Every tokenizer ran on one Apple M5 thread with Python 3.12 and banglatok 0.3.0 on 25 November 2026. Speed is the median of five passes, in MB/s, for single calls and batches of 64.

| Tokenizer | Vocab | Tokens per word ↓ | Lossless round trip ↑ | Texts with UNK ↓ | Single MB/s ↑ | Batch MB/s ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **banglatok `bn-bpe-58k`** | 57,925 | **1.379** | **100%** | **0** | 102.9 | **124.0** |
| **banglatok `bn-bpe-28k-v2`** | 27,760 | 1.570 | **100%** | **0** | 102.5 | 123.5 |
| banglatok `bn-bpe-28k` | 27,760 | 1.587 | 100% | 0 | **103.1** | 121.1 |
| SentencePiece Unigram, same data and size as `bn-bpe-58k` | 57,925 | 1.410 | 100% | 0 | 43.6 | 41.3 |
| SentencePiece BPE, same data and size as `bn-bpe-58k` | 57,925 | 1.407 | 100% | 0 | 17.5 | 17.6 |
| SentencePiece Unigram, same data and size as `bn-bpe-28k` | 27,760 | 1.572 | 100% | 0 | 45.6 | 44.1 |
| SentencePiece BPE, same data and size as `bn-bpe-28k` | 27,760 | 1.578 | 100% | 0 | 16.6 | 16.6 |
| BNLP `bn_spm` (SentencePiece) | 50,000 | 1.825 | 80.1% | 1,932 | 40.5 | 39.4 |
| tiktoken `o200k_base` (GPT-4o) | 200,019 | 1.780 | 100% | 0 | 36.9 | 12.2 |
| BanglaBERT (csebuetnlp) | 32,000 | 1.771 | 51.1% | 1,110 | 9.0 | 9.1 |
| MuRIL (Google) | 197,258 | 1.498 | 47.9% | 1,617 | 8.7 | 8.8 |
| XLM-R / multilingual-E5 / BGE-M3 | 250,002 | 1.740 | 98.2% | 109 | 6.5 | 6.5 |
| mBERT cased | 119,547 | 2.122 | 48.0% | 1,706 | 8.0 | 8.1 |
| Qwen3 | 151,669 | 3.588 | 100% | 0 | 6.0 | 6.0 |

`bn-bpe-28k-v2` uses 1.1% fewer tokens than the original 28k model on these
reviews while keeping its vocabulary size, exact round trips and zero-unknown
behavior. On all 10 M5 cores, batches of 1,024 reviews reached 219 MB/s with
`bn-bpe-58k` and 193 MB/s with `bn-bpe-28k-v2` in this run.

How to read the table:

- Compression comes mostly from the training data, not from the algorithm. SentencePiece trained on the original data at the same vocabulary size is within 2% of banglatok. `bn-bpe-28k-v2` is 0.1% shorter than the original-data 28k SentencePiece Unigram baseline on these reviews.
- The implementation accounts for the speed difference, and results depend on the CPU. On the M5, banglatok is 2.2 to 3.0× as fast as SentencePiece Unigram. On a 4-vCPU Linux x86_64 VM with an AMD EPYC 9V74, the original models measured 1.8 to 2.5× as fast. They reached 41 MB/s for single calls and 55 MB/s for batches of 64.
- BanglaBERT is about 3% shorter on Bengali-only text, but it reproduces only half of its inputs. Among lossless tokenizers, `bn-bpe-58k` is the most compact on Bengali, Latin and mixed script.
- Shorter sequences cut compute and context use. They do not by themselves make a downstream model more accurate.

The [full report](https://github.com/iamaber/banglatok/blob/main/benchmarks/POPULAR_COMPARISON.md) adds the per-script breakdown, the in-domain validation split, more tokenizers, the method, the limitations and the commands to reproduce every number.

## Models

| Name | Vocab | Choose it for |
|---|---:|---|
| `bn-bpe-58k` (default) | 57,925 | The shortest sequences |
| `bn-bpe-28k-v2` | 27,760 | The best current smaller model |
| `bn-bpe-28k` | 27,760 | Compatibility with original 28k token IDs |

All models are BPE vocabularies with *space fusion*. A space and the token after
it can share one ID, which removes most standalone space tokens without
reserving a marker character such as `▁`. `bn-bpe-28k-v2` adds licensed Global
Voices Bengali and Storybooks Bangladesh training text; it uses no YouTube,
viewer-comment or synthetic text. All three models are also on Hugging Face:
[`bn-bpe-58k`](https://huggingface.co/iamaber/banglatok-bn-bpe-58k),
[`bn-bpe-28k-v2`](https://huggingface.co/iamaber/banglatok-bn-bpe-28k-v2) and
[`bn-bpe-28k`](https://huggingface.co/iamaber/banglatok-bn-bpe-28k).

Their IDs differ from those of pretrained encoders such as BanglaBERT or XLM-R,
and the two 28k versions also differ from each other. To pair one with an
existing encoder, you must adapt the encoder's embeddings. In our experiments,
swapping a banglatok vocabulary into multilingual-E5 did not beat the unchanged
model. Treat these vocabularies as a starting point for training a model, not as
a drop-in accuracy gain for an existing encoder.

## Training data

The original models use the Hugging Face dataset [`iamaber/banglatok-training-corpus`](https://huggingface.co/datasets/iamaber/banglatok-training-corpus), with 563,476 training records, 14,779 validation records and 16,712 test records from Bengali Wikipedia, Tatoeba and BanglaTLit. `bn-bpe-28k-v2` adds licensed Global Voices Bengali and Storybooks Bangladesh text after leakage-aware deduplication. Only training splits were used to learn vocabularies, merges and word caches.

Each record keeps its source and attribution. The dataset card explains the source-specific licenses, the filtering and the privacy limitations. The MIT license of the software does not cover the corpus text.

## Train your own

The Go `bntok` command trains Unigram or BPE models on a JSONL corpus, and `Tokenizer(path)` loads the result:

```sh
go run ./cmd/bntok train -algorithm bpe -vocab 384 \
  -input examples/training-records.jsonl -out models/toy-bpe.json
banglatok -m models/toy-bpe.json --json 'আমি বাংলা বলি। ami bangla boli'
python benchmarks/build_fused_model.py --base models/toy-bpe.json --budget 512 \
  --training examples/training-records.jsonl --output models/toy-fused.json
```

[docs/TRAINING_AND_RESEARCH.md](https://github.com/iamaber/banglatok/blob/main/docs/TRAINING_AND_RESEARCH.md) describes the corpus format, the full research pipeline and the model contract.

## Documentation

| Document | Contents |
|---|---|
| [User guide](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md) | Install, API, command line, Hugging Face, recipes, troubleshooting |
| [Models](https://github.com/iamaber/banglatok/blob/main/docs/MODELS.md) | How the bundled models were built, checksums, limitations |
| [Training dataset](https://huggingface.co/datasets/iamaber/banglatok-training-corpus) | Downloadable splits, sources, attribution, licenses and data limitations |
| [Training and research](https://github.com/iamaber/banglatok/blob/main/docs/TRAINING_AND_RESEARCH.md) | Training your own models, corpus format, model contract |
| [Benchmarks](https://github.com/iamaber/banglatok/blob/main/benchmarks/POPULAR_COMPARISON.md) | Full comparison with popular tokenizers |
| [Changelog](https://github.com/iamaber/banglatok/blob/main/CHANGELOG.md) | Release notes |

## Contributing

[CONTRIBUTING.md](https://github.com/iamaber/banglatok/blob/main/CONTRIBUTING.md) explains how to build, test and change the project. [SECURITY.md](https://github.com/iamaber/banglatok/blob/main/SECURITY.md) explains how to report a vulnerability, and [docs/RELEASING.md](https://github.com/iamaber/banglatok/blob/main/docs/RELEASING.md) how releases are made. Reports from new platforms, regression cases and independent benchmark runs, especially on Windows and Linux arm64, are welcome.

## Citation

If you use banglatok or its models, cite it with the metadata in [CITATION.cff](https://github.com/iamaber/banglatok/blob/main/CITATION.cff).

## License

- Code, documentation and the demo vocabulary: [MIT](https://github.com/iamaber/banglatok/blob/main/LICENSE).
- Bundled model files (`bn-bpe-58k`, `bn-bpe-28k`, `bn-bpe-28k-v2`): [CC BY-SA 4.0](https://github.com/iamaber/banglatok/blob/main/MODEL_LICENSE.md), because they are learned from attributed third-party text.
- Native wheels include BSD-3-Clause Go components ([notices](https://github.com/iamaber/banglatok/blob/main/THIRD_PARTY_NOTICES.md)).

The SPDX expression is `MIT AND BSD-3-Clause AND CC-BY-SA-4.0`. Corpus text keeps its source-specific terms. It is published separately on Hugging Face and is not part of the Python package.

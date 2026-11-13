# banglatok user guide

banglatok is a fast, lossless subword tokenizer for Bangla, English and Banglish (romanized Bengali). This guide covers installation, the Python API, the `banglatok` command, Hugging Face integration, and recipes for common jobs.

- [Install](#install)
- [Quick start](#quick-start)
- [Choosing a model](#choosing-a-model)
- [Encoding and decoding](#encoding-and-decoding)
- [Tokens and the vocabulary](#tokens-and-the-vocabulary)
- [Inputs for a model](#inputs-for-a-model)
- [Hugging Face](#hugging-face)
- [Command line](#command-line)
- [Threads and processes](#threads-and-processes)
- [Your own models](#your-own-models)
- [Guides](#guides)
  - [Count tokens and stay within a budget](#count-tokens-and-stay-within-a-budget)
  - [Tokenize a dataset with 🤗 Datasets](#tokenize-a-dataset-with--datasets)
  - [Feed a PyTorch DataLoader](#feed-a-pytorch-dataloader)
  - [Train a small language model from scratch](#train-a-small-language-model-from-scratch)
- [Troubleshooting](#troubleshooting)
- [API reference](#api-reference)

## Install

We recommend [uv](https://docs.astral.sh/uv/) for fast installs and Python environment management. In a uv project, `uv add` also records dependencies in `pyproject.toml` and exact resolved versions in `uv.lock`; `uv pip install` only installs into an environment. pip remains fully supported.

```sh
uv add banglatok                     # add to a uv project (pyproject.toml + uv.lock)
uv pip install banglatok             # install into the active virtual environment
uvx banglatok "আমি বাংলা বলি"          # run the command once without installing
```

pip works just as well:

```sh
pip install banglatok
```

banglatok has no Python dependencies. Prebuilt wheels cover CPython 3.10–3.15 on:

| Platform | Requirement |
|---|---|
| Linux x86_64 and aarch64 | glibc 2.28 or newer (Ubuntu 20.04+, Debian 10+, RHEL 8+) |
| macOS x86_64 and arm64 | macOS 14 or newer |
| Windows x86_64 | Windows 10 or newer |

Elsewhere (Alpine/musl Linux, Windows on ARM, older macOS) the installer builds from source, which needs Go 1.24.9+ and a C compiler; on Windows that compiler is MinGW-w64 GCC. To install the development version from GitHub:

```sh
uv pip install git+https://github.com/iamaber/banglatok
```

Check the installation:

```sh
banglatok --version
```

## Quick start

```python
from banglatok import Tokenizer

tok = Tokenizer.from_pretrained()            # bundled bn-bpe-58k; no download, no network
ids = tok.encode("আমি বাংলায় গান গাই।")       # [4504, 35499, 32651, 37272, 1101]
tok.decode(ids)                              # 'আমি বাংলায় গান গাই।'
tok.tokenize("আমি বাংলায় গান গাই।")           # ['আমি', ' বাংলায়', ' গান', ' গাই', '।']
```

Load a tokenizer once and reuse it; loading parses the model and takes a fraction of a second. `with Tokenizer.from_pretrained() as tok:` releases the native model at the end of the block; otherwise it is released when the object is garbage-collected.

## Choosing a model

Two models ship inside the package and load offline:

`banglatok` is the package, import and command name. `bn-bpe-58k` and `bn-bpe-28k` are model identifiers, not separate packages. On Hugging Face they live at `iamaber/banglatok-bn-bpe-58k` and `iamaber/banglatok-bn-bpe-28k`.

| Name | Vocabulary | Use it when |
|---|---:|---|
| `bn-bpe-58k` (default) | 57,925 | You want the shortest sequences |
| `bn-bpe-28k` | 27,760 | You want a smaller embedding table; about 14–15% more tokens, same speed |

```python
from banglatok import Tokenizer, available_models

available_models()                           # ['bn-bpe-28k', 'bn-bpe-58k']
small = Tokenizer.from_pretrained("bn-bpe-28k")
```

Both learned from Bengali Wikipedia, Tatoeba and BanglaTLit. [MODELS.md](MODELS.md) has the build steps, checksums and limitations.

## Encoding and decoding

```python
tok.encode("ami bangla boli")                # one text -> list of IDs
tok.encode_batch(["আমি ভালো আছি", "kemon acho?"])   # many texts in one native call, same order
tok.decode([4405, 33616, 34749])             # 'ami bangla boli'
tok.decode_batch([[4405], [4504]])           # ['ami', 'আমি']
tok.count_tokens("ami bangla boli")          # 3, without building the list
```

**Normalization.** Before tokenizing, text is NFC-normalized, every run of whitespace becomes one space, and leading and trailing whitespace is removed. Nothing else changes: case, Bengali marks, ZWJ/ZWNJ, digits, punctuation and emoji are kept. `decode(encode(text))` returns exactly `normalize(text)`:

```python
from banglatok import normalize

normalize("  আমি\tবাংলা \n বলি  ")             # 'আমি বাংলা বলি'
```

**No unknown tokens.** A character that is not in the vocabulary is encoded as its UTF-8 bytes, so every input round-trips. Literal text such as `<unk>`, `<s>` or `▁` is ordinary text, never a control token.

**Control tokens.** `encode` never adds them. IDs 0–3 are `<pad>`, `<unk>`, `<s>` and `</s>`, available as `PAD_ID`, `UNK_ID`, `BOS_ID` and `EOS_ID`. `decode` raises `ValueError` on them unless you pass `skip_special_tokens=True`:

```python
from banglatok import BOS_ID, EOS_ID

tok.decode([BOS_ID, 4405, EOS_ID], skip_special_tokens=True)   # 'ami'
```

## Tokens and the vocabulary

```python
tok.tokenize("meetingটা ১০টায় 🙂")    # ['meeting', 'টা', ' ১০', 'টায়', ' 🙂']
tok.id_to_token(4405)                # 'ami'
tok.token_to_id(" বাংলা")             # 32430
tok.vocab_size                       # 57925
vocab = tok.get_vocab()              # {token: id} for every ID
```

How token strings look:

- A token that starts with a space, such as `" বাংলা"`, is a *fused* token: one ID for a space plus the token after it. Most spaces disappear into fused tokens.
- `" "` alone is the space character.
- `"<0xF0>"` is a raw byte, used for characters outside the vocabulary.
- `"".join(tok.tokenize(text)) == normalize(text)` whenever no byte tokens are involved.

## Inputs for a model

Calling the tokenizer returns `input_ids` and `attention_mask`, ready for an encoder or decoder. By default each sequence is wrapped in `<s>` … `</s>`.

```python
batch = tok(["আমি ভালো আছি", "hi"], max_length=128, padding=True)
batch["input_ids"]          # [[2, 4504, 32491, 45853, 3], [2, 4432, 3, 0, 0]]
batch["attention_mask"]     # [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
```

| Argument | Effect |
|---|---|
| `add_special_tokens=True` | Wrap in `<s>` … `</s>` |
| `max_length=None` | Truncate to this many IDs, keeping both control tokens |
| `padding=False` | `True` or `"longest"` pads to the longest sequence; `"max_length"` pads to `max_length` |

A single string returns flat lists; a list of strings returns nested lists. Convert with `torch.tensor(batch["input_ids"])` or `numpy.array(...)`.

## Hugging Face

The bundled models are on the Hugging Face Hub and work with `transformers` without banglatok installed:

Install this optional integration with `uv pip install transformers` (or `pip install transformers`).

```python
from transformers import AutoTokenizer

hf = AutoTokenizer.from_pretrained("iamaber/banglatok-bn-bpe-58k")   # or iamaber/banglatok-bn-bpe-28k
hf("আমি বাংলা বলি")["input_ids"]                                      # same IDs as tok("আমি বাংলা বলি")
```

To export any BPE model yourself:

```python
tok.save_huggingface("bn-bpe-58k-hf")        # writes tokenizer.json and tokenizer_config.json
hf = AutoTokenizer.from_pretrained("bn-bpe-58k-hf")
```

The export produces the same IDs and decoded text as banglatok. Use the native package for offline preprocessing without extra dependencies, and the Hugging Face tokenizer where another library expects one. With the `tokenizers` library alone, set `hf.encode_special_tokens = True` so literal `<s>` in text stays text; `transformers` does this from the saved configuration.

These vocabularies are new: a pretrained encoder such as BanglaBERT or XLM-R was trained with its own vocabulary, so switching it to banglatok means re-initializing and retraining its embeddings. They fit best when you train or continue training a model on them.

## Command line

```sh
$ banglatok "আমি বাংলা বলি"
আমি |  বাংলা |  বলি
$ echo "ami bangla boli" | banglatok --ids
4405 33616 34749
$ banglatok --json "ami valo"
{"ids": [4405, 32047], "tokens": ["ami", " valo"]}
$ banglatok --count --model bn-bpe-28k < lines.txt    # one count per input line
$ banglatok --list-models
bn-bpe-28k
bn-bpe-58k
```

Without text arguments the command reads standard input line by line. `--model` takes a bundled name or a path to a JSON model. `python -m banglatok` is the same command. Input and output are always UTF-8, including on Windows.

## Threads and processes

**Threads.** One tokenizer can be shared by any number of threads. Native calls release the GIL, so threads encode in parallel.

**Processes.** The Go runtime inside banglatok does not survive `fork()`. A process forked after its parent created a tokenizer (or called `normalize`) cannot use banglatok, and any call raises `RuntimeError` instead of hanging. Use the `spawn` or `forkserver` start method and create the tokenizer inside each worker:

```python
import multiprocessing as mp

from banglatok import Tokenizer

def init():
    global tok
    tok = Tokenizer.from_pretrained()

def encode(text):
    return tok.encode(text)

if __name__ == "__main__":
    with mp.get_context("spawn").Pool(4, initializer=init) as pool:
        ids = pool.map(encode, ["আমি বাংলা বলি", "ami bangla boli"])
```

`spawn` is the default on macOS and Windows, and `forkserver` on Linux from Python 3.14. On Linux with Python 3.13 or older the default is `fork`, so pass a context explicitly. Tokenizers cannot be pickled; pass texts and IDs between processes, not tokenizers.

## Your own models

`Tokenizer(path)` loads any model in banglatok's JSON format, for example one trained with the Go `bntok` command:

```sh
go run ./cmd/bntok train -algorithm bpe -vocab 384 \
  -input examples/training-records.jsonl -out models/toy-bpe.json
```

```python
tok = Tokenizer("models/toy-bpe.json")
```

[TRAINING_AND_RESEARCH.md](TRAINING_AND_RESEARCH.md) describes the corpus format, training options and the model contract.

## Guides

### Count tokens and stay within a budget

`count_tokens` measures text for a model trained with the selected banglatok vocabulary. These counts are not the token counts of GPT, Claude, BanglaBERT or other models with different tokenizers. This recipe budgets independently encoded documents; if you concatenate them, count the final text too because separators can change tokenization:

```python
from banglatok import Tokenizer

tok = Tokenizer.from_pretrained()
documents = ["আমি বাংলায় গান গাই।", "meetingটা ১০টায়, don't be late 🙂"]

sizes = [tok.count_tokens(text) for text in documents]
total = sum(sizes)

# Keep whole documents while they fit in 4,096 tokens.
budget, kept = 4096, []
for text, size in zip(documents, sizes):
    if size > budget:
        break
    kept.append(text)
    budget -= size
```

To cut one long text to a fixed number of tokens, use `tok(text, max_length=n, add_special_tokens=False)["input_ids"]`. Decoding a cut sequence can fail with `ValueError` if the cut splits a character that was stored as raw bytes; drop the trailing byte tokens or cut at a different point.

### Tokenize a dataset with 🤗 Datasets

The tokenizer's output is a plain dict of lists, which is exactly what `Dataset.map` expects:

Install the optional library with `uv pip install datasets` (or `pip install datasets`).

```python
from datasets import Dataset

from banglatok import Tokenizer

tok = Tokenizer.from_pretrained()
dataset = Dataset.from_dict({"text": ["আমি বাংলায় গান গাই।", "ami banglay gaan gai", "I sing in Bangla"]})

def tokenize(batch):
    return tok(batch["text"], max_length=512)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized[0]["input_ids"]     # [2, 4504, 35499, 32651, 37272, 1101, 3]
```

Keep `num_proc` unset for this recipe. A tokenizer cannot be pickled for worker processes; to parallelize, use a worker initializer that creates its own tokenizer as shown in [Threads and processes](#threads-and-processes).

### Feed a PyTorch DataLoader

Tokenize inside the collate function, so each batch is padded only to its own longest text:

Install PyTorch with `uv pip install torch` (or `pip install torch`); select a CUDA build using the [PyTorch installer](https://pytorch.org/get-started/locally/) when needed.

```python
import torch
from torch.utils.data import DataLoader

from banglatok import Tokenizer

tok = Tokenizer.from_pretrained()
texts = ["আমি বাংলায় গান গাই।", "ami banglay gaan gai", "I sing in Bangla", "meetingটা কখন?"]

def collate(batch):
    encoded = tok(batch, max_length=256, padding=True)
    return {name: torch.tensor(values) for name, values in encoded.items()}

loader = DataLoader(texts, batch_size=2, collate_fn=collate)
next(iter(loader))["input_ids"].shape     # torch.Size([2, 7])
```

With `num_workers > 0`, also pass `multiprocessing_context="spawn"` and create the tokenizer inside the workers, or tokenize the whole dataset up front as in the previous guide.

### Train a small language model from scratch

A model trained on banglatok IDs needs only the vocabulary size and the control-token IDs. This example builds a small GPT-2-style model with `transformers` and runs one training step:

Install these optional training dependencies with `uv pip install torch transformers` (or `pip install torch transformers`).

```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from banglatok import BOS_ID, EOS_ID, PAD_ID, Tokenizer

tok = Tokenizer.from_pretrained()
config = GPT2Config(
    vocab_size=tok.vocab_size,
    n_positions=512, n_embd=256, n_layer=4, n_head=4,
    bos_token_id=BOS_ID, eos_token_id=EOS_ID, pad_token_id=PAD_ID,
)
model = GPT2LMHeadModel(config)

batch = tok(["আমি বাংলায় গান গাই।", "ami banglay gaan gai"], max_length=512, padding=True)
input_ids = torch.tensor(batch["input_ids"])
attention_mask = torch.tensor(batch["attention_mask"])
labels = input_ids.masked_fill(attention_mask == 0, -100)   # no loss on padding

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
loss.backward()
optimizer.step()
```

For real training, tokenize the corpus once with the Datasets guide above and use `transformers.Trainer` or your own loop. To share the model, save the tokenizer next to it with `tok.save_huggingface(output_dir)` so `AutoTokenizer.from_pretrained(output_dir)` works.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| The install tries to build from source and fails | No wheel for your platform or Python version. Use CPython 3.10+ on a platform from [Install](#install), or install Go 1.24.9+ and a C compiler. |
| `RuntimeError: banglatok cannot run in a child process forked after the parent loaded it` | Use the `spawn` or `forkserver` start method. See [Threads and processes](#threads-and-processes). |
| `RuntimeError: tokenizer is closed` | The tokenizer was used after `close()` or after its `with` block ended. |
| `ValueError: invalid or reserved token ID 2` from `decode` | The IDs contain control tokens; pass `skip_special_tokens=True`. |
| `ValueError: token sequence is not valid UTF-8` from `decode` | The IDs end in the middle of a character stored as raw bytes, usually after truncation. |
| Garbled Bangla in a Windows terminal | Use Windows Terminal or another UTF-8 terminal; banglatok writes UTF-8. |

Report other problems at <https://github.com/iamaber/banglatok/issues>.

## API reference

Everything is importable from `banglatok`.

| Member | Description |
|---|---|
| `Tokenizer.from_pretrained(name="bn-bpe-58k")` | Load a bundled model, offline |
| `Tokenizer(path)` | Load a JSON model file |
| `encode(text)` / `encode_batch(texts)` | Token IDs, without control tokens |
| `decode(ids, skip_special_tokens=False)` / `decode_batch(batch, skip_special_tokens=False)` | Normalized text |
| `count_tokens(text)` | `len(encode(text))`, without building the list |
| `tokenize(text)` | Token strings |
| `tok(texts, add_special_tokens=True, max_length=None, padding=False)` | `{"input_ids", "attention_mask"}` |
| `id_to_token(id)`, `token_to_id(token)`, `get_vocab()` | Vocabulary lookups |
| `vocab_size`, `algorithm`, `closed` | Properties |
| `save_huggingface(directory)` | Write a Hugging Face tokenizer with identical IDs (BPE models) |
| `close()`, `with ... as tok:` | Release the native model |
| `normalize(text)` | NFC plus whitespace collapse and trim |
| `available_models()`, `DEFAULT_MODEL` | Bundled model names |
| `PAD_ID`, `UNK_ID`, `BOS_ID`, `EOS_ID`, `SPECIAL_TOKENS` | Control-token IDs `0`–`3` and their strings |
| `__version__` | Installed version |

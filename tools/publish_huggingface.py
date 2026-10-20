"""Export the bundled models to Hugging Face format and upload them to the Hub.

Requires ``huggingface_hub`` and a login (``hf auth login``). Use ``--dry-run``
to only write the folders.
"""

import argparse
import tempfile
from pathlib import Path

from banglatok import Tokenizer, __version__, available_models

CARD = """---
license: cc-by-sa-4.0
language:
- bn
- en
library_name: transformers
datasets:
- iamaber/banglatok-training-corpus
tags:
- tokenizer
- bangla
- bengali
- banglish
- code-mixed
- bpe
---

# banglatok `{name}`

`{name}` is a banglatok BPE tokenizer for Bangla, English and Banglish
(romanized Bengali). It has {vocab:,} token IDs. This repository contains the
tokenizer, not a language model.

## Use with Transformers

Install Transformers in your Python environment:

```sh
uv pip install transformers
```

You can use `pip install transformers` instead.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo}")
encoded = tok("meetingটা ১০টায়, don't be late 🙂")
text = tok.decode(encoded["input_ids"], skip_special_tokens=True)
```

## Use with banglatok

To use the native tokenizer, add banglatok to your project with `uv add banglatok`,
or install it with `pip install banglatok`.

```python
from banglatok import Tokenizer

tok = Tokenizer.from_pretrained("{name}")
ids = tok.encode("আমি বাংলা বলি")
print(tok.decode(ids))
```

Both versions use the same token IDs. Text is normalized to NFC, with whitespace
collapsed and trimmed. Unseen characters use UTF-8 byte tokens. The Transformers
tokenizer adds `<s>` and `</s>` by default; `banglatok.encode()` does not.

## Training data

The actual normalized training corpus is public at
[iamaber/banglatok-training-corpus](https://huggingface.co/datasets/iamaber/banglatok-training-corpus):
563,476 training records, 14,779 validation records and 16,712 test records.
Vocabulary and merge learning used only the training split, not validation or test.
Both bundled models use this corpus.

Sources are [Bengali Wikipedia, `20231101.bn`](https://huggingface.co/datasets/wikimedia/wikipedia)
(CC BY-SA 3.0 / GFDL), [Tatoeba via ManyThings](https://www.manythings.org/anki/)
(Bengali–English and the English side of French–English; CC BY 2.0 France, with
per-row contributor attribution), and
[BanglaTLit](https://github.com/farhanishmam/BanglaTLit) (upstream MIT notice retained).
The dataset preserves per-record source attribution and source notices; consult
its card and [corpus documentation](https://github.com/iamaber/banglatok/blob/main/data/CORPUS.md)
for source-specific terms. The corpus has mixed source licenses, not a blanket
CC BY-SA 4.0 license. BanglaTLit's MIT repository metadata does not establish
every original commenter's rights. No comprehensive rights/privacy review has
been performed.

Preparation applies NFC and collapsed/trimmed whitespace, retains 12–2,000
normalized characters, and filters replacement-character corruption, recognizable
emails, Bangladeshi phone patterns and simple credential assignments. These are
heuristics, not complete PII removal. Case, punctuation, emoji, joiners and
spelling variation are preserved. Paired records/article paragraphs stay grouped;
canonical and approximate long-text deduplication reduce split leakage, with
official held-out membership taking precedence. No synthetic transliteration is
generated.

## Benchmarks

This table compares **both** bundled variants, not just the model in this
repository. Measured 13 November 2026 with banglatok 0.2.0 (these model IDs are
unchanged), Python 3.12, on 10,000 unseen-source
[BanglishRev reviews](https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset).
All candidates receive the same NFC/whitespace-normalized input. Exact round trip
means `decode(encode(text)) == text` on that input; token counts exclude special
tokens and words are whitespace-separated.

Speed is MB/s, **single / batch of 64**, with five-pass medians on an Apple M5
(macOS) and an AMD EPYC 9V74 Linux x86_64 VM. Every candidate is single-threaded:
`TOKENIZERS_PARALLELISM=false`, `RAYON_NUM_THREADS=1`, and `num_threads=1` for
SentencePiece and tiktoken batches.

SentencePiece baselines below use the same training data and match each bundled
model's vocabulary size.

| Tokenizer | Vocab | Tokens/word | Exact round trip | MB/s, Apple M5 | MB/s, Linux x86_64 |
|---|---:|---:|---:|---:|---:|
| **banglatok `bn-bpe-58k`** | 57,925 | **1.379** | **100.00%** | 58.5 / 71.6 | 24.4 / 29.7 |
| **banglatok `bn-bpe-28k`** | 27,760 | 1.587 | **100.00%** | 61.6 / 72.9 | 26.1 / 33.6 |
| SentencePiece Unigram 58k | 57,925 | 1.410 | 100.00% | 42.4 / 41.7 | 20.5 / 20.9 |
| SentencePiece BPE 58k | 57,925 | 1.407 | 100.00% | 17.7 / 17.4 | 9.5 / 9.8 |
| SentencePiece Unigram 28k | 27,760 | 1.572 | 100.00% | 46.4 / 43.2 | 22.8 / 23.4 |
| SentencePiece BPE 28k | 27,760 | 1.578 | 100.00% | 16.6 / 16.4 | 9.8 / 9.8 |
| BNLP `bn_spm` (SentencePiece) | 50,000 | 1.825 | 80.14% | 40.1 / 39.2 | 19.3 / 19.8 |
| tiktoken `o200k_base` (GPT-4o) | 200,019 | 1.780 | 100.00% | 37.3 / 12.1 | 19.4 / 5.4 |
| BanglaBERT (csebuetnlp) | 32,000 | 1.771 | 51.10% | 9.1 / 9.2 | 4.8 / 4.6 |
| MuRIL (Google) | 197,258 | 1.498 | 47.94% | 8.8 / 8.9 | 4.6 / 4.4 |
| XLM-R / multilingual-E5 / BGE-M3 | 250,002 | 1.740 | 98.16% | 6.5 / 6.4 | 3.7 / 3.5 |
| mBERT cased | 119,547 | 2.122 | 47.97% | 8.1 / 8.1 | 4.2 / 4.1 |
| Qwen3 | 151,669 | 3.588 | 100.00% | 6.0 / 6.0 | 3.3 / 3.1 |

The **58k** variant produces about 22% fewer tokens than BanglaBERT or `o200k_base`
and 21% fewer than XLM-R. Same-data, same-size SentencePiece is within about 2%
of 58k; **28k** is about 1% longer than its SentencePiece baselines. The advantage
over popular tokenizers largely reflects training data and vocabulary size,
not a better compression algorithm.

Across these two CPUs and single/batch modes, banglatok is 1.1–1.7× faster than
same-size SentencePiece Unigram, 2.6–4.4× than SentencePiece BPE, 1.3–5.9× than
tiktoken and 5–12× than Hugging Face `tokenizers`. Ratios depend on CPU and batch
mode; the Linux VM is shared. Hugging Face multithreaded throughput is unmeasured.
Shorter sequences do not establish downstream model accuracy, and using these
vocabularies with an existing model requires training/adapting its embeddings.

See the [full report](https://github.com/iamaber/banglatok/blob/main/benchmarks/POPULAR_COMPARISON.md)
and raw JSONs for
[Apple M5](https://github.com/iamaber/banglatok/blob/main/benchmarks/results/popular-comparison.json),
[Linux](https://github.com/iamaber/banglatok/blob/main/benchmarks/results/popular-comparison-linux-x86_64.json)
and [same-data baselines](https://github.com/iamaber/banglatok/blob/main/benchmarks/results/sentencepiece-baselines.json).

## Guides

The [usage guide](USAGE.md) covers installation, Python and command-line usage,
token counting, dataset preprocessing, PyTorch batches, model training and
troubleshooting.

For the training data and model limitations, see
[MODELS.md](https://github.com/iamaber/banglatok/blob/main/docs/MODELS.md).
Benchmark results are in
[POPULAR_COMPARISON.md](https://github.com/iamaber/banglatok/blob/main/benchmarks/POPULAR_COMPARISON.md).

## License

The tokenizer files are licensed under CC BY-SA 4.0; see
[MODEL_LICENSE.md](https://github.com/iamaber/banglatok/blob/main/MODEL_LICENSE.md).
This does not relicense the training corpus: its source-specific terms and
attribution requirements remain applicable, as described above.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="iamaber")
    parser.add_argument("--dry-run", type=Path, help="write folders here instead of uploading")
    args = parser.parse_args()
    guide = (Path(__file__).resolve().parents[1] / "docs" / "USAGE.md").read_text(encoding="utf-8")
    for name in available_models():
        repo = f"{args.namespace}/banglatok-{name}"
        with tempfile.TemporaryDirectory() as temporary:
            folder = args.dry_run / name if args.dry_run else Path(temporary)
            with Tokenizer.from_pretrained(name) as tok:
                tok.save_huggingface(folder)
                card = CARD.format(name=name, vocab=tok.vocab_size, repo=repo)
            (folder / "README.md").write_text(card, encoding="utf-8")
            (folder / "USAGE.md").write_text(guide, encoding="utf-8")
            if args.dry_run:
                print(f"wrote {folder}")
                continue
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo, repo_type="model", exist_ok=True)
            api.upload_folder(
                repo_id=repo,
                folder_path=folder,
                commit_message=f"banglatok {__version__}: {name} tokenizer",
            )
            print(f"uploaded https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()

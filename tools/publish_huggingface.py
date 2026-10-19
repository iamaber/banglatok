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

## Guides

The [usage guide](USAGE.md) covers installation, Python and command-line usage,
token counting, dataset preprocessing, PyTorch batches, model training and
troubleshooting.

For the training data and model limitations, see
[MODELS.md](https://github.com/iamaber/banglatok/blob/main/docs/MODELS.md).
Benchmark results are in
[POPULAR_COMPARISON.md](https://github.com/iamaber/banglatok/blob/main/benchmarks/POPULAR_COMPARISON.md).

## License

CC BY-SA 4.0, because the vocabulary was learned from Bengali Wikipedia (CC BY-SA 3.0 / GFDL),
Tatoeba (CC BY 2.0) and BanglaTLit (MIT) text. See
[MODEL_LICENSE.md](https://github.com/iamaber/banglatok/blob/main/MODEL_LICENSE.md).
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

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

Fast, lossless subword tokenizers for Bangla, English and Banglish (romanized Bengali),
including code-mixed text.

This is banglatok's `{name}` BPE model. Vocabulary: {vocab:,} IDs.
This repository holds only the tokenizer; there are no model weights.

It is the Hugging Face export of `{name}` from
[banglatok](https://github.com/iamaber/banglatok) {version}, and gives exactly the same IDs.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo}")
tok("meetingটা ১০টায়, don't be late 🙂")
```

Or use banglatok, the native package with no Python dependencies. We recommend
[uv](https://docs.astral.sh/uv/); pip works too:

```sh
uv add banglatok       # in a uv project
# or: pip install banglatok
```

```python
from banglatok import Tokenizer

tok = Tokenizer.from_pretrained("{name}")
```

- Lossless: decoding returns the input after NFC and whitespace normalization.
  There is no `<unk>` output; unseen characters fall back to UTF-8 bytes.
- `<s>` and `</s>` wrap each sequence when special tokens are added; `<pad>` is ID 0.
  Literal `<s>` text in the input stays ordinary text.
- Training data, build steps and limitations:
  [docs/MODELS.md](https://github.com/iamaber/banglatok/blob/main/docs/MODELS.md).
  Benchmarks: [benchmarks/POPULAR_COMPARISON.md](https://github.com/iamaber/banglatok/blob/main/benchmarks/POPULAR_COMPARISON.md).
- [User guide](https://github.com/iamaber/banglatok/blob/main/docs/USAGE.md):
  installation, Python API, CLI, token budgets, datasets and model training.

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
    for name in available_models():
        repo = f"{args.namespace}/banglatok-{name}"
        with tempfile.TemporaryDirectory() as temporary:
            folder = args.dry_run / name if args.dry_run else Path(temporary)
            with Tokenizer.from_pretrained(name) as tok:
                tok.save_huggingface(folder)
                card = CARD.format(name=name, vocab=tok.vocab_size, repo=repo, version=__version__)
            (folder / "README.md").write_text(card, encoding="utf-8")
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

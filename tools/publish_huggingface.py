"""Export bundled or explicit versioned native models to Hugging Face format.

Requires ``huggingface_hub`` and a login (``hf auth login``) for uploads.
Use ``--dry-run DIR`` to write folders without any Hub access. Without ``--model``,
the original bundled v1 models and cards are exported as before. Repeat
``--model PATH`` for custom models; stems such as ``bn-bpe-28k-v2`` become distinct
repository IDs. Custom uploads create new repositories and never overwrite one.
Use ``--training PATH`` for the exact training JSONL and repeat ``--benchmark PATH``
for measured ``benchmarks/compare_popular.py`` JSON reports.
"""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from banglatok import Tokenizer, __version__

V1_MODELS = ("bn-bpe-28k", "bn-bpe-58k")

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


CUSTOM_CARD = """---
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
- corpus-v2
---

# banglatok `{name}`

`{name}` is a native BPE tokenizer for Bangla, English and Banglish
(romanized Bengali), with **{vocab:,} token IDs**. It is not a language model.
This versioned artifact is separate from the immutable v1 models; its token IDs
are not interchangeable with theirs.

## Usage

Install with `uv pip install transformers` (recommended), or `pip install transformers`:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo}")
encoded = tok("meetingটা ১০টায়, don't be late 🙂")
text = tok.decode(encoded["input_ids"], skip_special_tokens=True)
```

For the native API, use `uv pip install banglatok` or `pip install banglatok`,
download the native model file from this repository, and load its local path:

```sh
curl -fL https://huggingface.co/{repo}/resolve/main/{name}.json -o {name}.json
```

```python
from banglatok import Tokenizer

with Tokenizer("{name}.json") as tok:
    ids = tok.encode("আমি বাংলা বলি")
    print(tok.decode(ids))
```

Both exports use the same IDs. Input is normalized to NFC with collapsed,
trimmed whitespace; unseen characters use UTF-8 byte tokens. Transformers adds
`<s>` and `</s>` by default; native `encode()` does not. Changing a tokenizer
requires training or adapting an existing language model's embeddings.
The copied [usage guide](USAGE.md) covers the broader API; examples naming the
bundled v1 IDs there still refer to v1, not this local model.

## Training data and provenance

{training_data}

## Evaluation and limitations

{benchmarks}

The preregistered [corpus-v2 protocol](https://github.com/iamaber/banglatok/blob/main/research/reports/CORPUS_V2_PROTOCOL.md)
defines growth/audit gates, validation-only selection, vocabulary budgets and
held-out testing. Publishing a candidate does not assert that it passed promotion
gates or improved over v1. Comparisons depend on the actual vocabulary, training
mixture, corpus and machine; tokenizer metrics do not establish downstream
accuracy. Read the measured reports and selection record before choosing a model.

## License

Tokenizer files are licensed under CC BY-SA 4.0; see
[MODEL_LICENSE.md](https://github.com/iamaber/banglatok/blob/main/MODEL_LICENSE.md).
This does not relicense source data or remove its attribution requirements.
"""

PUBLIC_TRAINING_DATA = """Dataset: [iamaber/banglatok-training-corpus, configuration `v2`](https://huggingface.co/datasets/iamaber/banglatok-training-corpus/tree/main/v2).
The expanded data and its manifests live under `v2/`; the root v1 data is unchanged.
Use the dataset's `v2` configuration, not its original default configuration.

- Data version: `v2`.
- Native model file: [`{name}.json`]({name}.json).
- Native model SHA-256: `{model_sha256}`.
- Exact training JSONL SHA-256: `{training_sha256}`.
- Actual vocabulary size: {vocab:,}, including special and byte token IDs.
- Exported with banglatok {package_version}.
- Machine-readable export provenance: [model_provenance.json](model_provenance.json).

Vocabulary learning, space fusion and caches use training text only, not held-out
splits. The v1 validation/test files remain frozen; expanded sources have separate
extra-validation/extra-test splits. Consult the dataset's versioned manifests,
source locks, attribution and audit for exact composition, counts and checksums.

Sources combine the original Wikipedia, ManyThings/Tatoeba and BanglaTLit data
with expanded Wikimedia material. Source-specific CC BY, CC BY-SA, GFDL and
upstream MIT notices remain applicable; the corpus is not uniformly licensed
under the tokenizer's license. BanglaTLit's repository notice does not establish
every original commenter's rights. No comprehensive rights/privacy review is
claimed.

Preparation uses normalization, corruption/PII/repetition/boilerplate heuristics,
canonical and approximate deduplication, and group/split exclusions. These do not
guarantee complete PII removal or absence of near duplicates. Language buckets
are source labels or script proxies, not independently verified annotations.
No synthetic transliteration is generated, and BanglishRev evaluation sources
are excluded from acquisition. Larger size alone is not evidence of better
linguistic quality."""


def load_restricted_summary(path: Path, training_hash: str, marker: Path | None) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != 1
        or summary.get("classification") != "restricted-training-data"
        or summary.get("raw_redistribution_allowed") is not False
        or summary.get("training_sha256") != training_hash
        or not isinstance(summary.get("public_baseline_training_sha256"), str)
    ):
        raise ValueError("restricted summary does not match the training corpus")
    sources = summary.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("restricted summary requires aggregate source descriptions")
    for source in sources:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("description"), str)
            or not source["description"].strip()
            or source.get("rights_basis")
            not in {
                "direct-consent",
                "first-party-terms",
                "written-contract",
            }
            or not isinstance(source.get("records"), int)
            or source["records"] < 0
            or not isinstance(source.get("buckets"), dict)
        ):
            raise ValueError("restricted summary has an invalid aggregate source")
    if marker is not None:
        marker_data = json.loads(marker.read_text(encoding="utf-8"))
        if marker_data.get("manifest_sha256") != summary.get("manifest_sha256"):
            raise ValueError("restricted marker and summary manifest hashes differ")
    return summary


def restricted_training_data(summary: dict) -> str:
    def buckets(source: dict) -> str:
        return ", ".join(
            f"`{name}`: {count:,}" for name, count in sorted(source["buckets"].items())
        )

    rows = "\n".join(
        f"| {source['description']} | `{source['rights_basis']}` | "
        f"{source['records']:,} | {buckets(source)} |"
        for source in summary["sources"]
    )
    return f"""The public baseline is
[iamaber/banglatok-training-corpus, configuration `v2`](https://huggingface.co/datasets/iamaber/banglatok-training-corpus/tree/main/v2).
Training additionally used private, training-only natural text. Its agreements
permit transformation, commercial training and distribution of resulting model
artifacts, but not redistribution of the retained rows. The raw text, agreements
and private manifests are therefore not included in this repository.

| Restricted source category | Rights basis | Accepted training records | Buckets |
|---|---|---:|---|
{rows}

- Data version: `v2-plus-restricted-v1`.
- Native model file: [`{{name}}.json`]({{name}}.json).
- Native model SHA-256: `{{model_sha256}}`.
- Exact combined training JSONL SHA-256: `{{training_sha256}}`.
- Public baseline training SHA-256: `{summary["public_baseline_training_sha256"]}`.
- Actual vocabulary size: {{vocab:,}}, including special and byte token IDs.
- Exported with banglatok {{package_version}}.
- Machine-readable export provenance: [model_provenance.json](model_provenance.json).

The restricted preparation path applies the same normalization, quality,
canonical/near-deduplication and holdout-protection pipeline as public v2.
Rights evidence is checksum-verified locally. Privacy review and manual review
are required, but neither they nor automated PII heuristics guarantee complete
removal. BanglishRev and other benchmark surfaces remain evaluation-only."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_card(
    reports: list[tuple[Path, dict]], model: Path, model_hash: str, vocab_size: int
) -> str:
    if not reports:
        return (
            "No published comparison is attached to this artifact. Historical v1 "
            "benchmark results do not measure this model, and no improvement is claimed."
        )
    candidate = f"banglatok {model.stem}"
    sections = []
    for index, (path, report) in enumerate(reports, 1):
        metadata = report["candidates"].get(candidate)
        expected_source = f"{model.name} sha256 {model_hash}"
        if (
            metadata is None
            or metadata["source"] != expected_source
            or metadata["vocab_size"] != vocab_size
        ):
            raise ValueError(
                f"{path}: benchmark candidate does not match {model.name} hash/vocabulary"
            )
        corpora = report["corpora"]
        if not corpora or any(candidate not in c["candidates"] for c in corpora.values()):
            raise ValueError(f"{path}: benchmark does not measure {candidate} in every corpus")
        sections.extend(
            [
                f"### Measured report {index}",
                "",
                f"Raw results: [benchmark-{index}.json](benchmark-{index}.json); "
                f"SHA-256 `{sha256(path)}`.",
                f"Recorded {report['created_utc']}; CPU {report['cpu_model']}; "
                f"platform {report['platform']}; Python {report['python']}.",
                f"{report['repeats']} timing repeats; batch size {report['batch_size']}. "
                "The compare_popular.py harness uses NFC/whitespace-normalized input, "
                "excludes special tokens from counts, and requires single-thread settings.",
                f"Package versions: `{json.dumps(report['packages'], sort_keys=True)}`.",
                f"Candidate stress round trips: {metadata['stress_round_trips']}"
                f"/{metadata['stress_cases']}.",
            ]
        )
        for label, corpus in corpora.items():
            sections.extend(
                [
                    "",
                    f"#### {label} ({corpus['records']:,} records)",
                    "",
                    f"Corpus SHA-256: `{corpus['source_sha256']}`; "
                    f"ordered normalized input SHA-256: `{corpus['ordered_text_sha256']}`.",
                    "",
                    "| Tokenizer | Vocab | Tokens/word | Exact round trip | Texts with UNK "
                    "| Single MB/s | Batch MB/s |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for name, row in corpus["candidates"].items():
                quality, speed = row["quality"]["all"], row["speed"]
                vocabulary = report["candidates"][name]["vocab_size"]
                sections.append(
                    f"| {name} | {vocabulary:,} | {quality['tokens_per_word']:.3f}"
                    f" | {quality['round_trip_percent']:.2f}%"
                    f" | {quality['texts_with_unknowns']:,}"
                    f" | {speed['single_MB_per_second']:.1f}"
                    f" | {speed['batch_MB_per_second']:.1f} |"
                )
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="iamaber")
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        default=[],
        help="native JSON model with a versioned stem; repeat for multiple models",
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=Path("data/processed/expanded-v2/train.jsonl"),
        help="exact corpus-v2 training JSONL used by custom models",
    )
    parser.add_argument(
        "--restricted-summary",
        type=Path,
        help="checksum-bound aggregate summary for training-only restricted data",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        action="append",
        default=[],
        help="measured compare_popular.py JSON report for custom models; repeatable",
    )
    parser.add_argument("--dry-run", type=Path, help="write folders here instead of uploading")
    args = parser.parse_args()
    if args.benchmark and not args.model:
        parser.error("--benchmark requires --model; bundled v1 cards remain unchanged")
    if args.restricted_summary and not args.model:
        parser.error("--restricted-summary requires --model")
    names = set()
    for model in args.model:
        if model.stem in V1_MODELS:
            parser.error(f"{model.stem} is an immutable v1 ID; use a new versioned stem")
        match = re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*-v([0-9]+)", model.stem)
        if not match or int(match[1]) < 2:
            parser.error(f"{model.stem}: custom model stem must end in -v2 or a later version")
        if model.stem in names:
            parser.error(f"duplicate custom model ID: {model.stem}")
        names.add(model.stem)
        if not model.is_file():
            parser.error(f"model file does not exist: {model}")
    if args.model and not args.training.is_file():
        parser.error(f"exact custom training file does not exist: {args.training}")
    training_hash = sha256(args.training) if args.model else None
    if args.model:
        assert training_hash is not None
    restricted_summary = None
    if args.model:
        marker = args.training.parent / ".restricted-corpus.json"
        summary_path = args.restricted_summary
        if marker.is_file() and summary_path is None:
            summary_path = args.training.parent / "restricted-summary.json"
        if marker.is_file() and (summary_path is None or not summary_path.is_file()):
            parser.error("restricted training data requires restricted-summary.json")
        if summary_path is not None:
            try:
                restricted_summary = load_restricted_summary(
                    summary_path,
                    training_hash,
                    marker if marker.is_file() else None,
                )
            except (OSError, ValueError, TypeError) as error:
                parser.error(f"invalid restricted training summary: {error}")
    training_data = (
        restricted_training_data(restricted_summary) if restricted_summary else PUBLIC_TRAINING_DATA
    )
    reports = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in args.benchmark]
    guide = (Path(__file__).resolve().parents[1] / "docs" / "USAGE.md").read_text(encoding="utf-8")
    models = [(path.stem, path) for path in args.model]
    if not models:
        models = [(name, None) for name in V1_MODELS]
    for name, model in models:
        repo = f"{args.namespace}/banglatok-{name}"
        with tempfile.TemporaryDirectory() as temporary:
            folder = args.dry_run / name if args.dry_run else Path(temporary)
            tokenizer = Tokenizer(model) if model else Tokenizer.from_pretrained(name)
            with tokenizer as tok:
                if model:
                    model_hash = sha256(model)
                    provenance = {
                        "model_file": model.name,
                        "model_sha256": model_hash,
                        "training_sha256": training_hash,
                        "vocab_size": tok.vocab_size,
                        "dataset": "iamaber/banglatok-training-corpus",
                        "dataset_config": ("v2-plus-restricted" if restricted_summary else "v2"),
                        "data_version": ("v2-plus-restricted-v1" if restricted_summary else "v2"),
                        "repository": repo,
                        "banglatok_version": __version__,
                    }
                    if restricted_summary:
                        provenance["restricted_training_summary"] = restricted_summary
                    sidecar = model.with_suffix(".provenance.json")
                    if sidecar.is_file():
                        training_provenance = json.loads(sidecar.read_text(encoding="utf-8"))
                        if training_provenance["training_sha256"] != training_hash:
                            parser.error(f"{sidecar}: training SHA-256 does not match --training")
                        provenance["training_provenance"] = training_provenance
                    try:
                        benchmarks = benchmark_card(reports, model, model_hash, tok.vocab_size)
                    except (KeyError, TypeError, ValueError) as error:
                        parser.error(f"invalid benchmark: {error}")
                    card = CUSTOM_CARD.format(
                        name=name,
                        vocab=tok.vocab_size,
                        repo=repo,
                        model_sha256=model_hash,
                        training_sha256=training_hash,
                        package_version=__version__,
                        benchmarks=benchmarks,
                        training_data=training_data.format(
                            name=name,
                            model_sha256=model_hash,
                            training_sha256=training_hash,
                            vocab=tok.vocab_size,
                            package_version=__version__,
                        ),
                    )
                else:
                    card = CARD.format(name=name, vocab=tok.vocab_size, repo=repo)
                tok.save_huggingface(folder)
            if model:
                shutil.copyfile(model, folder / model.name)
                (folder / "model_provenance.json").write_text(
                    json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                for index, (path, _) in enumerate(reports, 1):
                    shutil.copyfile(path, folder / f"benchmark-{index}.json")
            (folder / "README.md").write_text(card, encoding="utf-8")
            (folder / "USAGE.md").write_text(guide, encoding="utf-8")
            if args.dry_run:
                print(f"wrote {folder}")
                continue
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo, repo_type="model", exist_ok=model is None)
            api.upload_folder(
                repo_id=repo,
                folder_path=folder,
                commit_message=f"banglatok {__version__}: {name} tokenizer",
            )
            print(f"uploaded https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()

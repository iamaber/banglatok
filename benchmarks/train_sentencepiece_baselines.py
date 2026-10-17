"""Train SentencePiece baselines with the same ID budgets as the bundled models.

Each model is trained on the same NFC/whitespace-normalized training split as
the bundled models, with byte fallback, so vocabulary size is the only thing
that differs from the bundled tokenizers in the comparison. ``vocab_size`` is
the total ID space, matching ``Tokenizer.vocab_size`` of the bundled models.
"""

import argparse
import hashlib
import json
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

# Total ID space of bn-bpe-28k and bn-bpe-58k.
SIZES = [27760, 57925]
MODEL_TYPES = ["unigram", "bpe"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_training_text(source: Path, target: Path) -> int:
    """Write one normalized training record per line; return the longest in bytes."""
    longest = 0
    with source.open(encoding="utf-8") as records, target.open("w", encoding="utf-8") as stream:
        for line in records:
            record = json.loads(line)
            if record["split"] != "train":
                raise ValueError("non-training record in training input")
            text = " ".join(unicodedata.normalize("NFC", record["text"]).split())
            longest = max(longest, len(text.encode()))
            stream.write(text + "\n")
    return longest


def train(job: tuple[Path, Path, dict[str, object]]) -> tuple[str, float]:
    import sentencepiece as spm

    training, prefix, settings = job
    started = perf_counter()
    spm.SentencePieceTrainer.train(input=str(training), model_prefix=str(prefix), **settings)
    return prefix.name, perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("models/comparison"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("benchmarks/results/sentencepiece-baselines.json")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    text = args.output / "train.txt"
    longest = write_training_text(args.training, text)
    settings = {
        "character_coverage": 1.0,
        "byte_fallback": True,
        "normalization_rule_name": "identity",
        "num_threads": 1,
        "shuffle_input_sentence": False,
        "input_sentence_size": 0,
        "max_sentence_length": longest + 1,
        "max_sentencepiece_length": 12,
        "hard_vocab_limit": False,
    }
    jobs = [
        (
            text,
            args.output / f"sentencepiece-{model_type}-{size}",
            settings | {"model_type": model_type, "vocab_size": size},
        )
        for model_type in MODEL_TYPES
        for size in SIZES
    ]
    with ProcessPoolExecutor(len(jobs)) as pool:
        seconds = dict(pool.map(train, jobs))
    metadata = {
        "training_source_sha256": sha256(args.training),
        "normalized_training_sha256": sha256(text),
        "sentencepiece_version": version("sentencepiece"),
        "settings": settings,
        "models": {
            prefix.name: {
                "model_type": job_settings["model_type"],
                "vocab_size": job_settings["vocab_size"],
                "training_seconds": seconds[prefix.name],
                "sha256": sha256(prefix.with_suffix(".model")),
            }
            for _, prefix, job_settings in jobs
        },
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

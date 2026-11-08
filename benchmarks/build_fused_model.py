"""Build a version-2 fused BPE model from a base BPE model and the training split.

Fusion IDs represent a space followed by one base ID. Candidates are the base
IDs that most often start a word (follow a space) in the training split; ties
break by ID. The training-word cache lists the most frequent training words
and only speeds up encoding: it never changes IDs.
"""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from banglatok import Tokenizer, normalize

SPACE = 36


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_texts(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["split"] != "train":
                raise ValueError("fusion and cache must use training records only")
            yield normalize(record["text"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--training", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--budget", type=int, required=True, help="total vocabulary size")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--cache-words", type=int, default=64000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    if base["algorithm"] != "bpe" or base.get("space_fusion"):
        raise ValueError("base must be an unfused BPE model")
    starts: Counter[int] = Counter()
    words: Counter[str] = Counter()
    with Tokenizer(args.base) as tokenizer:
        base_vocab = tokenizer.vocab_size
        if args.budget <= base_vocab:
            parser.error("budget must exceed the base vocabulary")
        texts = training_texts(args.training)
        while batch := list(islice(texts, 1024)):
            for ids in tokenizer.encode_batch(batch):
                starts.update(b for a, b in zip(ids, ids[1:], strict=False) if a == SPACE)
            for text in batch:
                words.update(w for w in text.split() if len(w.encode()) <= 1024)
    ranked = sorted(
        (token for token, count in starts.items() if count >= args.min_count),
        key=lambda token: (-starts[token], token),
    )
    fusion = ranked[: args.budget - base_vocab]
    pieces = {piece["text"] for piece in base["pieces"]}
    cache = [w for w in sorted(words, key=lambda w: (-words[w], w)) if w not in pieces]
    model = {
        **base,
        "version": 2,
        "space_fusion": fusion,
        "cache_words": cache[: args.cache_words],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False) + "\n", encoding="utf-8")
    with Tokenizer(args.output) as fused:
        vocabulary = fused.vocab_size
    provenance = {
        "artifact": args.output.name,
        "sha256": sha256(args.output),
        "base": args.base.name,
        "base_sha256": sha256(args.base),
        "training_sha256": sha256(args.training),
        "vocabulary": vocabulary,
        "fusion_ids": len(fusion),
        "fusion_selection": f"word-initial base IDs by training frequency, count >= "
        f"{args.min_count}, ID tie break",
        "cache_words": len(model["cache_words"]),
        "cache_selection": "training word frequency descending, lexical tie break, "
        "excluding learned pieces, at most 1024 UTF-8 bytes",
    }
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()

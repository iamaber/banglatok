"""Compare the bundled tokenizer with tokenizers people already use for Bangla.

Two corpora: the frozen in-domain validation split, and an unseen-source
sample of natural e-commerce reviews (BanglishRev) in Bengali script, Latin
script (English and Banglish) and mixed script. Every candidate gets the same
NFC/whitespace-normalized text. Timings are single-threaded and include each
API's own preprocessing and conversion to Python ID lists.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from banglatok import DEFAULT_MODEL, UNK_ID, Tokenizer, available_models, normalize

SEED = 20261110

# Hugging Face models popular for Bangla; mE5 and BGE-M3 share XLM-R's tokenizer.
HF_MODELS = [
    ("BanglaBERT (csebuetnlp)", "csebuetnlp/banglabert"),
    ("Bangla-BERT (sagorsarker)", "sagorsarker/bangla-bert-base"),
    ("MuRIL (Google)", "google/muril-base-cased"),
    ("mBERT cased", "google-bert/bert-base-multilingual-cased"),
    ("XLM-R / mE5 / BGE-M3", "FacebookAI/xlm-roberta-base"),
    ("Qwen3", "Qwen/Qwen3-0.6B"),
]

STRESS_TEXTS = [
    "আমি বাংলায় গান গাই।",
    "ami banglay gaan gai!",
    "meetingটা ১০টায়, don't be late 🙂",
    "cafe\u0301 ＡＢＣ ①",
    "বাংলা\u200dভাষা বাংলা\u200cভাষা",
    "👩🏽‍💻 🫠 🚀",
    "<unk> <s> </s> <pad> ▁ <0x20>",
    "a\x00b",
    "  বাংলা\tEnglish\nBanglish  ",
    "",
    "中文 العربية हिन्दी",
    "https://example.com/a?x=1&y=2",
    "১২৩৪৫ 12345 3.14159",
    "কি? কী! বইগুলোতে",
    "Hello WORLD",
    "বাংলা" * 200,
]


@dataclass
class Candidate:
    name: str
    source: str
    vocab_size: int
    encode: Callable[[str], list[int]]
    batch: Callable[[list[str]], list[list[int]]]
    decode: Callable[[list[int]], str]
    unknown_id: int | None
    note: str
    byte_fallback_ids: range | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidates(
    cache_dir: Path,
    sentencepiece_models: list[Path],
    model_paths: list[Path],
    native_only: bool,
) -> list[Candidate]:
    result = []
    for name in sorted(available_models(), key=lambda n: n != DEFAULT_MODEL):
        ours = Tokenizer.from_pretrained(name)
        result.append(
            Candidate(
                f"banglatok {name}",
                "bundled",
                ours.vocab_size,
                ours.encode,
                ours.encode_batch,
                ours.decode,
                UNK_ID,
                "banglatok bundled model",
                byte_fallback_ids=range(4, 260),
            )
        )
    for path in model_paths:
        native = Tokenizer(path)
        result.append(
            Candidate(
                f"banglatok {path.stem}",
                f"{path.name} sha256 {sha256(path)}",
                native.vocab_size,
                native.encode,
                native.encode_batch,
                native.decode,
                UNK_ID,
                "candidate JSON model",
                byte_fallback_ids=range(4, 260),
            )
        )
    if native_only:
        return result
    import sentencepiece as spm

    for path in sentencepiece_models:
        sp = spm.SentencePieceProcessor(model_file=str(path))
        if path.name == "bnlp.model":
            label = "BNLP bn_spm (SentencePiece)"
        elif match := re.fullmatch(r"sentencepiece-(unigram|bpe)-(\d+)\.model", path.name):
            kind = "Unigram" if match[1] == "unigram" else "BPE"
            label = f"SentencePiece {kind} {int(match[2]):,}, same training data"
        else:
            label = f"SentencePiece {path.stem}"
        result.append(
            Candidate(
                label,
                f"{path.name} sha256 {sha256(path)}",
                sp.vocab_size(),
                lambda text, sp=sp: sp.encode(text, out_type=int),
                lambda texts, sp=sp: sp.encode(texts, out_type=int, num_threads=1),
                sp.decode,
                sp.unk_id(),
                "sentencepiece " + version("sentencepiece"),
            )
        )
    import tiktoken

    for name in ["o200k_base", "cl100k_base"]:
        enc = tiktoken.get_encoding(name)
        result.append(
            Candidate(
                f"tiktoken {name}",
                name,
                enc.n_vocab,
                enc.encode_ordinary,
                lambda texts, enc=enc: enc.encode_ordinary_batch(texts, num_threads=1),
                lambda ids, enc=enc: enc.decode(ids),
                None,
                "tiktoken " + version("tiktoken"),
            )
        )
    from transformers import AutoTokenizer

    for label, repository in HF_MODELS:
        hf = AutoTokenizer.from_pretrained(repository, cache_dir=str(cache_dir))
        backend = hf.backend_tokenizer
        result.append(
            Candidate(
                label,
                repository,
                backend.get_vocab_size(),
                lambda text, b=backend: b.encode(text, add_special_tokens=False).ids,
                lambda texts, b=backend: [
                    e.ids for e in b.encode_batch(texts, add_special_tokens=False)
                ],
                lambda ids, b=backend: b.decode(ids, skip_special_tokens=False),
                hf.unk_token_id,
                "tokenizers " + version("tokenizers"),
            )
        )
    return result


def quality(candidate: Candidate, texts: list[str]) -> dict[str, int | float | None]:
    encoded = candidate.batch(texts)
    lengths = sorted(map(len, encoded))
    tokens = sum(lengths)
    words = sum(len(text.split()) for text in texts)
    characters = sum(len(text) for text in texts)
    exact = sum(candidate.decode(ids) == text for ids, text in zip(encoded, texts, strict=True))
    unknown = candidate.unknown_id
    fallback = candidate.byte_fallback_ids
    fallback_tokens = (
        sum(token in fallback for ids in encoded for token in ids) if fallback is not None else None
    )
    return {
        "texts": len(texts),
        "tokens": tokens,
        "whitespace_words": words,
        "tokens_per_word": tokens / words if words else 0,
        "characters_per_token": characters / tokens if tokens else 0,
        "exact_round_trips": exact,
        "round_trip_percent": 100 * exact / len(texts) if texts else 0,
        "texts_with_unknowns": sum(unknown in ids for ids in encoded) if unknown is not None else 0,
        "byte_fallback_tokens": fallback_tokens,
        "byte_fallback_token_percent": (
            100 * fallback_tokens / tokens if fallback_tokens is not None and tokens else None
        ),
        "over_510_token_percent": 100 * sum(n > 510 for n in lengths) / len(texts),
        "sequence_p95": lengths[math.ceil(0.95 * len(lengths)) - 1] if lengths else 0,
    }


def timings(
    engines: list[Candidate], texts: list[str], repeats: int, batch_size: int
) -> dict[str, dict[str, object]]:
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    samples: dict[str, dict[str, list[float]]] = {
        e.name: {"single": [], "batch": []} for e in engines
    }
    operations = [(engine, mode) for engine in engines for mode in ["single", "batch"]]
    for engine in engines:
        engine.batch(texts[:batch_size])
        for text in texts[:batch_size]:
            engine.encode(text)
    rng = random.Random(SEED)
    for _ in range(repeats):
        rng.shuffle(operations)
        for engine, mode in operations:
            started = perf_counter()
            if mode == "single":
                for text in texts:
                    engine.encode(text)
            else:
                for batch in batches:
                    engine.batch(batch)
            samples[engine.name][mode].append(perf_counter() - started)
    size = sum(len(text.encode()) for text in texts)
    return {
        name: {
            f"{mode}_MB_per_second": size / 1e6 / statistics.median(values)
            for mode, values in modes.items()
        }
        | {f"{mode}_seconds": values for mode, values in modes.items()}
        for name, modes in samples.items()
    }


BENGALI = re.compile(r"[\u0980-\u09ff]")
LATIN = re.compile(r"[A-Za-z]")
REVIEW = re.compile(r'"Review Content": ("(?:[^"\\]|\\.)*")')


def script(text: str) -> str:
    bengali, latin = bool(BENGALI.search(text)), bool(LATIN.search(text))
    return "mixed script" if bengali and latin else "bengali script" if bengali else "latin script"


def load_reviews(path: Path, limit: int) -> list[tuple[str, str]]:
    """Unique reviews of 12-2,000 characters from a raw BanglishRev byte prefix."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    seen, reviews = set(), []
    for match in REVIEW.finditer(raw):
        text = normalize(json.loads(match.group(1)))
        if (
            12 <= len(text) <= 2000
            and text not in seen
            and (BENGALI.search(text) or LATIN.search(text))
        ):
            seen.add(text)
            reviews.append(text)
    random.Random(SEED).shuffle(reviews)
    return [(script(text), text) for text in reviews[:limit]]


def load_validation(path: Path) -> list[tuple[str, str]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    random.Random(SEED).shuffle(records)
    return [(record["bucket"], normalize(record["text"])) for record in records]


def evaluate(
    engines: list[Candidate], corpus: list[tuple[str, str]], repeats: int, batch: int
) -> dict:
    texts = [text for _, text in corpus]
    groups: dict[str, list[str]] = defaultdict(list)
    for bucket, text in corpus:
        groups[bucket].append(text)
    speed = timings(engines, texts, repeats, batch)
    return {
        "records": len(texts),
        "utf8_bytes": sum(len(t.encode()) for t in texts),
        "ordered_text_sha256": hashlib.sha256(json.dumps(texts).encode()).hexdigest(),
        "candidates": {
            engine.name: {
                "speed": speed[engine.name],
                "quality": {
                    name: quality(engine, group)
                    for name, group in [("all", texts), *sorted(groups.items())]
                },
            }
            for engine in engines
        },
    }


def markdown(result: dict) -> str:
    lines = []
    ours = next(iter(result["candidates"]))
    for corpus_name, corpus in result["corpora"].items():
        lines.append(f"\n### {corpus_name} ({corpus['records']:,} texts)\n")
        lines.append(
            "| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK"
            " | Single MB/s | Batch MB/s |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, row in corpus["candidates"].items():
            q, s = row["quality"]["all"], row["speed"]
            vocab = result["candidates"][name]["vocab_size"]
            lines.append(
                f"| {name} | {vocab:,} | {q['tokens_per_word']:.3f}"
                f" | {q['round_trip_percent']:.2f}% | {q['texts_with_unknowns']:,}"
                f" | {s['single_MB_per_second']:.1f}"
                f" | {s['batch_MB_per_second']:.1f} |"
            )
        buckets = [b for b in corpus["candidates"][ours]["quality"] if b != "all"]
        lines.append(f"\nTokens per word by bucket ({corpus_name}):\n")
        lines.append("| Tokenizer | " + " | ".join(buckets) + " |")
        lines.append("|---|" + "---:|" * len(buckets))
        for name, row in corpus["candidates"].items():
            cells = [f"{row['quality'][b]['tokens_per_word']:.3f}" for b in buckets]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=Path("data/processed/validation.jsonl"))
    parser.add_argument("--validation-label", default="In-domain validation")
    parser.add_argument("--heldout", type=Path, action="append", default=[])
    parser.add_argument(
        "--reviews", type=Path, default=Path("data/raw/banglishrev/reviews-v1-head64m.json")
    )
    parser.add_argument("--review-limit", type=int, default=10000)
    parser.add_argument("--review-offset", type=int, default=0)
    parser.add_argument("--sentencepiece", type=Path, action="append", default=[])
    parser.add_argument("--model", type=Path, action="append", default=[])
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--hf-cache", type=Path, default=Path("models/hf-cache"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-model", default=platform.processor())
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/popular-comparison.json")
    )
    args = parser.parse_args()
    if (
        os.environ.get("TOKENIZERS_PARALLELISM") != "false"
        or os.environ.get("RAYON_NUM_THREADS") != "1"
    ):
        parser.error(
            "set TOKENIZERS_PARALLELISM=false and RAYON_NUM_THREADS=1 for single-thread timing"
        )
    engines = candidates(args.hf_cache, args.sentencepiece, args.model, args.native_only)
    corpora = {}
    if args.validation.exists():
        corpora[args.validation_label] = (args.validation, load_validation(args.validation))
    for path in args.heldout:
        corpora[f"Held-out {path}"] = (path, load_validation(path))
    if args.reviews.exists():
        reviews = load_reviews(args.reviews, args.review_offset + args.review_limit)
        corpora["Unseen-source BanglishRev reviews"] = (
            args.reviews,
            reviews[args.review_offset :],
        )
    if not corpora:
        parser.error("no corpus found")
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": args.cpu_model,
        "packages": {
            name: version(name)
            for name in [
                "banglatok",
                "sentencepiece",
                "tiktoken",
                "tokenizers",
                "transformers",
            ]
        },
        "repeats": args.repeats,
        "batch_size": args.batch_size,
        "seed": SEED,
        "review_offset": args.review_offset,
        "harness_sha256": sha256(Path(__file__)),
        "candidates": {
            e.name: {
                "source": e.source,
                "vocab_size": e.vocab_size,
                "note": e.note,
                "stress_round_trips": sum(
                    e.decode(e.encode(normalize(t))) == normalize(t) for t in STRESS_TEXTS
                ),
                "stress_cases": len(STRESS_TEXTS),
            }
            for e in engines
        },
        "corpora": {
            name: {"source": str(path), "source_sha256": sha256(path)}
            | evaluate(engines, corpus, args.repeats, args.batch_size)
            for name, (path, corpus) in corpora.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(markdown(result))


if __name__ == "__main__":
    main()

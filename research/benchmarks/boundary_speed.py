"""Measure Python-to-Go boundary throughput of the installed banglatok package.

Each case is timed as the median of --repeats passes; a pass repeats the case
enough times to last about --pass-seconds. Throughput is UTF-8 input MB/s.

    python research/benchmarks/boundary_speed.py --corpus corpus.jsonl --json out.json
"""

import argparse
import inspect
import json
import os
import platform
import statistics
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from banglatok import Tokenizer


def load_texts(path: Path) -> list[str]:
    texts = []
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                texts.append(json.loads(line)["text"])
    return texts


def shapes(texts: list[str]) -> dict[str, list[str]]:
    words = " ".join(texts).split(" ")
    short = [" ".join(words[i : i + 4]) for i in range(0, len(words), 4)]
    long = [" ".join(texts[i : i + 16]) for i in range(0, len(texts), 16)]
    return {"short": short, "mixed": texts, "long": long}


def repeated(texts: list[str], count: int) -> list[str]:
    return [texts[i % len(texts)] for i in range(count)]


def chunks(texts: list[str], size: int) -> list[list[str]]:
    return [texts[i : i + size] for i in range(0, len(texts), size)]


def measure(operation: Callable[[], object], size: int, repeats: int, seconds: float) -> dict:
    operation()
    started = perf_counter()
    operation()
    loops = max(1, round(seconds / max(perf_counter() - started, 1e-9)))
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        for _ in range(loops):
            operation()
        samples.append((perf_counter() - started) / loops)
    median = statistics.median(samples)
    return {
        "MB_per_second": size / 1e6 / median,
        "median_seconds": median,
        "spread_percent": 100 * (max(samples) - min(samples)) / median,
        "passes": repeats,
        "loops_per_pass": loops,
    }


def cases(tok: Tokenizer, texts: dict[str, list[str]], threads: int | None) -> dict:
    batch_options = {}
    if threads is not None:
        batch_options["num_threads"] = threads
    result: dict[str, tuple[Callable[[], object], list[str]]] = {}
    for shape, items in texts.items():
        result[f"{shape}/encode"] = (lambda items=items: [tok.encode(t) for t in items], items)
        result[f"{shape}/count_tokens"] = (
            lambda items=items: [tok.count_tokens(t) for t in items],
            items,
        )
        groups = chunks(items, 64)
        result[f"{shape}/encode_batch_64"] = (
            lambda groups=groups: [tok.encode_batch(g, **batch_options) for g in groups],
            items,
        )
        result[f"{shape}/call_64_pad_trunc128"] = (
            lambda groups=groups: [
                tok(g, max_length=128, padding=True, **batch_options) for g in groups
            ],
            items,
        )
        result[f"{shape}/call_64_plain"] = (
            lambda groups=groups: [tok(g, **batch_options) for g in groups],
            items,
        )
    medium = repeated(texts["mixed"], 1_000)
    result["mixed/encode_batch_1k"] = (lambda: tok.encode_batch(medium, **batch_options), medium)
    large = repeated(texts["mixed"], 10_000)
    result["mixed/encode_batch_10k"] = (lambda: tok.encode_batch(large, **batch_options), large)
    result["mixed/call_10k_pad_trunc128"] = (
        lambda: tok(large, max_length=128, padding=True, **batch_options),
        large,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="JSONL with a text field")
    parser.add_argument("--model", default="bn-bpe-58k")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--pass-seconds", type=float, default=0.3)
    parser.add_argument("--threads", type=int, help="num_threads for batch calls, if supported")
    parser.add_argument("--only", help="comma-separated case-name substrings")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    texts = shapes(load_texts(args.corpus))
    with Tokenizer.from_pretrained(args.model) as tok:
        threads = args.threads
        if (
            threads is not None
            and "num_threads" not in inspect.signature(tok.encode_batch).parameters
        ):
            raise SystemExit("installed banglatok has no num_threads option")
        results = {}
        for name, (operation, items) in cases(tok, texts, threads).items():
            if args.only and not any(part in name for part in args.only.split(",")):
                continue
            size = sum(len(t.encode("utf-8")) for t in items)
            results[name] = measure(operation, size, args.repeats, args.pass_seconds)
            row = results[name]
            print(f"{name:34} {row['MB_per_second']:8.1f} MB/s  ±{row['spread_percent']:.1f}%")
    report = {
        "banglatok": version("banglatok"),
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs",
        "model": args.model,
        "corpus": str(args.corpus),
        "threads": threads,
        "text_bytes": {
            k: sum(len(t.encode("utf-8")) for t in v) / len(v) for k, v in texts.items()
        },
        "results": results,
    }
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

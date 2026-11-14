#!/usr/bin/env python3
"""Write a deterministic group-preserving subset of the audited corpus-v2 training split.

Each ``--cap SOURCE=ROWS`` keeps whole groups of that source, ranked by SHA-256 of the
group name, until adding the next group would exceed ROWS. Other sources are kept whole.
Rows keep their original order and bytes, so the subset adds no text and no leakage path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_cap(value: str) -> tuple[str, int]:
    source, _, rows = value.partition("=")
    if not source or not rows.isdigit():
        raise argparse.ArgumentTypeError("expected SOURCE=ROWS")
    return source, int(rows)


def kept_groups(training: Path, caps: dict[str, int]) -> dict[str, set[str]]:
    sizes: dict[str, Counter[str]] = defaultdict(Counter)
    with training.open("rb") as stream:
        for line in stream:
            record = json.loads(line)
            if record["source"] in caps:
                sizes[record["source"]][record["group"]] += 1
    kept: dict[str, set[str]] = {}
    for source, limit in caps.items():
        if source not in sizes:
            raise ValueError(f"capped source has no training rows: {source}")
        ranked = sorted(sizes[source], key=lambda g: (hashlib.sha256(g.encode()).digest(), g))
        total, chosen = 0, set()
        for group in ranked:
            if total + sizes[source][group] > limit:
                continue
            total += sizes[source][group]
            chosen.add(group)
        kept[source] = chosen
    return kept


def subset(training: Path, output: Path, caps: dict[str, int]) -> dict:
    if output.exists():
        raise ValueError(f"output exists: {output}")
    kept = kept_groups(training, caps)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[tuple[str, str]] = Counter()
    partial = output.with_name(output.name + ".part")
    with training.open("rb") as source, partial.open("wb") as target:
        for line in source:
            record = json.loads(line)
            if record["source"] in kept and record["group"] not in kept[record["source"]]:
                continue
            target.write(line)
            counts[(record["bucket"], record["source"])] += 1
    partial.replace(output)
    manifest = {
        "input": str(training),
        "input_sha256": sha256(training),
        "caps": caps,
        "selection": "whole groups by SHA-256(group) rank; original row order and bytes",
        "output_sha256": sha256(output),
        "rows": sum(counts.values()),
        "bucket_source_rows": {f"{b}/{s}": n for (b, s), n in sorted(counts.items())},
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training", type=Path, default=Path("data/processed/expanded-v2/train.jsonl")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap", type=parse_cap, action="append", required=True)
    args = parser.parse_args()
    result = subset(args.training, args.output, dict(args.cap))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

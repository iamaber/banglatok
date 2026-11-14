"""Apply the preregistered corpus-v2 promotion gates to validation benchmarks.

This consumes compare_popular.py JSON, never test scores. A rejected candidate
remains in the output with its failed gates; no passing candidate exits nonzero.
"""

import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL = Path("research/reports/CORPUS_V2_PROTOCOL.md")
BUCKET_LIMITS = {"bangla": 1.01, "english": 1.01, "banglish": 1.01, "code-mixed": 1.02}
BASELINES = ("banglatok bn-bpe-28k", "banglatok bn-bpe-58k")
FROZEN = "Frozen v1 validation"
EXTRA = "Held-out data/processed/expanded-v2/extra-validation.jsonl"


def assess(report: dict, frozen_name: str = FROZEN, extra_name: str = EXTRA) -> dict:
    """Return observed gates and the lowest-extra-token passing model per size."""
    corpora = {label: report["corpora"][label] for label in (frozen_name, extra_name)}
    baselines = sorted((report["candidates"][name]["vocab_size"], name) for name in BASELINES)
    outcomes, selected = {}, {}
    for name, metadata in sorted(report["candidates"].items()):
        if not name.startswith("banglatok ") or metadata["source"] == "bundled":
            continue
        # Fusion can fall short of its budget; the candidate then competes with the
        # smallest baseline at least as large, so extra vocabulary never wins on size.
        tier = next(((v, b) for v, b in baselines if metadata["vocab_size"] <= v), None)
        if tier is None:
            outcomes[name] = {"passes": False, "gates": {"same_size_baseline": False}}
            continue
        size, baseline = tier
        candidate = {key: corpus["candidates"][name] for key, corpus in corpora.items()}
        original = {key: corpus["candidates"][baseline] for key, corpus in corpora.items()}
        gates = {
            "same_size_baseline": True,
            "normalized_round_trips_and_zero_unknowns": all(
                score["quality"]["all"]["texts"] > 0
                and score["quality"]["all"]["exact_round_trips"] == score["quality"]["all"]["texts"]
                and score["quality"]["all"]["texts_with_unknowns"] == 0
                for score in candidate.values()
            ),
            "stress_round_trips": metadata["stress_cases"] > 0
            and metadata["stress_round_trips"] == metadata["stress_cases"],
            "frozen_aggregate_no_regression": candidate[frozen_name]["quality"]["all"]["tokens"]
            <= original[frozen_name]["quality"]["all"]["tokens"],
        }
        ratios = {}
        for bucket, limit in BUCKET_LIMITS.items():
            old = original[frozen_name]["quality"][bucket]["tokens_per_word"]
            new = candidate[frozen_name]["quality"][bucket]["tokens_per_word"]
            ratios[f"frozen_{bucket}_fertility"] = new / old
            gates[f"frozen_{bucket}_within_tolerance"] = new <= old * limit
        old_extra = original[extra_name]["quality"]["all"]["tokens"]
        new_extra = candidate[extra_name]["quality"]["all"]["tokens"]
        ratios["extra_validation_tokens"] = new_extra / old_extra
        gates["extra_validation_at_least_one_percent_fewer_tokens"] = (
            new_extra * 100 <= old_extra * 99
        )
        for label, key in (("frozen", frozen_name), ("extra", extra_name)):
            for mode in ("single", "batch"):
                metric = f"{mode}_MB_per_second"
                ratio = candidate[key]["speed"][metric] / original[key]["speed"][metric]
                ratios[f"{label}_{mode}_throughput"] = ratio
                gates[f"{label}_{mode}_throughput_at_least_eighty_percent"] = ratio >= 0.8
        passes = all(gates.values())
        outcomes[name] = {
            "baseline": baseline,
            "vocab_size": metadata["vocab_size"],
            "baseline_vocab_size": size,
            "passes": passes,
            "gates": gates,
            "ratios_to_baseline": ratios,
            "extra_validation_tokens": new_extra,
            "frozen_validation_tokens": candidate[frozen_name]["quality"]["all"]["tokens"],
        }
        if passes:
            rank = (new_extra, outcomes[name]["frozen_validation_tokens"], name)
            if size not in selected or rank < selected[size][0]:
                selected[size] = (rank, name)
    return {
        "selection_corpora": [frozen_name, extra_name],
        "throughput_gate": "both single and batch medians on both validation corpora",
        "candidates": outcomes,
        "selected_by_vocabulary_size": {
            str(size): item[1] for size, item in sorted(selected.items())
        },
        "default_candidate": selected.get(57925, (None, None))[1],
        "published_default_retained": 57925 not in selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--frozen-corpus", default=FROZEN)
    parser.add_argument("--extra-corpus", default=EXTRA)
    parser.add_argument(
        "--corpus-audit", type=Path, default=Path("data/processed/expanded-v2/audit.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.benchmark.read_bytes()
    report = json.loads(raw)
    audit_bytes = args.corpus_audit.read_bytes()
    audit = json.loads(audit_bytes)
    if not audit["gates"] or not all(value is True for value in audit["gates"].values()):
        parser.error("corpus growth and leakage gates must pass before model promotion")
    expected = {
        args.frozen_corpus: audit["frozen_sha256"]["validation"],
        args.extra_corpus: audit["split_sha256"]["extra-validation"],
    }
    if any(
        report["corpora"][label]["source_sha256"] != digest for label, digest in expected.items()
    ):
        parser.error(
            "selection requires frozen validation and audited extra-validation, not test data"
        )
    if report["repeats"] < 5 or report["batch_size"] != 64:
        parser.error("promotion needs at least five repeats and batches of 64")
    result = assess(report, args.frozen_corpus, args.extra_corpus)
    result["corpus_audit_sha256"] = hashlib.sha256(audit_bytes).hexdigest()
    result["benchmark_sha256"] = hashlib.sha256(raw).hexdigest()
    result["protocol_sha256"] = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["selected_by_vocabulary_size"], sort_keys=True))
    return 0 if result["selected_by_vocabulary_size"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

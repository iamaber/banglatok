#!/usr/bin/env python3
"""Prepare a private training corpus without making source text publishable.

The input manifest is local evidence, not a license generator. Every declared
source must independently authorize training and public model artifacts. Output
is marked restricted and MUST NOT be passed to the public dataset publisher.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import prepare_corpus_v2 as corpus

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "data/processed/expanded-v2"
DEFAULT_OUTPUT = ROOT / "data/processed/restricted-training"
RESTRICTED_MARKER = ".restricted-corpus.json"
RIGHTS_BASES = {"direct-consent", "first-party-terms", "written-contract"}
REQUIRED_GRANTS = (
    "rights_holder_authorized",
    "training_allowed",
    "transformations_allowed",
    "commercial_use_allowed",
    "model_artifact_distribution_allowed",
    "platform_terms_allow_training",
    "privacy_reviewed",
    "manual_privacy_review_required",
)


def sha256(path: Path) -> str:
    return corpus.digest_file(path)


def local_only(path: Path, description: str) -> Path:
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT) and not resolved.is_relative_to(ROOT / "data/raw"):
        raise ValueError(
            f"{description} inside the repository must be under ignored data/raw: {path}"
        )
    return resolved


def load_manifest(path: Path) -> tuple[dict, dict[str, dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("restricted manifest must have schema_version 1")
    entries = manifest.get("sources")
    if not isinstance(entries, list) or not entries:
        raise ValueError("restricted manifest requires a nonempty sources list")
    sources: dict[str, dict] = {}
    for number, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"source declaration {number} must be an object")
        source = entry.get("record_source")
        if not isinstance(source, str) or not source.strip() or source in sources:
            raise ValueError(
                f"source declaration {number} has a missing or duplicate record_source"
            )
        if entry.get("rights_basis") not in RIGHTS_BASES:
            raise ValueError(f"{source}: rights_basis must be one of {sorted(RIGHTS_BASES)}")
        missing = [grant for grant in REQUIRED_GRANTS if entry.get(grant) is not True]
        if missing:
            raise ValueError(f"{source}: required grants/reviews are not true: {missing}")
        if entry.get("raw_redistribution_allowed") is not False:
            raise ValueError(f"{source}: this path requires raw_redistribution_allowed=false")
        for field in ("retention_policy", "deletion_process"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{source}: {field} is required")
        description = entry.get("public_description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{source}: public_description is required")
        evidence = entry.get("evidence_path")
        expected = entry.get("evidence_sha256")
        if not isinstance(evidence, str) or not isinstance(expected, str):
            raise ValueError(f"{source}: evidence_path and evidence_sha256 are required")
        evidence_path = local_only(path.parent / evidence, f"{source} rights evidence")
        if not evidence_path.is_file() or sha256(evidence_path) != expected:
            raise ValueError(f"{source}: rights evidence is missing or its checksum changed")
        sources[source] = entry
    return manifest, sources


def expand_candidates(patterns: list[str]) -> list[Path]:
    candidates: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            raise ValueError(f"restricted candidate path/glob has no matches: {pattern}")
        for match in matches:
            path = local_only(Path(match), "restricted candidate")
            if not path.is_file():
                raise ValueError(f"restricted candidate is not a file: {path}")
            candidates.add(path)
    return sorted(candidates)


def validate_candidate_sources(candidates: list[Path], declarations: dict[str, dict]) -> dict:
    seen: set[str] = set()
    hashes = {}
    for path in candidates:
        hashes[str(path)] = sha256(path)
        for number, record in enumerate(corpus.read_records(path), 1):
            source = record["source"]
            if source not in declarations:
                raise ValueError(f"{path}:{number}: undeclared restricted source {source!r}")
            if record["split"]:
                raise ValueError(
                    f"{path}:{number}: restricted training candidate split must be empty"
                )
            if "banglishrev" in (source + record["group"] + record["attribution"]).lower():
                raise ValueError(f"{path}:{number}: BanglishRev remains evaluation-only")
            seen.add(source)
    unused = set(declarations) - seen
    if unused:
        raise ValueError(
            f"restricted manifest declarations have no candidate records: {sorted(unused)}"
        )
    return hashes


def output_allowed(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == (ROOT / "data/processed/expanded-v2").resolve():
        raise ValueError("restricted output cannot replace the public expanded-v2 corpus")
    if resolved.is_relative_to(ROOT) and not resolved.is_relative_to(ROOT / "data/processed"):
        raise ValueError(
            "restricted output inside the repository must be under ignored data/processed"
        )
    return resolved


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--manifest", type=Path, required=True, help="local rights/privacy manifest"
    )
    result.add_argument(
        "--candidates", action="append", required=True, help="repeatable JSONL path/glob"
    )
    result.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    result.add_argument("--reserved-eval", type=Path, default=ROOT / corpus.DEFAULT_RESERVED)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--work-db", type=Path)
    return result


def prepare(args: argparse.Namespace) -> bool:
    manifest_path = local_only(args.manifest, "restricted manifest")
    _manifest, declarations = load_manifest(manifest_path)
    candidates = expand_candidates(args.candidates)
    candidate_hashes = validate_candidate_sources(candidates, declarations)
    output = output_allowed(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"restricted output directory must be empty: {output}")
    marker = {
        "schema_version": 1,
        "classification": "restricted-training-data",
        "raw_redistribution_allowed": False,
        "warning": "Do not upload this directory or any JSONL file from it.",
        "manifest_sha256": sha256(manifest_path),
        "candidate_sha256": candidate_hashes,
        "record_sources": sorted(declarations),
    }
    (output / RESTRICTED_MARKER).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reserved_candidates = [
        str(path)
        for name in ("extra-validation.jsonl", "extra-test.jsonl")
        if (path := args.baseline_dir.resolve() / name).is_file()
    ]
    prepared = corpus.prepare(
        argparse.Namespace(
            candidates=[str(path) for path in candidates],
            reserved_candidates=reserved_candidates,
            baseline_dir=args.baseline_dir.resolve(),
            output=output,
            min_train_records=0,
            min_train_characters=0,
            work_db=args.work_db.resolve() if args.work_db else None,
            reserved_eval=args.reserved_eval.resolve(),
        )
    )
    if prepared:
        audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
        accepted = {
            source: {
                "description": declarations[source]["public_description"],
                "rights_basis": declarations[source]["rights_basis"],
                "records": 0,
                "buckets": {},
            }
            for source in declarations
        }
        for item in audit["source_bucket_records"]:
            source = item["source"]
            if item["split"] == "train" and source in accepted:
                accepted[source]["records"] += item["count"]
                accepted[source]["buckets"][item["bucket"]] = item["count"]
        summary = {
            "schema_version": 1,
            "classification": "restricted-training-data",
            "raw_redistribution_allowed": False,
            "manifest_sha256": marker["manifest_sha256"],
            "training_sha256": sha256(output / "train.jsonl"),
            "public_baseline_training_sha256": sha256(args.baseline_dir.resolve() / "train.jsonl"),
            "sources": sorted(accepted.values(), key=lambda item: item["description"]),
        }
        (output / "restricted-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return prepared


def main() -> int:
    args = parser().parse_args()
    try:
        return 0 if prepare(args) else 1
    except (OSError, ValueError) as error:
        print(f"prepare_restricted_corpus: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

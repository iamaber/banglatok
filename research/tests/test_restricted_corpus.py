"""Restricted training-data fixtures; no private data or network access."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PREPARER = Path(__file__).resolve().parents[2] / "tools" / "prepare_restricted_corpus.py"


def row(text, group, *, source="public-fixture", bucket="english", split=""):
    return {
        "text": text,
        "bucket": bucket,
        "source": source,
        "group": group,
        "attribution": "fixture attribution",
        "split": split,
    }


def write_rows(path, rows):
    encoded = (json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in rows)
    path.write_text("".join(encoded), encoding="utf-8")


class RestrictedCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.baseline = self.root / "public-v2"
        self.baseline.mkdir()
        write_rows(
            self.baseline / "train.jsonl",
            [
                row(
                    "This public baseline sentence remains available.",
                    "public:train",
                    split="train",
                )
            ],
        )
        write_rows(
            self.baseline / "validation.jsonl",
            [
                row(
                    "This public validation sentence remains frozen.",
                    "public:validation",
                    split="validation",
                )
            ],
        )
        write_rows(
            self.baseline / "test.jsonl",
            [row("This public test sentence remains frozen.", "public:test", split="test")],
        )
        write_rows(
            self.baseline / "extra-validation.jsonl",
            [row("This public extra validation sentence is protected.", "public:extra-validation")],
        )
        write_rows(
            self.baseline / "extra-test.jsonl",
            [row("This public extra test sentence is protected.", "public:extra-test")],
        )
        self.candidate = self.root / "partner.jsonl"
        write_rows(
            self.candidate,
            [
                row(
                    "ami ajke order ta pelam na, support e keu reply dicche?",
                    "restricted:1",
                    source="partner-natural-banglish",
                    bucket="banglish",
                )
            ],
        )
        self.evidence = self.root / "agreement.txt"
        self.evidence.write_text(
            "Fixture agreement authorizes the declared uses.\n", encoding="utf-8"
        )
        self.manifest = self.root / "rights.json"
        self.declaration = {
            "record_source": "partner-natural-banglish",
            "rights_basis": "written-contract",
            "public_description": "Consented customer-support Banglish",
            "evidence_path": self.evidence.name,
            "evidence_sha256": hashlib.sha256(self.evidence.read_bytes()).hexdigest(),
            "rights_holder_authorized": True,
            "training_allowed": True,
            "transformations_allowed": True,
            "commercial_use_allowed": True,
            "model_artifact_distribution_allowed": True,
            "platform_terms_allow_training": True,
            "privacy_reviewed": True,
            "manual_privacy_review_required": True,
            "raw_redistribution_allowed": False,
            "retention_policy": ("Delete raw records after the contract retention period."),
            "deletion_process": (
                "Remove the source and retrain future releases after a valid request."
            ),
        }
        self.save_manifest()
        self.reserved = self.root / "reserved.json"
        self.reserved.write_text(
            json.dumps({"Review Content": "An unrelated held-out product review sentence."}),
            encoding="utf-8",
        )
        self.output = self.root / "restricted-output"

    def tearDown(self):
        self.temporary.cleanup()

    def save_manifest(self):
        self.manifest.write_text(
            json.dumps({"schema_version": 1, "sources": [self.declaration]}),
            encoding="utf-8",
        )

    def run_cli(self):
        return subprocess.run(
            [
                sys.executable,
                str(PREPARER),
                "--manifest",
                str(self.manifest),
                "--candidates",
                str(self.candidate),
                "--baseline-dir",
                str(self.baseline),
                "--reserved-eval",
                str(self.reserved),
                "--output",
                str(self.output),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_prepares_marked_private_training_corpus_without_changing_public_baseline(self):
        before = {path.name: path.read_bytes() for path in self.baseline.iterdir()}
        completed = self.run_cli()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        marker = json.loads((self.output / ".restricted-corpus.json").read_text())
        self.assertEqual(marker["classification"], "restricted-training-data")
        self.assertFalse(marker["raw_redistribution_allowed"])
        summary = json.loads((self.output / "restricted-summary.json").read_text())
        self.assertEqual(
            summary["training_sha256"],
            hashlib.sha256((self.output / "train.jsonl").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            summary["sources"],
            [
                {
                    "buckets": {"banglish": 1},
                    "description": "Consented customer-support Banglish",
                    "records": 1,
                    "rights_basis": "written-contract",
                }
            ],
        )
        sources = {
            json.loads(line)["source"]
            for line in (self.output / "train.jsonl").read_text(encoding="utf-8").splitlines()
        }
        self.assertEqual(sources, {"public-fixture", "partner-natural-banglish"})
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.baseline.iterdir()})

    def test_rejects_source_without_every_required_grant(self):
        self.declaration["model_artifact_distribution_allowed"] = False
        self.save_manifest()
        completed = self.run_cli()
        self.assertEqual(completed.returncode, 2)
        self.assertIn("model_artifact_distribution_allowed", completed.stderr)
        self.assertFalse(self.output.exists())

    def test_rejects_candidate_from_undeclared_source(self):
        records = [json.loads(line) for line in self.candidate.read_text().splitlines()]
        records[0]["source"] = "undeclared-source"
        write_rows(self.candidate, records)
        completed = self.run_cli()
        self.assertEqual(completed.returncode, 2)
        self.assertIn("undeclared restricted source", completed.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

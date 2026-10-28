"""Isolated publisher CLI fixtures: no ignored corpus or Hub access."""

import contextlib
import gzip
import hashlib
import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zipfile
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
preparer = import_module("prepare_corpus_v2")
publisher = import_module("publish_dataset")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def row(text, source, group, split):
    attribution = "tatoeba.org #123; Example author; CC BY 2.0 France"
    if source.startswith("wikipedia-"):
        attribution = (
            "https://en.wikipedia.org/wiki/Example; Wikipedia contributors; "
            "CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/); "
            "GFDL where applicable; snapshot 20231101"
        )
    if source == "pralekha-ben-alignable":
        attribution = (
            "Pralekha; AI4Bharat; CC BY 4.0; pinned dataset card; n_id=fixture; doc_id=fixture"
        )
    return dict(
        text=text,
        bucket="bangla" if source == "pralekha-ben-alignable" else "english",
        source=source,
        group=group,
        attribution=attribution,
        split=split,
    )


class PublishDatasetV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.baseline = self.root / "data/processed/frozen-v1"
        self.expanded = self.root / "data/processed/expanded-v2"
        self.expanded.mkdir(parents=True)
        self.baseline.mkdir(parents=True)
        records = {
            "train": row(
                "Original training sentence preserves its historical contributor notice.",
                "tatoeba-fra",
                "tatoeba:original",
                "train",
            ),
            "validation": row(
                "Amber birch cedar drift elm fir granite hazel iris jade.",
                "tatoeba-fra",
                "tatoeba:validation",
                "validation",
            ),
            "test": row(
                "Frozen testing material is retained with identical original bytes.",
                "tatoeba-fra",
                "tatoeba:test",
                "test",
            ),
        }
        for split, record in records.items():
            save_json(self.baseline / f"{split}.jsonl", record)
        for split in ("validation", "test"):
            (self.expanded / f"{split}.jsonl").write_bytes(
                (self.baseline / f"{split}.jsonl").read_bytes()
            )
        train = row(
            "Expanded encyclopedic paragraphs introduce genuine descriptive educational material.",
            "wikipedia-en-20231101",
            "wiki-en:123",
            "train",
        )
        domain = row(
            "সরকারি জনসেবার নথিতে নাগরিক সহায়তার বিস্তারিত প্রক্রিয়া ব্যাখ্যা করা হয়েছে।",
            "pralekha-ben-alignable",
            "pralekha:fixture",
            "train",
        )
        (self.expanded / "train.jsonl").write_bytes(
            (self.baseline / "train.jsonl").read_bytes()
            + (json.dumps(train, ensure_ascii=False) + "\n").encode()
            + (json.dumps(domain, ensure_ascii=False) + "\n").encode()
        )
        for split, text, group in (
            (
                "extra-validation",
                "Additional validation paragraphs describe coastal geography "
                "and local transportation.",
                "wiki-en:124",
            ),
            (
                "extra-test",
                "Separate testing paragraphs explain scientific discoveries "
                "and historical development.",
                "wiki-en:125",
            ),
        ):
            save_json(
                self.expanded / f"{split}.jsonl",
                row(text, "wikipedia-en-20231101", group, split.removeprefix("extra-")),
            )
        self.reserved = self.root / preparer.DEFAULT_RESERVED
        save_json(
            self.reserved,
            [
                {
                    "Reviews": [
                        {
                            "Review Content": (
                                "PRIVATE REVIEW NEVER UPLOAD THIS distinct "
                                "reserved consumer opinion."
                            )
                        }
                    ]
                }
            ],
        )
        self.frozen_hashes = {
            split: publisher.sha256(self.baseline / f"{split}.jsonl") for split in records
        }
        with contextlib.closing(sqlite3.connect(self.root / "audit.sqlite")) as db:
            self.audit = preparer.audit(
                self.expanded, self.reserved, db, self.frozen_hashes, self.baseline
            )
        self.audit["gates"] = dict(
            minimum_training_records=True, minimum_training_characters=True, no_leakage=True
        )
        self.stats = {
            "schema_version": 2,
            "frozen_sha256": self.frozen_hashes,
            "growth_requirements": {"records": 2, "normalized_characters": 80},
            "gates": self.audit["gates"],
            "near_duplicate_method": preparer.NEAR_METHOD,
            "quality_policy": {"characters": [12, 2000]},
            "output_source_bucket_records": [
                r
                for r in self.audit["source_bucket_records"]
                if r["split"] not in {"validation", "test"}
            ],
            "output_source_bucket_characters": [
                r
                for r in self.audit["source_bucket_characters"]
                if r["split"] not in {"validation", "test"}
            ],
            "candidate_source_sha256": {},
        }
        self.save_reports()
        original_lock = []
        raw = self.root / "data/raw"
        (raw / "banglatlit").mkdir(parents=True)
        (raw / "banglatlit/LICENSE").write_text("MIT fixture notice\n")
        (raw / "tatoeba").mkdir()
        for language in ("ben", "fra"):
            with zipfile.ZipFile(raw / f"tatoeba/{language}-eng.zip", "w") as archive:
                archive.writestr("_about.txt", "Tatoeba fixture authors; CC BY 2.0\n")
        for name in ("banglatlit/LICENSE", "tatoeba/ben-eng.zip", "tatoeba/fra-eng.zip"):
            path = raw / name
            original_lock.append(
                {"path": name, "sha256": publisher.sha256(path), "bytes": path.stat().st_size}
            )
        save_json(self.root / "data/sources.lock.json", original_lock)
        actual_lock = json.loads((TOOLS.parent / "data/corpus-v2-sources.lock.json").read_text())
        self.lock = {**actual_lock, "artifacts": []}
        for entry in actual_lock["artifacts"]:
            if entry["source"] != "license":
                continue
            path = raw / "expanded-v2/artifacts" / entry["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Pinned source licensing notice fixture.\n")
            self.lock["artifacts"].append(
                {**entry, "sha256": publisher.sha256(path), "bytes": path.stat().st_size}
            )
        save_json(self.root / "data/corpus-v2-sources.lock.json", self.lock)
        domain_lock = json.loads(
            (TOOLS.parent / "data/corpus-v2-domain-sources.lock.json").read_text()
        )
        save_json(self.root / "data/corpus-v2-domain-sources.lock.json", domain_lock)
        (self.root / "docs").mkdir()
        (self.root / "docs/DATASET.md").write_bytes((TOOLS.parent / "docs/DATASET.md").read_bytes())
        reports = self.root / "research/reports"
        reports.mkdir(parents=True)
        for name in ("CORPUS_V2_PROTOCOL.md", "CORPUS_V2_SOURCES.md"):
            (reports / name).write_bytes((TOOLS.parent / "research/reports" / name).read_bytes())
        for name in (
            "prepare_corpus_v2.py",
            "acquire_corpus_v2.py",
            "acquire_pralekha_v2.py",
            "publish_dataset.py",
        ):
            path = self.root / "tools" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes((TOOLS / name).read_bytes())
        self.output = self.root / "staged"
        self.addCleanup(patch.stopall)
        patch.object(publisher, "ROOT", self.root).start()
        patch.object(publisher, "V2_MIN_TRAIN_RECORDS", 2).start()
        patch.object(publisher, "V2_MIN_TRAIN_CHARACTERS", 80).start()
        patch.object(publisher, "FROZEN_HASHES", self.frozen_hashes).start()
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=Corpus Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                f"core.hooksPath={self.root / 'no-hooks'}",
                "commit",
                "--allow-empty",
                "--no-gpg-sign",
                "-q",
                "-m",
                "Research fixture",
            ],
            check=True,
            capture_output=True,
        )

    def save_reports(self):
        save_json(self.expanded / "audit.json", self.audit)
        save_json(self.expanded / "stats.json", self.stats)

    def stage(self):
        with (
            patch.object(
                sys,
                "argv",
                ["publish_dataset.py", "--corpus-version", "v2", "--dry-run", str(self.output)],
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            publisher.main()

    def test_explicit_v2_dry_run_stages_only_versioned_corpus_and_shared_card(self):
        original = {p: p.read_bytes() for p in self.baseline.iterdir()}
        self.stage()
        manifest = json.loads((self.output / "v2/manifest.json").read_text())
        self.assertEqual(manifest["corpus_version"], "v2")
        self.assertEqual(manifest["splits"]["train"]["rows"], 3)
        for split in ("train", "validation", "test", "extra-validation", "extra-test"):
            entry = manifest["splits"][split]
            self.assertEqual(entry["path"], f"v2/data/{split}.jsonl.gz")
            with gzip.open(self.output / entry["path"], "rb") as stream:
                decoded = stream.read()
            self.assertEqual(decoded, (self.expanded / f"{split}.jsonl").read_bytes())
            self.assertEqual(hashlib.sha256(decoded).hexdigest(), entry["uncompressed_sha256"])
        self.assertEqual(manifest["frozen_sha256"], self.frozen_hashes)
        self.assertTrue((self.output / "v2/independent-audit.json").is_file())
        self.assertTrue((self.output / "v2/PROTOCOL.md").is_file())
        self.assertTrue((self.output / "v2/frozen-v1-sources.lock.json").is_file())
        self.assertTrue((self.output / "v2/domain-sources.lock.json").is_file())
        card = (self.output / "README.md").read_text()
        self.assertIn("config_name: default\n  default: true", card)
        self.assertIn("config_name: v2", card)
        self.assertIn("split: extra_validation", card)
        self.assertIn("path: data/train.jsonl.gz", card)
        self.assertIn("path: v2/data/train.jsonl.gz", card)
        self.assertIn("CC BY-SA 4.0", card)
        self.assertFalse((self.output / "manifest.json").exists())
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertTrue(path.name == "README.md" or path.is_relative_to(self.output / "v2"))
                self.assertNotIn(b"PRIVATE REVIEW NEVER UPLOAD", path.read_bytes())
        self.assertEqual(original, {p: p.read_bytes() for p in self.baseline.iterdir()})

    def test_data_changed_after_audit_is_rejected_even_with_passing_flags(self):
        path = self.expanded / "train.jsonl"
        path.write_bytes(path.read_bytes().replace(b"Expanded", b"Modified"))
        with self.assertRaisesRegex(ValueError, "audit.*mismatch"):
            self.stage()

    def test_forged_unique_count_is_rejected(self):
        self.audit["accepted_unique_training_records"] = 9999999
        self.save_reports()
        with self.assertRaisesRegex(ValueError, "audit.*mismatch"):
            self.stage()

    def test_growth_is_recomputed_not_taken_from_passing_flags(self):
        with patch.object(publisher, "V2_MIN_TRAIN_RECORDS", 3):
            with self.assertRaisesRegex(ValueError, "growth"):
                self.stage()

    def test_failed_gate_is_rejected(self):
        self.audit["gates"]["no_leakage"] = False
        self.save_reports()
        with self.assertRaisesRegex(ValueError, "gate"):
            self.stage()

    def test_frozen_heldout_bytes_cannot_be_replaced(self):
        path = self.expanded / "validation.jsonl"
        path.write_bytes(path.read_bytes().replace(b"Amber", b"Green"))
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.stage()

    def test_missing_attribution_is_rejected(self):
        path = self.expanded / "train.jsonl"
        path.write_bytes(path.read_bytes().replace(b"Wikipedia contributors", b"Uncredited writer"))
        with self.assertRaisesRegex(ValueError, "attribution"):
            self.stage()

    def test_source_license_evidence_checksum_is_verified(self):
        (self.root / "data/raw/expanded-v2/artifacts/README.md").write_text("Changed notice")
        with self.assertRaisesRegex(ValueError, "notice.*mismatch"):
            self.stage()

    def test_default_cli_still_uses_v1(self):
        for split in self.frozen_hashes:
            (self.root / f"data/processed/{split}.jsonl").write_bytes(
                (self.baseline / f"{split}.jsonl").read_bytes()
            )
        save_json(self.root / "data/processed/stats.json", self.stats)
        for name in ("cmd/bntok/main.go", "go.mod", "go.sum"):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((TOOLS.parent / name).read_bytes())
        with (
            patch.object(publisher, "EXPECTED_ROWS", dict.fromkeys(self.frozen_hashes, 1)),
            patch.object(publisher, "TRAIN_SHA256", self.frozen_hashes["train"]),
            patch.object(sys, "argv", ["publish_dataset.py", "--dry-run", str(self.output)]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            publisher.main()
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["manifest_version"], 1)
        self.assertEqual(
            manifest["splits"]["train"]["uncompressed_sha256"], self.frozen_hashes["train"]
        )
        with gzip.open(self.output / "data/train.jsonl.gz", "rb") as stream:
            self.assertEqual(stream.read(), (self.baseline / "train.jsonl").read_bytes())
        self.assertFalse((self.output / "v2").exists())


if __name__ == "__main__":
    unittest.main()

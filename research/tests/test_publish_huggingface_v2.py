"""Exercise publication folders through the offline publisher CLI."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from banglatok import Tokenizer

ROOT = Path(__file__).resolve().parents[2]


class VersionedModelPublishingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "bn-bpe-28k-v2.json"
        self.model.write_text(
            json.dumps(
                {
                    "version": 1,
                    "algorithm": "bpe",
                    "normalization": "nfc-whitespace-v1",
                    "pieces": [],
                    "merges": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.training = self.root / "train.jsonl"
        self.training.write_text('{"text":"আমি বাংলা বলি","split":"train"}\n', encoding="utf-8")
        self.output = self.root / "export"

    def run_publisher(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / "tools/publish_huggingface.py"), *arguments],
            cwd=ROOT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            capture_output=True,
            text=True,
            check=False,
        )

    def custom_arguments(self) -> list[str]:
        return [
            "--model",
            str(self.model),
            "--training",
            str(self.training),
            "--dry-run",
            str(self.output),
        ]

    def test_custom_models_export_exact_provenance_without_historical_claims(self) -> None:
        second = self.root / "bn-bpe-58k-v2.json"
        second.write_bytes(self.model.read_bytes())
        result = self.run_publisher(*self.custom_arguments(), "--model", str(second))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({p.name for p in self.output.iterdir()}, {self.model.stem, second.stem})
        folder = self.output / self.model.stem
        card = (folder / "README.md").read_text(encoding="utf-8")
        self.assertIn("iamaber/banglatok-bn-bpe-28k-v2", card)
        self.assertIn("configuration `v2`", card)
        self.assertIn("/tree/main/v2", card)
        self.assertIn(hashlib.sha256(self.model.read_bytes()).hexdigest(), card)
        self.assertIn(hashlib.sha256(self.training.read_bytes()).hexdigest(), card)
        self.assertIn("260 token IDs", card)
        self.assertIn('Tokenizer("bn-bpe-28k-v2.json")', card)
        self.assertIn("pip install banglatok", card)
        self.assertIn("No published comparison", card)
        self.assertIn("CORPUS_V2_PROTOCOL.md", card)
        for historical in ("22% fewer", "1.1–1.7", "58.5 / 71.6", "563,476"):
            self.assertNotIn(historical, card)
        self.assertEqual((folder / self.model.name).read_bytes(), self.model.read_bytes())
        self.assertEqual((folder / "USAGE.md").read_bytes(), (ROOT / "docs/USAGE.md").read_bytes())
        provenance = json.loads((folder / "model_provenance.json").read_text())
        self.assertEqual(provenance["vocab_size"], 260)
        self.assertEqual(provenance["dataset_config"], "v2")
        with Tokenizer(folder / self.model.name) as tokenizer:
            self.assertEqual(tokenizer.decode(tokenizer.encode("আমি বাংলা বলি")), "আমি বাংলা বলি")

    def benchmark(self, source_hash: str | None = None) -> Path:
        digest = source_hash or hashlib.sha256(self.model.read_bytes()).hexdigest()
        name = f"banglatok {self.model.stem}"
        report = {
            "created_utc": "2026-11-13T00:00:00+00:00",
            "cpu_model": "Isolated fixture CPU",
            "platform": "fixture-platform",
            "python": "3.12.13",
            "packages": {"banglatok": "0.2.2"},
            "repeats": 5,
            "batch_size": 64,
            "candidates": {
                name: {
                    "source": f"{self.model.name} sha256 {digest}",
                    "vocab_size": 260,
                    "stress_round_trips": 16,
                    "stress_cases": 16,
                }
            },
            "corpora": {
                "Isolated test fixture": {
                    "records": 2,
                    "source": "fixture.jsonl",
                    "source_sha256": "a" * 64,
                    "ordered_text_sha256": "b" * 64,
                    "candidates": {
                        name: {
                            "quality": {
                                "all": {
                                    "tokens_per_word": 2.125,
                                    "round_trip_percent": 100.0,
                                    "texts_with_unknowns": 0,
                                }
                            },
                            "speed": {"single_MB_per_second": 12.5, "batch_MB_per_second": 15.5},
                        }
                    },
                }
            },
        }
        path = self.root / "measured.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_measured_report_uses_real_comparison_schema(self) -> None:
        report = self.benchmark()
        result = self.run_publisher(*self.custom_arguments(), "--benchmark", str(report))
        self.assertEqual(result.returncode, 0, result.stderr)
        folder = self.output / self.model.stem
        card = (folder / "README.md").read_text(encoding="utf-8")
        self.assertIn("Isolated test fixture", card)
        self.assertIn("Isolated fixture CPU", card)
        self.assertIn("2.125", card)
        self.assertIn("12.5", card)
        self.assertNotIn("No published comparison", card)
        self.assertNotIn("22% fewer", card)
        self.assertEqual((folder / "benchmark-1.json").read_bytes(), report.read_bytes())

    def test_stale_benchmark_model_hash_is_rejected(self) -> None:
        report = self.benchmark("0" * 64)
        result = self.run_publisher(*self.custom_arguments(), "--benchmark", str(report))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match", result.stderr)

    def test_custom_v1_names_and_duplicate_ids_are_rejected(self) -> None:
        old = self.root / "bn-bpe-28k.json"
        old.write_bytes(self.model.read_bytes())
        result = self.run_publisher("--model", str(old), "--dry-run", str(self.output))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("immutable v1", result.stderr)
        result = self.run_publisher(*self.custom_arguments(), "--model", str(self.model))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate", result.stderr)

    def test_training_provenance_mismatch_is_rejected(self) -> None:
        self.model.with_suffix(".provenance.json").write_text(
            json.dumps({"training_sha256": "0" * 64}), encoding="utf-8"
        )
        result = self.run_publisher(*self.custom_arguments())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("training SHA-256 does not match", result.stderr)

    def test_original_bundled_cli_still_exports_both_v1_models(self) -> None:
        result = self.run_publisher("--dry-run", str(self.output))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({p.name for p in self.output.iterdir()}, {"bn-bpe-28k", "bn-bpe-58k"})
        card = (self.output / "bn-bpe-28k" / "README.md").read_text(encoding="utf-8")
        self.assertIn("563,476 training records", card)
        self.assertIn('Tokenizer.from_pretrained("bn-bpe-28k")', card)


if __name__ == "__main__":
    unittest.main()

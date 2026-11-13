"""Deterministic public-CLI fixtures; no native library or real corpus required."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PREPARER = Path(__file__).resolve().parents[1] / "tools" / "prepare_corpus_v2.py"


def row(text, group, bucket="english", source="fixture", split=""):
    return dict(
        text=text,
        bucket=bucket,
        source=source,
        group=group,
        attribution="Fixture author; CC0",
        split=split,
    )


def write_rows(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


class CorpusV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.baseline = self.root / "baseline"
        self.baseline.mkdir()
        self.original = row(
            "Original training provenance remains intact for this ordinary sentence.",
            "original:one",
            split="train",
        )
        self.validation = (
            "amber birch cedar delta elm fern granite hazel iris jade kelp larch "
            "maple north olive pine"
        )
        write_rows(self.baseline / "train.jsonl", [self.original])
        write_rows(
            self.baseline / "validation.jsonl",
            [row(self.validation, "wiki-bn:reserved-article", split="validation")],
        )
        write_rows(
            self.baseline / "test.jsonl",
            [
                row(
                    "Frozen testing sentences must remain completely unchanged.",
                    "frozen:test",
                    split="test",
                )
            ],
        )
        self.reserved = self.root / "reserved.json"
        self.reserved_text = (
            "Reserved customer opinions about a distinctive lovely fragrant product."
        )
        self.reserved.write_text(
            json.dumps([{"Reviews": [{"Review Content": self.reserved_text}]}]), encoding="utf-8"
        )

    def run_cli(self, records, name, minimum=0):
        candidates = self.root / (name + ".jsonl")
        write_rows(candidates, records)
        output = self.root / name
        completed = subprocess.run(
            [
                sys.executable,
                str(PREPARER),
                "--candidates",
                str(candidates),
                "--baseline-dir",
                str(self.baseline),
                "--reserved-eval",
                str(self.reserved),
                "--output",
                str(output),
                "--min-train-records",
                str(minimum),
                "--min-train-characters",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertIn(completed.returncode, (0, 1), completed.stderr)
        report = json.loads((output / "audit.json").read_text(encoding="utf-8"))
        stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
        accepted = []
        for split in ("train", "extra-validation", "extra-test"):
            accepted.extend(
                json.loads(line)
                for line in (output / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()
            )
        return completed, output, accepted, report, stats

    def test_late_near_match_protects_siblings_and_order_is_irrelevant(self):
        candidates = [
            row(
                "জ্যোতির্বিজ্ঞানীরা প্রতিদিন আকাশের দূরবর্তী নক্ষত্র সম্পর্কে নতুন তথ্য সংগ্রহ করেন।",
                "wiki-bn:new-article",
                "bangla",
            ),
            row(
                "A second distinct paragraph must not escape through another article.",
                "other:linked",
            ),
            row(
                "জ্যোতির্বিজ্ঞানীরা প্রতিদিন আকাশের দূরবর্তী নক্ষত্র সম্পর্কে নতুন তথ্য সংগ্রহ করেন!",
                "other:linked",
                "bangla",
            ),
            row(
                "বাংলা ভাষার প্রাচীন সাহিত্য নিয়ে গবেষকেরা নতুন প্রবন্ধ প্রকাশ করেছেন।",
                "wiki-bn:reserved-article",
                "bangla",
            ),
            row(" ".join(reversed(self.validation.split())), "wiki-bn:new-article"),
            row(self.reserved_text, "reserved:copy"),
            row(
                "Another paragraph from the reserved review group must be excluded.",
                "reserved:copy",
            ),
            row(
                "Quiet observatories measure distant galaxies with sensitive instruments "
                "beneath exceptionally clear winter skies.",
                "clean:article",
            ),
            row(
                "skies. winter clear exceptionally beneath instruments sensitive with "
                "galaxies distant measure observatories Quiet",
                "clean:duplicate",
            ),
        ]
        first, output, accepted, audit, _ = self.run_cli(candidates, "forward")
        second, reverse, _, _, _ = self.run_cli(list(reversed(candidates)), "reverse")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertFalse(any(audit["overlaps"].values()))
        self.assertTrue(
            all(
                record["group"] in {"original:one", "clean:article", "clean:duplicate"}
                for record in accepted
            )
        )
        self.assertEqual(len(accepted), 2)
        self.assertEqual(
            next(record for record in accepted if record["group"] == "original:one"), self.original
        )
        for split in ("train", "extra-validation", "extra-test", "validation", "test"):
            self.assertEqual(
                (output / (split + ".jsonl")).read_bytes(),
                (reverse / (split + ".jsonl")).read_bytes(),
            )
        for split in ("validation", "test"):
            self.assertEqual(
                (output / (split + ".jsonl")).read_bytes(),
                (self.baseline / (split + ".jsonl")).read_bytes(),
            )

    def test_unicode_quality_and_canonical_deduplication(self):
        good = (
            "  Cafe\u0301\u00a0readers\u2003enjoy\tcarefully written multilingual stories today.  "
        )
        bn = "আমি ক্\u200dষ এবং ক্\u200cষ শব্দের বাংলা চিহ্নগুলো যত্নসহকারে সংরক্ষণ করি 🙂"
        candidates = [
            row(good, "unicode:accent"),
            row(
                "CAFÉ readers enjoy carefully written multilingual stories today!",
                "unicode:duplicate",
            ),
            row(bn, "unicode:bn", "bangla"),
            row(
                "The letter İstanbul appears naturally in this particular sentence.",
                "unicode:dotted",
            ),
            row(
                "The letter istanbul appears naturally in this particular sentence!",
                "unicode:dotted-copy",
            ),
            row(
                "Number 2 appears beside these mathematical descriptions in prose.",
                "unicode:decimal",
            ),
            row(
                "Number ² appears beside these mathematical descriptions in prose.",
                "unicode:superscript",
            ),
            row(
                "Do contact private.person@example.com for more personal information today.",
                "bad:email",
            ),
            row(
                "Contact the local agent using 01712345678 during office hours please.", "bad:phone"
            ),
            row(
                "The current password=veryprivate should not be shared with anyone.",
                "bad:credential",
            ),
            row(
                "Unexpected \x1c control corruption ruins an otherwise ordinary sentence.",
                "bad:control",
            ),
            row(
                "Randomly corrupted \ufffd text should never enter the training corpus.",
                "bad:replacement",
            ),
            row("banana orchard banana grove banana tree banana garden fruit", "bad:repetition"),
            row(
                "This is an entirely Latin sentence incorrectly declared to be Bengali.",
                "bad:script",
                "bangla",
            ),
            row("A tiny fragment", "bad:fragment"),
            row(
                "Visit https://example.org/long/path/that/overwhelms/the/text "
                "for additional details now please.",
                "bad:url",
            ),
        ]
        completed, _, accepted, audit, stats = self.run_cli(candidates, "quality")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(any(audit["overlaps"].values()))
        self.assertFalse(any(record["group"].startswith("bad:") for record in accepted))
        self.assertEqual(
            sum(
                record["group"].startswith("unicode:accent")
                or record["group"] == "unicode:duplicate"
                for record in accepted
            ),
            1,
        )
        accent = next(
            record["text"] for record in accepted if "carefully written" in record["text"]
        )
        self.assertNotIn("\u0301", accent)
        self.assertNotIn("\u00a0", accent)
        self.assertNotIn("\u2003", accent)
        self.assertIn(bn, [record["text"] for record in accepted])
        self.assertEqual(
            sum(record["group"].startswith("unicode:dotted") for record in accepted), 1
        )
        self.assertTrue(
            {"unicode:decimal", "unicode:superscript"} <= {record["group"] for record in accepted}
        )
        reasons = {item["reason"] for item in stats["quality_rejections"]}
        self.assertTrue(
            {
                "private_information",
                "control_corruption",
                "replacement_or_invalid_unicode",
                "word_repetition",
                "bucket_script_mismatch",
                "short_new_fragment",
                "url_heavy",
            }
            <= reasons,
            reasons,
        )

    def test_growth_failure_emits_honest_audit_and_nonzero_exit(self):
        completed, _, _, audit, _ = self.run_cli([], "insufficient", minimum=100)
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(audit["accepted_unique_training_records"], 1)
        self.assertFalse(audit["gates"]["minimum_training_records"])
        self.assertTrue(audit["gates"]["no_leakage"])


if __name__ == "__main__":
    unittest.main()

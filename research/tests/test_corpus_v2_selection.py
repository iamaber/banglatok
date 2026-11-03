"""A bigger corpus must not promote a tokenizer that regresses old users."""

import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "select_corpus_v2", Path(__file__).resolve().parents[2] / "benchmarks/select_corpus_v2.py"
)
assert SPEC is not None and SPEC.loader is not None
SELECTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTION)
NEW = "banglatok bn-bpe-58k-v2-default"
OLD = "banglatok bn-bpe-58k"


def score(tokens, speed=100):
    quality = {
        bucket: {"tokens_per_word": count / 1000}
        for bucket, count in zip(SELECTION.BUCKET_LIMITS, tokens, strict=True)
    }
    quality["all"] = {
        "texts": 400,
        "exact_round_trips": 400,
        "texts_with_unknowns": 0,
        "tokens": sum(tokens),
    }
    return {
        "quality": quality,
        "speed": {"single_MB_per_second": speed, "batch_MB_per_second": speed},
    }


def report():
    return {
        "candidates": {
            SELECTION.BASELINES[0]: {"vocab_size": 27760, "source": "bundled"},
            OLD: {"vocab_size": 57925, "source": "bundled"},
            NEW: {
                "vocab_size": 57925,
                "source": "candidate.json sha256 fixture",
                "stress_cases": 16,
                "stress_round_trips": 16,
            },
        },
        "corpora": {
            SELECTION.FROZEN: {
                "candidates": {OLD: score([1000] * 4), NEW: score([995, 995, 995, 1000], 90)}
            },
            SELECTION.EXTRA: {"candidates": {OLD: score([1200] * 4), NEW: score([1150] * 4, 90)}},
        },
    }


class PromotionTests(unittest.TestCase):
    def test_bangla_regression_blocks_better_aggregate_compression(self):
        observed = report()
        observed["corpora"][SELECTION.FROZEN]["candidates"][NEW] = score([1020, 960, 960, 960])
        result = SELECTION.assess(observed)
        outcome = result["candidates"][NEW]
        self.assertTrue(outcome["gates"]["frozen_aggregate_no_regression"])
        self.assertFalse(outcome["gates"]["frozen_bangla_within_tolerance"])
        self.assertTrue(result["published_default_retained"])
        self.assertEqual(result["selected_by_vocabulary_size"], {})

    def test_unknowns_or_slow_extra_encoding_block_promotion(self):
        for failure in ("unknown", "slow"):
            with self.subTest(failure=failure):
                observed = report()
                extra = observed["corpora"][SELECTION.EXTRA]["candidates"][NEW]
                if failure == "unknown":
                    extra["quality"]["all"]["texts_with_unknowns"] = 1
                else:
                    extra["speed"]["batch_MB_per_second"] = 79
                result = SELECTION.assess(observed)
                self.assertIsNone(result["default_candidate"])
                self.assertFalse(result["candidates"][NEW]["passes"])

    def test_passing_smaller_token_count_selects_new_default(self):
        result = SELECTION.assess(report())
        self.assertEqual(result["default_candidate"], NEW)
        self.assertEqual(result["selected_by_vocabulary_size"], {"57925": NEW})
        self.assertFalse(result["published_default_retained"])

    def test_underfilled_fusion_budget_competes_with_its_size_tier(self):
        observed = report()
        observed["candidates"][NEW]["vocab_size"] = 56081
        result = SELECTION.assess(observed)
        self.assertEqual(result["candidates"][NEW]["baseline"], OLD)
        self.assertEqual(result["default_candidate"], NEW)
        self.assertEqual(result["selected_by_vocabulary_size"], {"57925": NEW})

    def test_larger_vocabulary_than_baseline_cannot_be_promoted(self):
        observed = report()
        observed["candidates"][NEW]["vocab_size"] = 60000
        result = SELECTION.assess(observed)
        self.assertFalse(result["candidates"][NEW]["passes"])
        self.assertIsNone(result["default_candidate"])


if __name__ == "__main__":
    unittest.main()

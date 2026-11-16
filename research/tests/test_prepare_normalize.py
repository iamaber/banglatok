import importlib.util
import sys
import unicodedata
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "prepare_corpus_v2.py"
spec = importlib.util.spec_from_file_location("prepare_corpus_v2", SCRIPT)
assert spec is not None and spec.loader is not None
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@unittest.skipUnless(importlib.util.find_spec("banglatok"), "banglatok is not installed")
class NormalizeMatchesGoTests(unittest.TestCase):
    def test_separators_and_controls_collapse_like_the_go_tokenizer(self):
        import banglatok

        candidates = [
            chr(code)
            for code in range(sys.maxunicode + 1)
            if unicodedata.category(chr(code)) in {"Zs", "Zl", "Zp", "Cc", "Cf"}
        ]
        self.assertIn("\u00a0", candidates)
        self.assertIn("\x1c", candidates)
        for character in candidates:
            text = f"a{character}b{character}{character}c"
            self.assertEqual(
                prepare.normalize(text), banglatok.normalize(text), f"U+{ord(character):04X}"
            )


if __name__ == "__main__":
    unittest.main()

"""Exercise the release gate with actual wheel/sdist archives."""

import io
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path


class DistributionTests(unittest.TestCase):
    def check(self, *, omit_native=False, leak=False, wrong_version=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notices = ["LICENSE", "THIRD_PARTY_NOTICES.md", "third_party/licenses/GO-LICENSE",
                       "third_party/licenses/X-TEXT-LICENSE", "third_party/licenses/X-TEXT-PATENTS"]
            source = ["pyproject.toml", "setup.py", "go.mod", "go.sum", "tokenizer/model.go",
                      "internal/native/native.go", "cmd/bntok-shared/main.go",
                      "src/bn_tokenizers_embedding/data/demo-unigram.json", *notices]
            with tarfile.open(root / "bn_tokenizers_embedding-0.1.0a1.tar.gz", "w:gz") as archive:
                for name in source:
                    entry = tarfile.TarInfo(f"bn_tokenizers_embedding-0.1.0a1/{name}")
                    entry.size = 1
                    archive.addfile(entry, io.BytesIO(b"x"))
            info = "bn_tokenizers_embedding-0.1.0a1.dist-info"
            with zipfile.ZipFile(root / "bn_tokenizers_embedding-0.1.0a1-py3-none-linux_x86_64.whl", "w") as archive:
                version = "9.9.9" if wrong_version else "0.1.0a1"
                archive.writestr(f"{info}/METADATA", f"Name: bn-tokenizers-embedding\nVersion: {version}\nLicense-Expression: MIT AND BSD-3-Clause\nRequires-Python: >=3.10\n")
                archive.writestr(f"{info}/WHEEL", "Root-Is-Purelib: false\nTag: py3-none-linux_x86_64\n")
                for name in notices:
                    archive.writestr(f"{info}/licenses/{name}", "notice")
                for name in ("data/demo-unigram.json", "py.typed"):
                    archive.writestr(f"bn_tokenizers_embedding/{name}", "fixture")
                if not omit_native:
                    archive.writestr("bn_tokenizers_embedding/_go.so", b"native fixture")
                if leak:
                    archive.writestr("data/raw/private.jsonl", "must not ship")
            return subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "tools/check_dist.py"),
                                   str(root), "--version", "0.1.0a1"], capture_output=True, text=True)

    def test_complete_release_archives_are_accepted(self):
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_native_runtime_is_rejected(self):
        result = self.check(omit_native=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("native runtime", result.stderr)

    def test_raw_corpus_is_rejected(self):
        result = self.check(leak=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("excluded content", result.stderr)

    def test_wrong_release_version_is_rejected(self):
        result = self.check(wrong_version=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("version", result.stderr)


if __name__ == "__main__":
    unittest.main()

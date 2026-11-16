import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "acquire_corpus_v2.py"
spec = importlib.util.spec_from_file_location("acquire_corpus_v2", SCRIPT)
assert spec is not None and spec.loader is not None
acquire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquire)


class PublishShardTests(unittest.TestCase):
    def test_metadata_exists_before_shard_appears(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "metadata.json"
            partial = root / "shard.jsonl.part"
            destination = root / "shard.jsonl"
            partial.write_text("{}\n", encoding="utf-8")
            replace = Path.replace

            def checked(self, target):
                if Path(target) == destination and not metadata_path.exists():
                    raise AssertionError("shard published before its metadata")
                return replace(self, target)

            with mock.patch.object(Path, "replace", checked):
                acquire.publish_shard(
                    metadata_path, {"output": str(destination)}, partial, destination
                )
            self.assertTrue(destination.exists())
            self.assertTrue(metadata_path.exists())

    def test_skipped_shard_writes_only_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "shard.jsonl"
            acquire.publish_shard(
                root / "metadata.json", {"output": None}, root / "absent.part", destination
            )
            self.assertFalse(destination.exists())
            self.assertTrue((root / "metadata.json").exists())


if __name__ == "__main__":
    unittest.main()

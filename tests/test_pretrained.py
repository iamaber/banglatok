"""Bundled models and the higher-level API, exercised through the real native library."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import as_file, files
from pathlib import Path

from bn_tokenizers_embedding import (
    BOS_ID,
    DEFAULT_MODEL,
    EOS_ID,
    PAD_ID,
    Tokenizer,
    available_models,
    normalize,
)
from bn_tokenizers_embedding.__main__ import main

STRESS = [
    "আমি বাংলায় গান গাই।",
    "ami banglay gaan gai!",
    "meetingটা ১০টায়, don't be late 🙂",
    "cafe\u0301 ＡＢＣ ①",
    "বাংলা\u200dভাষা বাংলা\u200cভাষা",
    "👩🏽‍💻 🫠 🚀",
    "<unk> <s> </s> <pad> ▁ <0x20>",
    "a\x00b",
    "  বাংলা\tEnglish\nBanglish  ",
    "",
    "中文 العربية हिन्दी",
    "কো ক\u09c7\u09be ড়া",
    "বাংলা" * 400,
]


class PretrainedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = {name: Tokenizer.from_pretrained(name) for name in available_models()}

    @classmethod
    def tearDownClass(cls) -> None:
        for tokenizer in cls.models.values():
            tokenizer.close()

    def test_bundled_models(self) -> None:
        self.assertEqual(available_models(), ["bn-bpe-28k", "bn-bpe-58k"])
        self.assertEqual(Tokenizer.from_pretrained().vocab_size, 57925)
        self.assertEqual(DEFAULT_MODEL, "bn-bpe-58k")
        self.assertEqual(self.models["bn-bpe-28k"].vocab_size, 27760)
        with self.assertRaisesRegex(ValueError, "unknown model 'missing'.*bn-bpe-58k"):
            Tokenizer.from_pretrained("missing")

    def test_lossless_without_unknowns(self) -> None:
        for name, tok in self.models.items():
            batch = tok.encode_batch(STRESS)
            for text, ids in zip(STRESS, batch, strict=True):
                with self.subTest(model=name, text=text[:20]):
                    self.assertEqual(ids, tok.encode(text))
                    self.assertEqual(tok.count_tokens(text), len(ids))
                    self.assertEqual(tok.decode(ids), normalize(text))
                    self.assertTrue(all(i > EOS_ID for i in ids))

    def test_output_buffer_growth_across_threads(self) -> None:
        tok = self.models["bn-bpe-28k"]
        # Thousands of byte-fallback IDs exceed the initial per-thread buffer.
        texts = ["\U00010000" * n for n in (1, 300, 3000, 10)]
        expected = tok.encode_batch(texts)
        self.assertGreater(len(expected[2]), 4096)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for _ in range(5):
                self.assertEqual(list(pool.map(tok.encode, texts)), expected)

    def test_token_strings_and_vocabulary(self) -> None:
        tok = self.models[DEFAULT_MODEL]
        vocab = tok.get_vocab()
        self.assertEqual(len(vocab), tok.vocab_size)
        self.assertEqual(sorted(vocab.values()), list(range(tok.vocab_size)))
        for token, token_id in vocab.items():
            self.assertEqual(tok.id_to_token(token_id), token)
            self.assertEqual(tok.token_to_id(token), token_id)
        text = "আমি বাংলায় গান গাই। ami banglay gaan gai"
        self.assertEqual("".join(tok.tokenize(text)), text)
        self.assertEqual(tok.tokenize("\U00010000"), ["<0xF0>", "<0x90>", "<0x80>", "<0x80>"])
        self.assertEqual(tok.id_to_token(PAD_ID), "<pad>")
        self.assertEqual(tok.token_to_id(" "), 36)
        self.assertIsNone(tok.token_to_id("no such token here"))
        with self.assertRaises(ValueError):
            tok.id_to_token(tok.vocab_size)
        with self.assertRaises(TypeError):
            tok.id_to_token(True)

    def test_encoder_inputs(self) -> None:
        tok = self.models[DEFAULT_MODEL]
        texts = ["আমি ভালো আছি, তুমি কেমন আছো?", "hi"]
        ids = tok.encode_batch(texts)
        batch = tok(texts, max_length=5, padding=True)
        self.assertEqual(batch["input_ids"][0], [BOS_ID, *ids[0][:3], EOS_ID])
        width = len(batch["input_ids"][0])
        self.assertEqual(
            batch["input_ids"][1], [BOS_ID, *ids[1], EOS_ID] + [PAD_ID] * (width - len(ids[1]) - 2)
        )
        self.assertEqual(
            batch["attention_mask"][1], [1] * (len(ids[1]) + 2) + [0] * (width - len(ids[1]) - 2)
        )
        fixed = tok(texts, max_length=12, padding="max_length")
        self.assertEqual({len(row) for row in fixed["input_ids"]}, {12})
        single = tok("hi", add_special_tokens=False)
        self.assertEqual(single, {"input_ids": ids[1], "attention_mask": [1] * len(ids[1])})
        self.assertEqual(tok.decode(batch["input_ids"][1], skip_special_tokens=True), "hi")
        self.assertEqual(tok.decode_batch(fixed["input_ids"], skip_special_tokens=True)[1], "hi")
        with self.assertRaises(ValueError):
            tok.decode(batch["input_ids"][1])
        with self.assertRaises(ValueError):
            tok(texts, padding="max_length")
        with self.assertRaises(ValueError):
            tok(texts, max_length=1)
        with self.assertRaises(ValueError):
            tok(texts, padding="sideways")  # type: ignore[call-overload]


@unittest.skipUnless(importlib.util.find_spec("tokenizers"), "tokenizers is not installed")
class HuggingFaceExportTests(unittest.TestCase):
    def test_exported_tokenizer_matches_native_ids(self) -> None:
        from tokenizers import Tokenizer as HFTokenizer

        texts = [*STRESS, " a\u00a0\u2003b\u3000c\n", "x\u2028y\u2029z", "ami 🙂 বাংলা"]
        for name in available_models():
            with (
                self.subTest(model=name),
                Tokenizer.from_pretrained(name) as tok,
                tempfile.TemporaryDirectory() as directory,
            ):
                tok.save_huggingface(directory)
                hf = HFTokenizer.from_file(str(Path(directory) / "tokenizer.json"))
                hf.encode_special_tokens = True
                self.assertEqual(hf.get_vocab_size(), tok.vocab_size)
                for text in texts:
                    ids = tok.encode(text)
                    self.assertEqual(hf.encode(text, add_special_tokens=False).ids, ids)
                    self.assertEqual(hf.decode(ids), normalize(text))
                    self.assertEqual(hf.encode(text).ids, tok(text)["input_ids"])

    def test_unigram_export_is_rejected(self) -> None:
        demo = files("bn_tokenizers_embedding").joinpath("data/demo-unigram.json")
        with as_file(demo) as path, Tokenizer(path) as tok, tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(ValueError, "BPE models only"):
                tok.save_huggingface(out)


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *argv: str) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(list(argv)), 0)
        return output.getvalue()

    def test_outputs(self) -> None:
        with Tokenizer.from_pretrained() as tok:
            ids = tok.encode("আমি বাংলা বলি")
        self.assertEqual(self.run_cli("--count", "আমি", "বাংলা", "বলি"), f"{len(ids)}\n")
        self.assertEqual(self.run_cli("--ids", "আমি বাংলা বলি"), " ".join(map(str, ids)) + "\n")
        record = json.loads(self.run_cli("--json", "-m", "bn-bpe-28k", "ami"))
        self.assertEqual("".join(record["tokens"]), "ami")
        self.assertEqual(self.run_cli("--list-models"), "bn-bpe-28k\nbn-bpe-58k\n")


if __name__ == "__main__":
    unittest.main()

"""Hugging Face ``tokenizers`` export for BPE models, built without extra dependencies.

The exported pipeline reproduces native IDs exactly:

- normalizer: NFC, then every Unicode whitespace run becomes one space, then strip;
- pre-tokenizer: split on spaces, each space kept at the start of the next word;
- model: the learned BPE merges in rank order with UTF-8 byte fallback, followed by
  one lowest-priority merge per fused ID, joining the leading space with the
  word's final first token;
- post-processor: ``<s> $A </s>`` when special tokens are requested.
"""

import json

_SPECIAL = ["<pad>", "<unk>", "<s>", "</s>"]
_PIECE_OFFSET = 260

# Unicode White_Space, the set Go's unicode.IsSpace uses.
_WHITESPACE = (
    r"[\t\n\x{0B}\f\r \x{85}\x{A0}\x{1680}\x{2000}-\x{200A}"
    r"\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}]+"
)


def tokenizer_json(tokens: list[str], merges: list[tuple[int, int]], fusion: list[int]) -> str:
    """Serialize a BPE model; ``tokens`` is the full ID-ordered token list."""
    if len(set(tokens)) != len(tokens):
        raise ValueError("token strings are not unique; the model cannot be exported")
    if any(base < _PIECE_OFFSET for base in fusion):
        raise ValueError("space fusion of byte IDs has no Hugging Face equivalent")
    pieces = tokens[_PIECE_OFFSET : len(tokens) - len(fusion)]
    rules = [[pieces[left], pieces[right]] for left, right in merges]
    rules += [[" ", tokens[base]] for base in fusion]
    special = [
        {
            "id": i,
            "content": text,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for i, text in enumerate(_SPECIAL)
    ]
    document = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": special,
        "normalizer": {
            "type": "Sequence",
            "normalizers": [
                {"type": "NFC"},
                {"type": "Replace", "pattern": {"Regex": _WHITESPACE}, "content": " "},
                {"type": "Strip", "strip_left": True, "strip_right": True},
            ],
        },
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"String": " "},
            "behavior": "MergedWithNext",
            "invert": False,
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
            ],
            "special_tokens": {
                text: {"id": text, "ids": [i], "tokens": [text]}
                for i, text in enumerate(_SPECIAL)
                if text in ("<s>", "</s>")
            },
        },
        "decoder": {
            "type": "Sequence",
            "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}],
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": {text: i for i, text in enumerate(tokens)},
            "merges": rules,
        },
    }
    return json.dumps(document, ensure_ascii=False, indent=1)


def tokenizer_config() -> str:
    """``transformers`` settings: literal special-token text stays ordinary text."""
    return json.dumps(
        {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "split_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        },
        indent=1,
    )

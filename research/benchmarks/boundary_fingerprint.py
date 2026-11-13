"""Fingerprint Python API results of the installed package, types included.

Run it under two installs and diff the output to check that a boundary change
keeps every ID, return type and error identical:

    python research/benchmarks/boundary_fingerprint.py --corpus corpus.jsonl > a.json
"""

import argparse
import hashlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path

from banglatok import Tokenizer, available_models

EDGE = [
    "",
    " ",
    "a\x00b",
    "\x00",
    "<unk> <s> </s> ▁ <0x20>",
    "👩🏽‍💻 🙂🇧🇩",
    " বাংলা\tEnglish \n ",
    "ক্‍ষ ক্‌ষ é ড়",
    "বাংলা" * 2000,
    "ami " * 5000,
]


def outcome(action: Callable[[], object]) -> str:
    """repr of the result, or of the exception type and message."""
    try:
        return repr(action())
    except Exception as error:  # noqa: BLE001 - errors are part of the contract.
        return f"raised {type(error).__name__}: {error}"


def cases(
    tok: Tokenizer, corpus: list[str], threads: int | None
) -> dict[str, Callable[[], object]]:
    extra = {} if threads is None else {"num_threads": threads}
    mixed = [corpus[i % len(corpus)] * (1 + i % 7) for i in range(300)] + EDGE
    batches = {"corpus": corpus, "edge": EDGE, "mixed": mixed, "empty": [], "one": ["আমি"]}
    result: dict[str, Callable[[], object]] = {
        "encode": lambda: [tok.encode(t) for t in corpus + EDGE],
        "count_tokens": lambda: [tok.count_tokens(t) for t in corpus + EDGE],
        "tokenize": lambda: [tok.tokenize(t) for t in EDGE],
        "encode_batch/tuple": lambda: tok.encode_batch(tuple(EDGE), **extra),
        "encode_batch/generator": lambda: tok.encode_batch((t for t in EDGE), **extra),
        "encode_batch/64": lambda: [
            tok.encode_batch(corpus[i : i + 64], **extra) for i in range(0, len(corpus), 64)
        ],
        "errors/encode": lambda: [
            outcome(lambda v=v: tok.encode(v)) for v in (None, b"a", 1, "\ud800")
        ],
        "errors/count": lambda: [
            outcome(lambda v=v: tok.count_tokens(v)) for v in (None, b"a", "\ud800")
        ],
        "errors/batch": lambda: [
            outcome(lambda v=v: tok.encode_batch(v, **extra))
            for v in (
                "ab",
                b"ab",
                ["a", None],
                [None, "\ud800"],
                ["\ud800", None],
                ["a", b"b"],
                [1],
                None,
            )
        ],
        "errors/call": lambda: [
            outcome(lambda v=v, kw=kw: tok(v, **kw, **extra))
            for v, kw in [
                (b"ab", {}),
                (["a", None], {}),
                (["\ud800"], {}),
                ("a", {"padding": "x"}),
                ("a", {"max_length": 1}),
                ("a", {"max_length": 0, "add_special_tokens": False}),
                ("a", {"max_length": True}),
                ("a", {"padding": "max_length"}),
                (None, {}),
            ]
        ],
    }
    for name, batch in batches.items():
        result[f"encode_batch/{name}"] = lambda batch=batch: tok.encode_batch(batch, **extra)
        for special in (True, False):
            for max_length in (None, 2, 5, 128):
                for padding in (False, True, "longest", "max_length"):
                    key = f"call/{name}/{special}/{max_length}/{padding}"
                    result[key] = lambda batch=batch, s=special, m=max_length, p=padding: outcome(
                        lambda: tok(batch, add_special_tokens=s, max_length=m, padding=p, **extra)
                    )
    for i, text in enumerate(EDGE + corpus[:20]):
        for padding in (False, True, "max_length"):
            result[f"call/single/{i}/{padding}"] = lambda t=text, p=padding: outcome(
                lambda: tok(t, max_length=16, padding=p, **extra)
            )
    return result


def closed_errors(model: str) -> str:
    closed = Tokenizer.from_pretrained(model)
    closed.close()
    actions: list[Callable[[], object]] = [
        lambda: closed.encode("a"),
        lambda: closed.encode(None),  # type: ignore[arg-type]
        lambda: closed.count_tokens("a"),
        lambda: closed.encode_batch(["a"]),
        lambda: closed.encode_batch("a"),
        lambda: closed(["a"], padding="x"),  # type: ignore[call-overload]
        lambda: closed(["a"]),
        lambda: closed.decode([5]),
    ]
    return repr([outcome(action) for action in actions])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="JSONL with a text field")
    parser.add_argument("--threads", type=int, help="num_threads for batch calls, if supported")
    parser.add_argument("--dump", type=Path, help="also write every result to this directory")
    args = parser.parse_args()
    corpus = [json.loads(line)["text"] for line in args.corpus.read_text("utf-8").splitlines()]
    digests: dict[str, dict[str, str]] = {}
    for model in available_models():
        with Tokenizer.from_pretrained(model) as tok:
            if args.threads and "num_threads" not in inspect.signature(tok.encode_batch).parameters:
                raise SystemExit("installed banglatok has no num_threads option")
            digests[model] = {}
            for name, action in cases(tok, corpus, args.threads).items():
                text = outcome(action)
                digests[model][name] = hashlib.sha256(text.encode()).hexdigest()[:16]
                if args.dump:
                    path = args.dump / model / (name.replace("/", "_") + ".txt")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text, encoding="utf-8")
        digests[model]["errors/closed"] = closed_errors(model)
    print(json.dumps(digests, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

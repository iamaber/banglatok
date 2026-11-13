"""Command line: ``bntokenize "আমি বাংলা বলি"`` or ``python -m bn_tokenizers_embedding``."""

import argparse
import json
import sys
from collections.abc import Sequence

from . import DEFAULT_MODEL, Tokenizer, __version__, available_models


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bntokenize",
        description="Tokenize Bangla, English and Banglish text. Reads standard input "
        "line by line when no text is given.",
    )
    parser.add_argument("text", nargs="*", help="text to tokenize (joined with spaces)")
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"bundled model name or path to a JSON model (default: {DEFAULT_MODEL})",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--ids", action="store_true", help="print token IDs only")
    output.add_argument("--count", action="store_true", help="print the token count only")
    output.add_argument("--json", action="store_true", help="print JSON with tokens and IDs")
    parser.add_argument("--list-models", action="store_true", help="list bundled models")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    if args.list_models:
        print("\n".join(available_models()))
        return 0
    if args.model in available_models():
        tokenizer = Tokenizer.from_pretrained(args.model)
    else:
        tokenizer = Tokenizer(args.model)
    lines = [" ".join(args.text)] if args.text else (line.rstrip("\n") for line in sys.stdin)
    with tokenizer:
        for line in lines:
            if args.count:
                print(tokenizer.count_tokens(line))
                continue
            ids = tokenizer.encode(line)
            if args.ids:
                print(" ".join(map(str, ids)))
            elif args.json:
                tokens = [tokenizer.id_to_token(i) for i in ids]
                print(json.dumps({"ids": ids, "tokens": tokens}, ensure_ascii=False))
            else:
                print(" | ".join(tokenizer.id_to_token(i) for i in ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

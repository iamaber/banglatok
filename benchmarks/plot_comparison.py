"""Draw the README benchmark chart from a compare_popular.py result file.

Writes a standalone SVG with no plotting dependency:

    python benchmarks/plot_comparison.py \
        benchmarks/results/popular-comparison.json assets/benchmark.svg
"""

import argparse
import json
from html import escape
from pathlib import Path

CORPUS = "Unseen-source BanglishRev reviews"
ROWS = [
    ("banglatok bn-bpe-58k", "banglatok bn-bpe-58k"),
    ("SentencePiece Unigram 57,925, same training data", "SentencePiece Unigram, same data"),
    ("tiktoken o200k_base", "tiktoken o200k_base (GPT-4o)"),
    ("MuRIL (Google)", "MuRIL"),
    ("BanglaBERT (csebuetnlp)", "BanglaBERT"),
    ("XLM-R / mE5 / BGE-M3", "XLM-R / mE5 / BGE-M3"),
    ("Qwen3", "Qwen3"),
]
LABEL_WIDTH = 250
PANEL_WIDTH = 270
GAP = 50
TOP = 64
ROW = 30
BAR = 18


def panel(x: float, title: str, values: list[float], digits: int) -> list[str]:
    largest = max(values)
    parts = [f'<text class="title" x="{x}" y="{TOP - 24}">{escape(title)}</text>']
    for i, value in enumerate(values):
        y = TOP + i * ROW
        width = PANEL_WIDTH * value / largest
        kind = "ours" if i == 0 else "other"
        parts.append(
            f'<rect class="{kind}" x="{x}" y="{y}" width="{width:.1f}" height="{BAR}" rx="3"/>'
        )
        parts.append(
            f'<text class="value" x="{x + width + 6:.1f}" y="{y + BAR - 4}">'
            f"{value:.{digits}f}</text>"
        )
    return parts


def chart(result: dict) -> str:
    candidates = result["corpora"][CORPUS]["candidates"]
    tokens = [candidates[name]["quality"]["all"]["tokens_per_word"] for name, _ in ROWS]
    speed = [candidates[name]["speed"]["single_MB_per_second"] for name, _ in ROWS]
    left = LABEL_WIDTH
    right = left + PANEL_WIDTH + GAP + 40
    width = right + PANEL_WIDTH + 60
    height = TOP + len(ROWS) * ROW + 36
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}" role="img"'
        ' aria-label="Tokens per word and single-thread speed of banglatok and other tokenizers">',
        "<style>",
        "text { font: 13px -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;"
        " fill: #1f2328; }",
        ".title { font-weight: 600; } .note { font-size: 11px; fill: #59636e; }",
        ".ours { fill: #0969da; } .other { fill: #afb8c1; }",
        "</style>",
        f'<rect width="{width}" height="{height}" rx="8" fill="#ffffff" stroke="#d1d9e0"/>',
    ]
    for i, (_, label) in enumerate(ROWS):
        y = TOP + i * ROW + BAR - 4
        weight = ' font-weight="600"' if i == 0 else ""
        parts.append(
            f'<text x="{left - 12}" y="{y}" text-anchor="end"{weight}>{escape(label)}</text>'
        )
    parts += panel(left, "Tokens per word (lower is better)", tokens, 2)
    parts += panel(right, "MB/s on one thread (higher is better)", speed, 1)
    reviews = result["corpora"][CORPUS]["records"]
    parts.append(
        f'<text class="note" x="{left}" y="{height - 12}">{reviews:,} unseen BanglishRev reviews,'
        f" {escape(result['cpu_model'])}, Python {escape(result['python'])}."
        " Source: benchmarks/POPULAR_COMPARISON.md</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(chart(json.loads(args.result.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()

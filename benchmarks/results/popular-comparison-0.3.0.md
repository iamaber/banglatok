# banglatok 0.3.0 comparison result

The benchmark ran on 25 November 2026 with the published
`banglatok==0.3.0` wheel, Python 3.12.13 and an Apple M5 on macOS 27.2.1. Each
number is the median of five shuffled repeats. Single-threaded tests used
individual calls and batches of 64. Multi-core tests used batches of 1,024 on
10 threads.

- Harness SHA-256: `a65f5df27c923e38b39c3d6298be94c2dc737ac30bf9d8d9f1f56e7a33ea3c92`
- Frozen validation SHA-256: `047e0acb94d621508467fcd0e7b3369204f7e040d4b9144b58614faac9f1cc2d`
- Ordered validation text SHA-256: `88b57174621deaf7538367891f377316b6fe8ebe5c1203e0d54ff81ff5ec33bd`
- BanglishRev source SHA-256: `553b8e1eab436e71fb29d7159acf508f0ddc34778f6eab8a9d710ba655497f72`
- Ordered review text SHA-256: `89ebf28fe456c073b1d2c6055dbf48296bfd5e65a5e789ea2494f889156be879`
- Versions: SentencePiece 0.2.3, tiktoken 0.14.0, tokenizers 0.23.3,
  Transformers 5.22.2
- README chart input: [popular-comparison-0.3.0-chart.json](popular-comparison-0.3.0-chart.json)

Speed is MB/s for individual calls / batches of 64. Tokens per word,
losslessness and unknown counts are deterministic; speed is the median of five
passes.

## Unseen-source BanglishRev reviews

10,000 texts, 1,297,966 UTF-8 bytes.

| Tokenizer | Vocab | Tokens/word | Round trip | Texts with UNK | MB/s |
|---|---:|---:|---:|---:|---:|
| banglatok `bn-bpe-58k` | 57,925 | 1.379 | 100.00% | 0 | 102.9 / 124.0 |
| banglatok `bn-bpe-28k-v2` | 27,760 | 1.570 | 100.00% | 0 | 102.5 / 123.5 |
| banglatok `bn-bpe-28k` | 27,760 | 1.587 | 100.00% | 0 | 103.1 / 121.1 |
| BNLP `bn_spm` | 50,000 | 1.825 | 80.14% | 1,932 | 40.5 / 39.4 |
| SentencePiece Unigram 57,925, original data | 57,925 | 1.410 | 100.00% | 0 | 43.6 / 41.3 |
| SentencePiece BPE 57,925, original data | 57,925 | 1.407 | 100.00% | 0 | 17.5 / 17.6 |
| SentencePiece Unigram 27,760, original data | 27,760 | 1.572 | 100.00% | 0 | 45.6 / 44.1 |
| SentencePiece BPE 27,760, original data | 27,760 | 1.578 | 100.00% | 0 | 16.6 / 16.6 |
| tiktoken `o200k_base` | 200,019 | 1.780 | 100.00% | 0 | 36.9 / 12.2 |
| tiktoken `cl100k_base` | 100,277 | 4.027 | 100.00% | 0 | 24.6 / 10.0 |
| BanglaBERT | 32,000 | 1.771 | 51.10% | 1,110 | 9.0 / 9.1 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.507 | 0.32% | 7,410 | 6.8 / 6.8 |
| MuRIL | 197,258 | 1.498 | 47.94% | 1,617 | 8.7 / 8.8 |
| mBERT cased | 119,547 | 2.122 | 47.97% | 1,706 | 8.0 / 8.1 |
| XLM-R / mE5 / BGE-M3 | 250,002 | 1.740 | 98.16% | 109 | 6.5 / 6.5 |
| Qwen3 | 151,669 | 3.588 | 100.00% | 0 | 6.0 / 6.0 |

| Bundled model | Bengali script | Latin script | Mixed script | 10-thread MB/s |
|---|---:|---:|---:|---:|
| `bn-bpe-58k` | 1.381 | 1.373 | 1.418 | 218.7 |
| `bn-bpe-28k-v2` | 1.595 | 1.544 | 1.627 | 192.8 |
| `bn-bpe-28k` | 1.603 | 1.570 | 1.633 | 206.8 |

## Frozen validation

14,779 texts, 3,013,257 UTF-8 bytes.

| Tokenizer | Vocab | Tokens/word | Round trip | Texts with UNK | MB/s |
|---|---:|---:|---:|---:|---:|
| banglatok `bn-bpe-58k` | 57,925 | 1.373 | 100.00% | 0 | 137.2 / 159.3 |
| banglatok `bn-bpe-28k-v2` | 27,760 | 1.553 | 100.00% | 0 | 135.9 / 154.6 |
| banglatok `bn-bpe-28k` | 27,760 | 1.563 | 100.00% | 0 | 135.7 / 153.0 |
| BNLP `bn_spm` | 50,000 | 1.861 | 87.95% | 647 | 43.0 / 42.9 |
| SentencePiece Unigram 57,925, original data | 57,925 | 1.384 | 100.00% | 0 | 45.4 / 45.1 |
| SentencePiece BPE 57,925, original data | 57,925 | 1.381 | 100.00% | 0 | 19.5 / 19.6 |
| SentencePiece Unigram 27,760, original data | 27,760 | 1.495 | 100.00% | 0 | 48.5 / 48.3 |
| SentencePiece BPE 27,760, original data | 27,760 | 1.512 | 100.00% | 0 | 18.4 / 18.6 |
| tiktoken `o200k_base` | 200,019 | 2.153 | 100.00% | 0 | 33.0 / 15.7 |
| tiktoken `cl100k_base` | 100,277 | 5.430 | 100.00% | 0 | 24.2 / 12.9 |
| BanglaBERT | 32,000 | 1.791 | 50.35% | 1,579 | 10.5 / 10.6 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.698 | 0.64% | 11,971 | 7.8 / 7.8 |
| MuRIL | 197,258 | 1.629 | 50.19% | 352 | 9.9 / 10.1 |
| mBERT cased | 119,547 | 2.481 | 49.89% | 1,641 | 8.8 / 8.9 |
| XLM-R / mE5 / BGE-M3 | 250,002 | 2.035 | 91.60% | 20 | 7.7 / 7.8 |
| Qwen3 | 151,669 | 4.852 | 100.00% | 0 | 6.6 / 6.5 |

| Bundled model | Bangla | Banglish | Code-mixed | English | 10-thread MB/s |
|---|---:|---:|---:|---:|---:|
| `bn-bpe-58k` | 1.423 | 1.356 | 1.397 | 1.121 | 325.9 |
| `bn-bpe-28k-v2` | 1.619 | 1.530 | 1.598 | 1.224 | 309.3 |
| `bn-bpe-28k` | 1.637 | 1.525 | 1.586 | 1.231 | 308.7 |

## Command

```sh
TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 python benchmarks/compare_popular.py \
  --sentencepiece models/comparison/bnlp.model \
  --sentencepiece models/comparison/sentencepiece-unigram-57925.model \
  --sentencepiece models/comparison/sentencepiece-bpe-57925.model \
  --sentencepiece models/comparison/sentencepiece-unigram-27760.model \
  --sentencepiece models/comparison/sentencepiece-bpe-27760.model \
  --cpu-model "Apple M5"
```

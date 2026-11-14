
### Frozen v1 validation (14,779 texts)

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| banglatok bn-bpe-58k | 57,925 | 1.373 | 100.00% | 0 | 75.1 | 86.6 |
| banglatok bn-bpe-28k | 27,760 | 1.563 | 100.00% | 0 | 76.1 | 88.4 |
| banglatok bn-bpe-28k-v2 | 25,147 | 1.746 | 100.00% | 0 | 70.8 | 81.8 |
| banglatok bn-bpe-28k-balanced-v2 | 25,150 | 1.781 | 100.00% | 0 | 73.1 | 83.5 |
| banglatok bn-bpe-58k-v2 | 56,081 | 1.430 | 100.00% | 0 | 72.2 | 83.2 |
| banglatok bn-bpe-58k-balanced-v2 | 55,150 | 1.468 | 100.00% | 0 | 73.1 | 84.0 |

Tokens per word by bucket (Frozen v1 validation):

| Tokenizer | bangla | banglish | code-mixed | english |
|---|---:|---:|---:|---:|
| banglatok bn-bpe-58k | 1.423 | 1.356 | 1.397 | 1.121 |
| banglatok bn-bpe-28k | 1.637 | 1.525 | 1.586 | 1.231 |
| banglatok bn-bpe-28k-v2 | 1.816 | 1.683 | 1.734 | 1.528 |
| banglatok bn-bpe-28k-balanced-v2 | 1.900 | 1.650 | 1.667 | 1.515 |
| banglatok bn-bpe-58k-v2 | 1.469 | 1.408 | 1.445 | 1.259 |
| banglatok bn-bpe-58k-balanced-v2 | 1.538 | 1.402 | 1.431 | 1.266 |

### Held-out data/processed/expanded-v2/extra-validation.jsonl (50,253 texts)

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| banglatok bn-bpe-58k | 57,925 | 1.661 | 100.00% | 0 | 62.8 | 67.4 |
| banglatok bn-bpe-28k | 27,760 | 1.912 | 100.00% | 0 | 63.5 | 68.7 |
| banglatok bn-bpe-28k-v2 | 25,147 | 1.892 | 100.00% | 0 | 76.0 | 83.6 |
| banglatok bn-bpe-28k-balanced-v2 | 25,150 | 1.916 | 100.00% | 0 | 77.5 | 82.8 |
| banglatok bn-bpe-58k-v2 | 56,081 | 1.523 | 100.00% | 0 | 80.3 | 85.9 |
| banglatok bn-bpe-58k-balanced-v2 | 55,150 | 1.556 | 100.00% | 0 | 78.8 | 84.5 |

Tokens per word by bucket (Held-out data/processed/expanded-v2/extra-validation.jsonl):

| Tokenizer | bangla | english |
|---|---:|---:|
| banglatok bn-bpe-58k | 1.437 | 1.770 |
| banglatok bn-bpe-28k | 1.666 | 2.031 |
| banglatok bn-bpe-28k-v2 | 1.816 | 1.928 |
| banglatok bn-bpe-28k-balanced-v2 | 1.931 | 1.909 |
| banglatok bn-bpe-58k-v2 | 1.459 | 1.553 |
| banglatok bn-bpe-58k-balanced-v2 | 1.544 | 1.561 |



### Frozen v1 test (16,712 texts)

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| banglatok bn-bpe-58k | 57,925 | 1.361 | 100.00% | 0 | 76.8 | 87.6 |
| banglatok bn-bpe-28k | 27,760 | 1.543 | 100.00% | 0 | 78.0 | 89.1 |
| banglatok bn-bpe-58k-v2-m1 | 57,252 | 1.387 | 100.00% | 0 | 75.9 | 87.6 |
| banglatok bn-bpe-58k-v2-m2 | 57,442 | 1.386 | 100.00% | 0 | 75.3 | 86.7 |
| banglatok bn-bpe-28k-v2-m2 | 26,978 | 1.593 | 100.00% | 0 | 77.7 | 89.8 |

Tokens per word by bucket (Frozen v1 test):

| Tokenizer | bangla | banglish | code-mixed | english |
|---|---:|---:|---:|---:|
| banglatok bn-bpe-58k | 1.406 | 1.339 | 1.406 | 1.121 |
| banglatok bn-bpe-28k | 1.606 | 1.505 | 1.576 | 1.235 |
| banglatok bn-bpe-58k-v2-m1 | 1.428 | 1.367 | 1.460 | 1.160 |
| banglatok bn-bpe-58k-v2-m2 | 1.430 | 1.360 | 1.446 | 1.155 |
| banglatok bn-bpe-28k-v2-m2 | 1.658 | 1.547 | 1.633 | 1.302 |

### Held-out data/processed/expanded-v2/extra-test.jsonl (49,123 texts)

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| banglatok bn-bpe-58k | 57,925 | 1.657 | 100.00% | 0 | 63.0 | 67.9 |
| banglatok bn-bpe-28k | 27,760 | 1.906 | 100.00% | 0 | 65.3 | 70.0 |
| banglatok bn-bpe-58k-v2-m1 | 57,252 | 1.533 | 100.00% | 0 | 74.4 | 81.0 |
| banglatok bn-bpe-58k-v2-m2 | 57,442 | 1.526 | 100.00% | 0 | 75.0 | 80.2 |
| banglatok bn-bpe-28k-v2-m2 | 26,978 | 1.798 | 100.00% | 0 | 77.5 | 84.1 |

Tokens per word by bucket (Held-out data/processed/expanded-v2/extra-test.jsonl):

| Tokenizer | bangla | english |
|---|---:|---:|
| banglatok bn-bpe-58k | 1.442 | 1.769 |
| banglatok bn-bpe-28k | 1.672 | 2.029 |
| banglatok bn-bpe-58k-v2-m1 | 1.434 | 1.585 |
| banglatok bn-bpe-58k-v2-m2 | 1.433 | 1.575 |
| banglatok bn-bpe-28k-v2-m2 | 1.679 | 1.860 |

### Unseen-source BanglishRev reviews (10,000 texts)

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| banglatok bn-bpe-58k | 57,925 | 1.379 | 100.00% | 0 | 60.7 | 72.0 |
| banglatok bn-bpe-28k | 27,760 | 1.587 | 100.00% | 0 | 62.0 | 72.0 |
| banglatok bn-bpe-58k-v2-m1 | 57,252 | 1.420 | 100.00% | 0 | 60.0 | 70.9 |
| banglatok bn-bpe-58k-v2-m2 | 57,442 | 1.413 | 100.00% | 0 | 60.8 | 72.1 |
| banglatok bn-bpe-28k-v2-m2 | 26,978 | 1.614 | 100.00% | 0 | 62.6 | 75.6 |

Tokens per word by bucket (Unseen-source BanglishRev reviews):

| Tokenizer | bengali script | latin script | mixed script |
|---|---:|---:|---:|
| banglatok bn-bpe-58k | 1.381 | 1.373 | 1.418 |
| banglatok bn-bpe-28k | 1.603 | 1.570 | 1.633 |
| banglatok bn-bpe-58k-v2-m1 | 1.458 | 1.385 | 1.478 |
| banglatok bn-bpe-58k-v2-m2 | 1.449 | 1.380 | 1.466 |
| banglatok bn-bpe-28k-v2-m2 | 1.658 | 1.574 | 1.680 |


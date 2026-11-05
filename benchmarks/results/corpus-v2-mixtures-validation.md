
### Frozen v1 validation (14,779 texts)

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| banglatok bn-bpe-58k | 57,925 | 1.373 | 100.00% | 0 | 66.4 | 79.0 |
| banglatok bn-bpe-28k | 27,760 | 1.563 | 100.00% | 0 | 69.3 | 78.5 |
| banglatok bn-bpe-28k-v2-m1 | 26,554 | 1.633 | 100.00% | 0 | 66.7 | 78.2 |
| banglatok bn-bpe-28k-balanced-v2-m1 | 26,508 | 1.671 | 100.00% | 0 | 68.2 | 78.6 |
| banglatok bn-bpe-58k-v2-m1 | 57,252 | 1.399 | 100.00% | 0 | 66.1 | 76.6 |
| banglatok bn-bpe-58k-balanced-v2-m1 | 56,064 | 1.435 | 100.00% | 0 | 66.4 | 75.5 |
| banglatok bn-bpe-28k-v2-m2 | 26,978 | 1.614 | 100.00% | 0 | 70.4 | 77.6 |
| banglatok bn-bpe-28k-balanced-v2-m2 | 26,882 | 1.650 | 100.00% | 0 | 69.9 | 78.1 |
| banglatok bn-bpe-58k-v2-m2 | 57,442 | 1.398 | 100.00% | 0 | 64.6 | 74.5 |
| banglatok bn-bpe-58k-balanced-v2-m2 | 56,160 | 1.435 | 100.00% | 0 | 64.8 | 74.7 |

Tokens per word by bucket (Frozen v1 validation):

| Tokenizer | bangla | banglish | code-mixed | english |
|---|---:|---:|---:|---:|
| banglatok bn-bpe-58k | 1.423 | 1.356 | 1.397 | 1.121 |
| banglatok bn-bpe-28k | 1.637 | 1.525 | 1.586 | 1.231 |
| banglatok bn-bpe-28k-v2-m1 | 1.701 | 1.603 | 1.649 | 1.321 |
| banglatok bn-bpe-28k-balanced-v2-m1 | 1.785 | 1.571 | 1.597 | 1.316 |
| banglatok bn-bpe-58k-v2-m1 | 1.444 | 1.387 | 1.435 | 1.156 |
| banglatok bn-bpe-58k-balanced-v2-m1 | 1.508 | 1.384 | 1.417 | 1.161 |
| banglatok bn-bpe-28k-v2-m2 | 1.687 | 1.571 | 1.616 | 1.300 |
| banglatok bn-bpe-28k-balanced-v2-m2 | 1.765 | 1.548 | 1.571 | 1.295 |
| banglatok bn-bpe-58k-v2-m2 | 1.447 | 1.380 | 1.425 | 1.151 |
| banglatok bn-bpe-58k-balanced-v2-m2 | 1.513 | 1.377 | 1.409 | 1.157 |

### Held-out data/processed/expanded-v2/extra-validation.jsonl (50,253 texts)

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| banglatok bn-bpe-58k | 57,925 | 1.661 | 100.00% | 0 | 56.8 | 61.1 |
| banglatok bn-bpe-28k | 27,760 | 1.912 | 100.00% | 0 | 58.6 | 61.9 |
| banglatok bn-bpe-28k-v2-m1 | 26,554 | 1.833 | 100.00% | 0 | 68.1 | 73.7 |
| banglatok bn-bpe-28k-balanced-v2-m1 | 26,508 | 1.856 | 100.00% | 0 | 68.9 | 74.8 |
| banglatok bn-bpe-58k-v2-m1 | 57,252 | 1.535 | 100.00% | 0 | 65.9 | 71.7 |
| banglatok bn-bpe-58k-balanced-v2-m1 | 56,064 | 1.566 | 100.00% | 0 | 67.8 | 73.0 |
| banglatok bn-bpe-28k-v2-m2 | 26,978 | 1.801 | 100.00% | 0 | 68.9 | 77.1 |
| banglatok bn-bpe-28k-balanced-v2-m2 | 26,882 | 1.822 | 100.00% | 0 | 70.2 | 75.9 |
| banglatok bn-bpe-58k-v2-m2 | 57,442 | 1.528 | 100.00% | 0 | 68.3 | 72.8 |
| banglatok bn-bpe-58k-balanced-v2-m2 | 56,160 | 1.559 | 100.00% | 0 | 65.9 | 74.0 |

Tokens per word by bucket (Held-out data/processed/expanded-v2/extra-validation.jsonl):

| Tokenizer | bangla | english |
|---|---:|---:|
| banglatok bn-bpe-58k | 1.437 | 1.770 |
| banglatok bn-bpe-28k | 1.666 | 2.031 |
| banglatok bn-bpe-28k-v2-m1 | 1.699 | 1.897 |
| banglatok bn-bpe-28k-balanced-v2-m1 | 1.806 | 1.880 |
| banglatok bn-bpe-58k-v2-m1 | 1.430 | 1.586 |
| banglatok bn-bpe-58k-balanced-v2-m1 | 1.511 | 1.593 |
| banglatok bn-bpe-28k-v2-m2 | 1.674 | 1.862 |
| banglatok bn-bpe-28k-balanced-v2-m2 | 1.776 | 1.844 |
| banglatok bn-bpe-58k-v2-m2 | 1.429 | 1.575 |
| banglatok bn-bpe-58k-balanced-v2-m2 | 1.509 | 1.583 |


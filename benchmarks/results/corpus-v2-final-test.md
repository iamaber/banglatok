
### Frozen v1 test (16,712 texts)

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| banglatok bn-bpe-58k | 57,925 | 1.361 | 100.00% | 0 | 77.0 | 89.4 |
| banglatok bn-bpe-28k | 27,760 | 1.543 | 100.00% | 0 | 78.8 | 91.1 |
| banglatok bn-bpe-28k-v2 | 25,147 | 1.723 | 100.00% | 0 | 73.9 | 84.9 |
| banglatok bn-bpe-28k-balanced-v2 | 25,150 | 1.756 | 100.00% | 0 | 73.6 | 83.4 |
| banglatok bn-bpe-58k-v2 | 56,081 | 1.416 | 100.00% | 0 | 75.4 | 87.1 |
| banglatok bn-bpe-58k-balanced-v2 | 55,150 | 1.452 | 100.00% | 0 | 75.8 | 87.5 |
| SentencePiece Unigram 27,760, same training data | 27,760 | 1.697 | 100.00% | 0 | 49.8 | 49.3 |
| SentencePiece BPE 27,760, same training data | 27,760 | 1.726 | 100.00% | 0 | 19.1 | 18.9 |
| SentencePiece Unigram 57,925, same training data | 57,925 | 1.480 | 100.00% | 0 | 46.5 | 46.2 |
| SentencePiece BPE 57,925, same training data | 57,925 | 1.491 | 100.00% | 0 | 19.9 | 19.6 |
| tiktoken o200k_base | 200,019 | 2.135 | 100.00% | 0 | 33.1 | 14.8 |
| tiktoken cl100k_base | 100,277 | 5.461 | 100.00% | 0 | 24.3 | 11.9 |
| BanglaBERT (csebuetnlp) | 32,000 | 1.759 | 48.15% | 2,135 | 10.5 | 10.5 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.699 | 0.75% | 13,069 | 7.8 | 7.7 |
| MuRIL (Google) | 197,258 | 1.619 | 48.38% | 384 | 9.8 | 9.9 |
| mBERT cased | 119,547 | 2.477 | 47.70% | 2,189 | 8.8 | 8.7 |
| XLM-R / mE5 / BGE-M3 | 250,002 | 2.025 | 89.43% | 23 | 7.5 | 7.5 |
| Qwen3 | 151,669 | 4.870 | 100.00% | 0 | 6.4 | 6.4 |

Tokens per word by bucket (Frozen v1 test):

| Tokenizer | bangla | banglish | code-mixed | english |
|---|---:|---:|---:|---:|
| banglatok bn-bpe-58k | 1.406 | 1.339 | 1.406 | 1.121 |
| banglatok bn-bpe-28k | 1.606 | 1.505 | 1.576 | 1.235 |
| banglatok bn-bpe-28k-v2 | 1.785 | 1.659 | 1.770 | 1.521 |
| banglatok bn-bpe-28k-balanced-v2 | 1.863 | 1.623 | 1.698 | 1.508 |
| banglatok bn-bpe-58k-v2 | 1.453 | 1.387 | 1.479 | 1.259 |
| banglatok bn-bpe-58k-balanced-v2 | 1.515 | 1.382 | 1.458 | 1.265 |
| SentencePiece Unigram 27,760, same training data | 1.648 | 1.851 | 1.926 | 1.421 |
| SentencePiece BPE 27,760, same training data | 1.681 | 1.879 | 1.951 | 1.430 |
| SentencePiece Unigram 57,925, same training data | 1.448 | 1.573 | 1.680 | 1.333 |
| SentencePiece BPE 57,925, same training data | 1.459 | 1.584 | 1.674 | 1.334 |
| tiktoken o200k_base | 2.434 | 1.824 | 2.072 | 1.188 |
| tiktoken cl100k_base | 7.909 | 2.018 | 4.419 | 1.243 |
| BanglaBERT (csebuetnlp) | 1.453 | 2.324 | 2.005 | 1.753 |
| Bangla-BERT (sagorsarker) | 2.023 | 1.205 | 1.564 | 1.289 |
| MuRIL (Google) | 1.494 | 1.921 | 1.862 | 1.315 |
| mBERT cased | 2.879 | 2.008 | 2.512 | 1.382 |
| XLM-R / mE5 / BGE-M3 | 2.211 | 1.845 | 2.007 | 1.380 |
| Qwen3 | 6.908 | 2.032 | 3.984 | 1.244 |

### Held-out data/processed/expanded-v2/extra-test.jsonl (49,123 texts)

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| banglatok bn-bpe-58k | 57,925 | 1.657 | 100.00% | 0 | 61.6 | 66.4 |
| banglatok bn-bpe-28k | 27,760 | 1.906 | 100.00% | 0 | 64.1 | 69.8 |
| banglatok bn-bpe-28k-v2 | 25,147 | 1.890 | 100.00% | 0 | 77.6 | 85.2 |
| banglatok bn-bpe-28k-balanced-v2 | 25,150 | 1.917 | 100.00% | 0 | 77.4 | 82.9 |
| banglatok bn-bpe-58k-v2 | 56,081 | 1.521 | 100.00% | 0 | 81.8 | 88.5 |
| banglatok bn-bpe-58k-balanced-v2 | 55,150 | 1.557 | 100.00% | 0 | 81.6 | 85.4 |
| SentencePiece Unigram 27,760, same training data | 27,760 | 1.577 | 100.00% | 0 | 45.6 | 47.1 |
| SentencePiece BPE 27,760, same training data | 27,760 | 1.605 | 100.00% | 0 | 14.8 | 14.9 |
| SentencePiece Unigram 57,925, same training data | 57,925 | 1.404 | 100.00% | 0 | 42.9 | 45.7 |
| SentencePiece BPE 57,925, same training data | 57,925 | 1.415 | 100.00% | 0 | 15.4 | 15.3 |
| tiktoken o200k_base | 200,019 | 1.767 | 100.00% | 0 | 36.3 | 21.2 |
| tiktoken cl100k_base | 100,277 | 3.818 | 100.00% | 0 | 25.6 | 15.7 |
| BanglaBERT (csebuetnlp) | 32,000 | 1.968 | 16.35% | 9,553 | 9.4 | 9.2 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.525 | 0.00% | 43,722 | 7.0 | 7.0 |
| MuRIL (Google) | 197,258 | 1.426 | 18.01% | 1,100 | 9.5 | 9.2 |
| mBERT cased | 119,547 | 1.947 | 17.57% | 6,336 | 8.6 | 8.5 |
| XLM-R / mE5 / BGE-M3 | 250,002 | 1.767 | 99.08% | 46 | 5.9 | 5.9 |
| Qwen3 | 151,669 | 3.497 | 100.00% | 0 | 5.8 | 5.7 |

Tokens per word by bucket (Held-out data/processed/expanded-v2/extra-test.jsonl):

| Tokenizer | bangla | english |
|---|---:|---:|
| banglatok bn-bpe-58k | 1.442 | 1.769 |
| banglatok bn-bpe-28k | 1.672 | 2.029 |
| banglatok bn-bpe-28k-v2 | 1.822 | 1.925 |
| banglatok bn-bpe-28k-balanced-v2 | 1.937 | 1.907 |
| banglatok bn-bpe-58k-v2 | 1.462 | 1.552 |
| banglatok bn-bpe-58k-balanced-v2 | 1.549 | 1.561 |
| SentencePiece Unigram 27,760, same training data | 1.616 | 1.556 |
| SentencePiece BPE 27,760, same training data | 1.661 | 1.576 |
| SentencePiece Unigram 57,925, same training data | 1.429 | 1.392 |
| SentencePiece BPE 57,925, same training data | 1.445 | 1.400 |
| tiktoken o200k_base | 2.483 | 1.393 |
| tiktoken cl100k_base | 8.401 | 1.417 |
| BanglaBERT (csebuetnlp) | 1.433 | 2.248 |
| Bangla-BERT (sagorsarker) | 2.106 | 1.222 |
| MuRIL (Google) | 1.459 | 1.409 |
| mBERT cased | 2.954 | 1.419 |
| XLM-R / mE5 / BGE-M3 | 2.254 | 1.512 |
| Qwen3 | 7.341 | 1.484 |

### Unseen-source BanglishRev reviews (10,000 texts)

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| banglatok bn-bpe-58k | 57,925 | 1.379 | 100.00% | 0 | 46.6 | 57.1 |
| banglatok bn-bpe-28k | 27,760 | 1.587 | 100.00% | 0 | 47.2 | 58.2 |
| banglatok bn-bpe-28k-v2 | 25,147 | 1.750 | 100.00% | 0 | 46.0 | 58.2 |
| banglatok bn-bpe-28k-balanced-v2 | 25,150 | 1.736 | 100.00% | 0 | 46.5 | 56.0 |
| banglatok bn-bpe-58k-v2 | 56,081 | 1.457 | 100.00% | 0 | 46.9 | 58.3 |
| banglatok bn-bpe-58k-balanced-v2 | 55,150 | 1.463 | 100.00% | 0 | 46.9 | 58.3 |
| SentencePiece Unigram 27,760, same training data | 27,760 | 1.677 | 100.00% | 0 | 34.7 | 36.2 |
| SentencePiece BPE 27,760, same training data | 27,760 | 1.689 | 100.00% | 0 | 12.9 | 13.0 |
| SentencePiece Unigram 57,925, same training data | 57,925 | 1.463 | 100.00% | 0 | 34.9 | 35.2 |
| SentencePiece BPE 57,925, same training data | 57,925 | 1.467 | 100.00% | 0 | 13.6 | 13.5 |
| tiktoken o200k_base | 200,019 | 1.780 | 100.00% | 0 | 30.0 | 9.8 |
| tiktoken cl100k_base | 100,277 | 4.027 | 100.00% | 0 | 18.7 | 8.0 |
| BanglaBERT (csebuetnlp) | 32,000 | 1.771 | 51.10% | 1,110 | 7.2 | 7.3 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.507 | 0.32% | 7,410 | 5.4 | 5.3 |
| MuRIL (Google) | 197,258 | 1.498 | 47.94% | 1,617 | 6.7 | 6.7 |
| mBERT cased | 119,547 | 2.122 | 47.97% | 1,706 | 6.2 | 6.3 |
| XLM-R / mE5 / BGE-M3 | 250,002 | 1.740 | 98.16% | 109 | 4.9 | 5.0 |
| Qwen3 | 151,669 | 3.588 | 100.00% | 0 | 4.6 | 4.6 |

Tokens per word by bucket (Unseen-source BanglishRev reviews):

| Tokenizer | bengali script | latin script | mixed script |
|---|---:|---:|---:|
| banglatok bn-bpe-58k | 1.381 | 1.373 | 1.418 |
| banglatok bn-bpe-28k | 1.603 | 1.570 | 1.633 |
| banglatok bn-bpe-28k-v2 | 1.803 | 1.706 | 1.808 |
| banglatok bn-bpe-28k-balanced-v2 | 1.796 | 1.686 | 1.800 |
| banglatok bn-bpe-58k-v2 | 1.499 | 1.421 | 1.514 |
| banglatok bn-bpe-58k-balanced-v2 | 1.515 | 1.419 | 1.525 |
| SentencePiece Unigram 27,760, same training data | 1.725 | 1.636 | 1.733 |
| SentencePiece BPE 27,760, same training data | 1.752 | 1.637 | 1.749 |
| SentencePiece Unigram 57,925, same training data | 1.493 | 1.435 | 1.509 |
| SentencePiece BPE 57,925, same training data | 1.514 | 1.427 | 1.515 |
| tiktoken o200k_base | 2.254 | 1.420 | 2.050 |
| tiktoken cl100k_base | 7.279 | 1.519 | 6.077 |
| BanglaBERT (csebuetnlp) | 1.343 | 2.099 | 1.515 |
| Bangla-BERT (sagorsarker) | 1.866 | 1.234 | 1.705 |
| MuRIL (Google) | 1.499 | 1.499 | 1.486 |
| mBERT cased | 2.805 | 1.604 | 2.500 |
| XLM-R / mE5 / BGE-M3 | 2.024 | 1.520 | 1.927 |
| Qwen3 | 6.282 | 1.504 | 5.323 |


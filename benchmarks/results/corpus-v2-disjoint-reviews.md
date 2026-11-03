
### Unseen-source BanglishRev reviews (10,000 texts)

| Tokenizer | Vocab | Tokens/word | Lossless round trip | Texts with UNK | Single MB/s | Batch MB/s |
|---|---:|---:|---:|---:|---:|---:|
| banglatok bn-bpe-58k | 57,925 | 1.375 | 100.00% | 0 | 39.2 | 51.5 |
| banglatok bn-bpe-28k | 27,760 | 1.582 | 100.00% | 0 | 43.7 | 52.9 |
| banglatok bn-bpe-28k-v2 | 25,147 | 1.744 | 100.00% | 0 | 42.4 | 51.4 |
| banglatok bn-bpe-28k-balanced-v2 | 25,150 | 1.728 | 100.00% | 0 | 40.7 | 49.7 |
| banglatok bn-bpe-58k-v2 | 56,081 | 1.454 | 100.00% | 0 | 44.2 | 51.6 |
| banglatok bn-bpe-58k-balanced-v2 | 55,150 | 1.459 | 100.00% | 0 | 43.4 | 52.8 |
| SentencePiece Unigram 27,760, same training data | 27,760 | 1.674 | 100.00% | 0 | 32.2 | 28.6 |
| SentencePiece BPE 27,760, same training data | 27,760 | 1.685 | 100.00% | 0 | 12.0 | 11.7 |
| SentencePiece Unigram 57,925, same training data | 57,925 | 1.459 | 100.00% | 0 | 30.8 | 31.0 |
| SentencePiece BPE 57,925, same training data | 57,925 | 1.462 | 100.00% | 0 | 12.3 | 12.3 |
| tiktoken o200k_base | 200,019 | 1.782 | 100.00% | 0 | 26.9 | 8.5 |
| tiktoken cl100k_base | 100,277 | 4.017 | 100.00% | 0 | 17.5 | 7.3 |
| BanglaBERT (csebuetnlp) | 32,000 | 1.774 | 51.87% | 1,067 | 6.4 | 6.3 |
| Bangla-BERT (sagorsarker) | 101,975 | 1.506 | 0.32% | 7,442 | 4.8 | 4.7 |
| MuRIL (Google) | 197,258 | 1.498 | 48.29% | 1,632 | 6.2 | 6.1 |
| mBERT cased | 119,547 | 2.122 | 48.33% | 1,732 | 5.6 | 5.5 |
| XLM-R / mE5 / BGE-M3 | 250,002 | 1.737 | 97.82% | 112 | 5.5 | 5.4 |
| Qwen3 | 151,669 | 3.582 | 100.00% | 0 | 4.5 | 4.4 |

Tokens per word by bucket (Unseen-source BanglishRev reviews):

| Tokenizer | bengali script | latin script | mixed script |
|---|---:|---:|---:|
| banglatok bn-bpe-58k | 1.376 | 1.367 | 1.424 |
| banglatok bn-bpe-28k | 1.597 | 1.563 | 1.639 |
| banglatok bn-bpe-28k-v2 | 1.793 | 1.699 | 1.818 |
| banglatok bn-bpe-28k-balanced-v2 | 1.784 | 1.678 | 1.804 |
| banglatok bn-bpe-58k-v2 | 1.495 | 1.416 | 1.516 |
| banglatok bn-bpe-58k-balanced-v2 | 1.511 | 1.414 | 1.524 |
| SentencePiece Unigram 27,760, same training data | 1.721 | 1.630 | 1.753 |
| SentencePiece BPE 27,760, same training data | 1.747 | 1.631 | 1.761 |
| SentencePiece Unigram 57,925, same training data | 1.488 | 1.427 | 1.533 |
| SentencePiece BPE 57,925, same training data | 1.509 | 1.420 | 1.534 |
| tiktoken o200k_base | 2.249 | 1.425 | 2.068 |
| tiktoken cl100k_base | 7.250 | 1.522 | 6.131 |
| BanglaBERT (csebuetnlp) | 1.344 | 2.100 | 1.529 |
| Bangla-BERT (sagorsarker) | 1.856 | 1.238 | 1.720 |
| MuRIL (Google) | 1.496 | 1.499 | 1.502 |
| mBERT cased | 2.799 | 1.604 | 2.536 |
| XLM-R / mE5 / BGE-M3 | 2.020 | 1.518 | 1.928 |
| Qwen3 | 6.264 | 1.509 | 5.357 |


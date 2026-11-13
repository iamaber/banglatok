# Training, normalization and research

The banglatok Python wheel is an inference library with two bundled pretrained models (see [MODELS.md](MODELS.md)). For installation and using the package, start with the [user guide](USAGE.md); uv is recommended and pip is supported. Training tokenizers and preparing corpora use the Go `bntok` command from a repository clone, not Python API methods. Raw corpora and research checkpoints are not bundled.

## Train without downloading a corpus

The eight records in [examples/training-records.jsonl](../examples/training-records.jsonl) are manually authored smoke data under the software license. They cover the four explicit language buckets; do not use the resulting model to claim trained quality.

```sh
go run ./cmd/bntok train -algorithm bpe -vocab 384 \
  -input examples/training-records.jsonl -out models/toy-bpe.json
go run ./cmd/bntok encode -model models/toy-bpe.json \
  -text 'আমি বাংলা বলি। ami bangla boli. meetingটা কখন?'
```

For your own corpus, use one JSON object per line with `text`, `bucket`, `group`, `source` and `split`. Buckets are `bangla`, `english`, `banglish`, and `code-mixed`; training rows use `split: "train"`. Attribution should be retained for external content. The default mixture gives sentence-weight contributions of 40/25/25/10; `-balanced` uses 25/25/25/25. Preserve related forms and duplicates within one group, and keep development/test groups out of training.

## Reproduce the research corpus

Read [CORPUS.md](../data/CORPUS.md) and source lock files first. Downloading the original sources is a separate, potentially large research operation with source-specific rights and privacy limits.

```sh
go build -o bin/bntok ./cmd/bntok
bin/bntok fetch
bin/bntok prepare
bin/bntok audit
bin/bntok train -vocab 32000 -out models/unigram-32000.json
bin/bntok train -algorithm bpe -vocab 32000 -out models/bpe-32000.json
bin/bntok evaluate -model models/unigram-32000.json
make matrix
```

Research reports refer to frozen local artifacts and their checksums. A fresh clone does not contain those datasets, trained vocabularies or checkpoints. Reproduction requires their documented source inputs, not just installing the Python wheel. In restricted environments, `BNTOK_GO` selects a local Go executable, while `GOCACHE`/`GOMODCACHE` can select writable caches.

## Model contract

Models are versioned JSON with `version`, `algorithm`, `normalization`, learned `pieces`, and BPE merge information where applicable. The supported normalization is `nfc-whitespace-v1`. Piece scores must be finite and nonpositive. Version 2 BPE may include `space_fusion` and training-word caches. Training writes a `.meta.json` sidecar with parameters, corpus/model hashes, toolchain and duration.

NFC and whitespace collapse/trim preserve case, Bengali marks, ZWJ/ZWNJ, punctuation, digits, emoji and spelling. Invalid UTF-8 on the Go side becomes U+FFFD; Python rejects invalid surrogate strings. ID positions `0..3` are reserved for encoder controls and are not emitted or accepted as ordinary decoded text. IDs `4..259` encode raw UTF-8 bytes, learned pieces start at `260`, and version-2 fused IDs follow the base vocabulary. Literal text such as `<unk>`, `▁` or `<0x20>` is never interpreted as an instruction.

Unigram uses substring seeding, EM, expected-count pruning and Viterbi decoding; BPE uses ranked merges over Unicode characters. Neither format is a SentencePiece model. Replacing an encoder's tokenizer requires a compatible embedding table and adaptation.

## Measurements and limits

- [Intrinsic tokenizer results](../research/reports/RESULTS.md), [framework comparison](../research/reports/FRAMEWORK_COMPARISON.md) and [Python API benchmarks](../research/reports/PYTHON_API.md).
- [Native performance changes](../research/reports/PERFORMANCE_IMPROVEMENTS.md) and [space-fusion experiment](../research/reports/SPEED_EXPERIMENT.md).
- [Retrieval protocol](../research/experiments/retrieval/README.md), [expanded-label audit](../research/reports/LABEL_AUDIT.md), and [latest original-tokenizer comparison](../research/reports/ROUND2_ORIGINAL_COMPARISON.md).

Tokenization speed, fertility or smaller vocabularies do not establish embedding quality. The completed retrieval experiments retain unchanged multilingual E5 under their declared gates. Labels and authored proxies are agent-generated where disclosed; independent human validation and natural-traffic generalization are not established.

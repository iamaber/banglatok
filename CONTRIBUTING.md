# Contributing

Small, reproducible changes are welcome. Discuss API changes in an issue first. Keep tokenizer behavior, model formats and research claims explicit; a speed improvement must preserve IDs unless a new model version is deliberate.

## Development

Use Python 3.10+ and Go 1.24+ with a C compiler. A supported, maintained Python version is preferable for development.

```sh
python -m venv .venv-dev
source .venv-dev/bin/activate
python -m pip install build twine ruff
python -m pip install .
python -m unittest discover -s tests -v
go test -race ./...
go vet ./...
ruff check src tests tools examples setup.py
```

Tests exercise the installed Go shared library. After changing Go or Python package code, reinstall with `python -m pip install --force-reinstall .` before testing. Running against an old wheel can test an obsolete native runtime.

Format changed Go files with `gofmt`. Prefer the standard library and the existing small API over new dependencies or generic abstractions. Add a focused regression test for behavior changes, and test actual native behavior rather than mocked return values.

## Package changes

Follow [docs/RELEASING.md](docs/RELEASING.md) to build from a source archive, inspect package contents and test installed wheels. Do not add corpora, private text, credentials or research checkpoints to distributions. The only bundled models are the documented ones in `src/bn_tokenizers_embedding/pretrained/` (see [docs/MODELS.md](docs/MODELS.md)). Performance changes must keep their IDs byte-for-byte identical; `go test ./tokenizer -bench Validation` with `BNTOK_MODEL`/`BNTOK_CORPUS` measures the native path.

## Research changes

Research environments are separate from the zero-dependency tokenizer API; see [research/experiments/retrieval/README.md](research/experiments/retrieval/README.md). Preserve frozen datasets and decisions. Report annotation origin, actual compute/data budgets, uncertainty and failed gates. Agent judgments must not be described as independent human review.

## Pull requests

Explain the problem, resulting behavior and verification. Include OS/architecture and Python/Go versions for native changes. Use a minimal hand-written reproduction instead of uploading private data. Keep discussion respectful and specific; review the code and evidence rather than the person.

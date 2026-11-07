# Public Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing tokenizer package reviewable and testable for its first public preview.
**Architecture:** Keep the ctypes/Go API. Add distribution metadata and a bundled toy model; build/test wheels from the sdist; keep research data outside distributions.
**Tech Stack:** Python stdlib, setuptools, Go/CGo, cibuildwheel, GitHub Actions, PyPI Trusted Publishing.
**Spec:** docs/superpowers/specs/2026-11-07-public-preview-design.md

## Global Constraints

- Python >=3.10; Go >=1.24 for source builds; no Python runtime dependencies.
- MIT original software; preserve native Go/x/text notices and separate corpus/model terms.
- Linux/macOS wheel build configuration; claim only platforms actually validated.
- No raw corpora, trained research models, automatic downloads, or invented quality claims.
- Publishing is opt-in after artifact verification and account configuration.

## Review Focus

- Installed wheel must find its native library and bundled model away from the source tree.
- Wheel must retain correct platform tag and native licenses.
- Sdist must rebuild without ignored local directories.
- Publishing must not execute on pull requests or ordinary pushes.
- README must work from a fresh clone and distinguish toy/demo assets from trained models.

### Task 1: Licensed, usable package

Files: LICENSE, THIRD_PARTY_NOTICES.md, third_party/licenses/*, pyproject.toml, MANIFEST.in, src/bn_tokenizers_embedding/data/demo-unigram.json, examples/quickstart.py, tests/test_python.py.
Interfaces: retain Tokenizer(model_path), normalize(text); demo loaded with importlib.resources.
- [ ] Add a test that asserts the bundled resource exists and round-trips Bengali/Banglish/English with the real native library; run and observe missing-resource failure.
- [ ] Add hand-authored model and metadata/license files; include demo in package data and examples in sdist.
- [ ] Build a wheel from an sdist in a clean directory; run installed-native integration tests and demo.
- [ ] Verify metadata, tags and archive inclusion/exclusion, then commit.

### Task 2: CI and release artifacts

Files: .github/workflows/ci.yml, .github/workflows/release.yml, pyproject.toml cibuildwheel configuration, tools/install_go.py if Linux container setup needs it.
Interfaces: build existing Go shared library through setup.py; run existing tests against installed artifacts.
- [ ] Add push/PR Go race/vet and Python installed-wheel CI with read-only permissions.
- [ ] Add manually dispatched Linux x86_64 and macOS x86_64/arm64 wheel matrix, sdist build, and Python 3.10–3.15 install tests.
- [ ] Add opt-in publishing job with environment protection and OIDC; validate tag/version and distributions before upload.
- [ ] Exercise any new installer/validation logic against real controlled inputs before implementation; validate YAML and local artifact paths.
- [ ] Commit independently.

### Task 3: Public documentation and verification

Files: README.md, CONTRIBUTING.md, SECURITY.md, CHANGELOG.md, docs/RELEASING.md, .github/ISSUE_TEMPLATE/*, .github/pull_request_template.md, reports/PUBLIC_RELEASE_READINESS.md.
Interfaces: document the artifacts and workflows from Tasks 1–2; research reports remain linked separately.
- [ ] Lead README with purpose, honest status, source install and offline demo; move workspace-dependent detail into developer/research references.
- [ ] Add contributor, responsible-reporting and opt-in release instructions, including external account/CI prerequisites.
- [ ] Run Go race/vet, installed Python tests on available interpreters, sdist rebuild and metadata validation.
- [ ] Obtain one fresh final review; fix concrete blockers and record unrun platform checks.
- [ ] Commit verified result and report exactly what remains before public publishing.

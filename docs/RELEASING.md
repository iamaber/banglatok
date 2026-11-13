# Release procedure

The current candidate is `0.2.0`. A local green build is necessary but does not validate the hosted Linux, macOS and Windows wheel matrix.

## 1. Prepare and review the candidate

1. Review the software license, the bundled-model license (`MODEL_LICENSE.md`, `docs/MODELS.md`) and retained native notices. Raw corpora and research checkpoints must not ship; only the two documented `pretrained/` models do.
2. Update `project.version` in `pyproject.toml`, `CITATION.cff` and CHANGELOG together.
3. Run CI on the exact commit to release. Review failures rather than changing gates to make them pass.
4. Use a fresh output directory; do not reuse old saved development wheels.

From a clone with Go, a C compiler and Python installed:

```sh
python -m pip install build twine
python -m build --outdir dist/0.2.0
python -m twine check --strict dist/0.2.0/*
python tools/check_dist.py dist/0.2.0 --version 0.2.0
python -m pip install --force-reinstall --no-deps dist/0.2.0/*.whl
python -m unittest discover -s tests -v
python examples/quickstart.py
go test -race ./...
go vet ./...
```

`python -m build` builds the wheel from the generated sdist. For source changes, verify this path rather than only building directly from a checkout containing ignored artifacts. The archive gate requires native code, the bundled models and demo, type marker, correct version/platform tags and license notices; it rejects excluded data/model/cache paths and unexpected Python runtime dependencies.

## 2. Run the hosted wheel workflow without publishing

Packaging pull requests run the build-and-test jobs without publishing. For manual dispatch, the workflow must first be available on the repository's default branch. Open **Actions → Build and optionally publish release → Run workflow**, select the reviewed commit/branch, and leave `publish` **false**.

It builds:

- Linux x86_64 and aarch64 in manylinux_2_28, each on a native runner.
- macOS x86_64 on an Intel runner.
- macOS arm64 on an Apple Silicon runner.
- Windows x86_64 on a Windows runner, with MinGW-w64 GCC from the preinstalled MSYS2 for cgo.

Each wheel has `py3-none` platform tags because the Go/ctypes bridge is independent of the CPython extension ABI. It is built once using CPython 3.11, then installed/tested on Python 3.10–3.15 on its matching runner. Free-threaded Python, PyPy and Windows on ARM are not validated.

Linux uses the verified setup-go toolchain mounted through cibuildwheel's documented `/host` filesystem. macOS sets a deployment target of 14.0. The filename encodes the platform floor; running on a newer CI OS alone does not validate every older OS release. Further deployment testing is welcome before broadening support claims.

Download the `release-distributions` artifact and inspect its sdist and five wheels. The workflow's final validation job runs Twine and the archive gate after all installed-wheel tests pass. Keep that exact artifact set for the release; do not replace a failed artifact with an unrelated local build.

## 3. Configure owner-controlled publishing

These account/repository settings are not performed by a local build:

- Confirm the intended GitHub repository is `iamaber/banglatok`; review visibility and enable the appropriate issue/security settings.
- Confirm the PyPI project name is available to your account. An absent project page does not reserve the name.
- Enable PyPI account 2FA and configure a pending Trusted Publisher with project `banglatok`, owner `iamaber`, repository `banglatok`, workflow `release.yml`, environment `pypi`.
- Create the GitHub environment `pypi` with required reviewers and tag restrictions. Review any environment restrictions before uploading.
- Retain the pinned action SHAs; Dependabot can propose reviewed updates. No long-lived PyPI API token is required.

See the [PyPA publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/) and [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/).

## 4. Publish the reviewed tag explicitly

After CI and platform tests pass, create/push the exact `v0.2.0` tag on the reviewed commit. Publishing requires repository/account authorization. Then dispatch `release.yml` from that tag with `publish=true`, for example with the authenticated GitHub CLI:

```sh
gh workflow run release.yml --ref v0.2.0 -f publish=true
```

The workflow rejects publishing from a branch or a tag that disagrees with `pyproject.toml`. Ordinary pushes, pull requests and build-only dispatches do not upload to PyPI. Only the final protected publishing job has `id-token: write`.

## 5. Verify the public install

In a clean environment on each claimed platform:

```sh
python -m pip install --only-binary=:all: banglatok==0.2.0
```

Run `examples/quickstart.py` and `banglatok --list-models` outside the repository checkout. Verify the published metadata, licenses and files. Update the changelog and public installation instructions only after the release is actually visible and installable. If a release is broken, follow PyPI's yanking procedure and publish a new version rather than silently replacing artifacts.

A new or retrained bundled model needs a `docs/MODELS.md` entry with its build command, checksums and limitations, a rerun of `benchmarks/compare_popular.py`, and a new minor version, because IDs change.

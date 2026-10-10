"""Fail closed on incomplete or accidentally oversized public distributions."""

import argparse
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

NAME = "bn_tokenizers_embedding"
NOTICES = {"LICENSE", "THIRD_PARTY_NOTICES.md", "third_party/licenses/GO-LICENSE",
           "third_party/licenses/X-TEXT-LICENSE", "third_party/licenses/X-TEXT-PATENTS"}


def check_paths(names: set[str]) -> None:
    excluded = {"models", "reports", ".tools", ".git", ".cache", "__pycache__"}
    for name in names:
        path = PurePosixPath(name)
        parts = path.parts
        if (path.is_absolute() or ".." in parts or excluded.intersection(parts)
                or any(p == ".env" or p.startswith(".env.") for p in parts)
                or "data/raw/" in name or "data/processed/" in name):
            raise ValueError(f"excluded content in distribution: {name}")


def check(directory: Path, version: str) -> None:
    sources = sorted(directory.glob("*.tar.gz"))
    wheels = sorted(directory.glob("*.whl"))
    source = directory / f"{NAME}-{version}.tar.gz"
    if sources != [source] or not wheels:
        raise ValueError("require one matching-version sdist and at least one wheel")
    with tarfile.open(source, "r:gz") as archive:
        prefix = f"{NAME}-{version}/"
        paths = {n.removeprefix(prefix) for n in archive.getnames()}
        check_paths(paths)
        required = {"pyproject.toml", "setup.py", "go.mod", "go.sum", "tokenizer/model.go",
                    "cmd/bntok-shared/main.go", "internal/native/native.go",
                    "src/bn_tokenizers_embedding/data/demo-unigram.json", *NOTICES}
        if not required <= paths:
            raise ValueError(f"sdist is missing required files: {sorted(required - paths)}")
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            paths = set(archive.namelist())
            check_paths(paths)
            info = f"{NAME}-{version}.dist-info"
            metadata_name = f"{info}/METADATA"
            if metadata_name not in paths:
                raise ValueError(f"wheel version differs: {wheel.name}")
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
            if metadata["Name"] != "bn-tokenizers-embedding" or metadata["Version"] != version:
                raise ValueError(f"wheel name/version differs: {wheel.name}")
            if metadata["License-Expression"] != "MIT AND BSD-3-Clause":
                raise ValueError("wheel license expression does not cover bundled native code")
            if metadata.get_all("Requires-Dist"):
                raise ValueError("unexpected Python runtime dependencies")
            if metadata["Requires-Python"] != ">=3.10":
                raise ValueError("unexpected Python support range")
            runtime = {f"{NAME}/_go{s}" for s in (".so", ".dylib", ".dll")}
            if len(runtime.intersection(paths)) != 1:
                raise ValueError("wheel must contain exactly one native runtime")
            required = {f"{NAME}/data/demo-unigram.json", f"{NAME}/py.typed"}
            required.update(f"{info}/licenses/{n}" for n in NOTICES)
            if not required <= paths:
                raise ValueError(f"wheel is missing demo/types/notices: {sorted(required - paths)}")
            wheel_metadata = BytesParser().parsebytes(archive.read(f"{info}/WHEEL"))
            tags = wheel_metadata.get_all("Tag", [])
            if (wheel_metadata["Root-Is-Purelib"] != "false" or not tags
                    or any(not t.startswith("py3-none-") or t.endswith("-any") for t in tags)):
                raise ValueError("native wheel must use Python-independent platform tags")
        print(f"Verified {wheel.name}")
    print(f"Verified {source.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    try:
        check(args.directory, args.version)
    except (ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.error(str(error))

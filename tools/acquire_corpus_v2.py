#!/usr/bin/env python3
"""Acquire checksum-pinned Wikimedia paragraphs outside runtime dependencies.

Use a separate research environment with pyarrow==22.0.0 and
mwparserfromhell==0.7.2. Candidate files are atomic, per-artifact JSONL;
preparation handles deduplication, held-out exclusions and training acceptance.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator
from http.client import IncompleteRead
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "data/corpus-v2-sources.lock.json"
DEFAULT_RAW = ROOT / "data/raw/expanded-v2"
FIELDS = {"text", "bucket", "source", "group", "attribution", "split"}
NOTICE = (
    "; Wikipedia contributors; CC BY-SA 3.0 "
    "(https://creativecommons.org/licenses/by-sa/3.0/); "
    "GFDL where applicable (https://www.gnu.org/licenses/fdl-1.3.html); "
    "snapshot 20231101; paragraphs NFC/whitespace normalized"
)
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?:\+?88[\s-]?)?01[3-9](?:[\s-]?\d){8}\b")
SECRET = re.compile(r"(?i)\b(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*\S+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_artifact(path: Path, item: dict) -> None:
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise ValueError(f"Pinned artifact mismatch (not overwritten): {path}")


def artifact_url(lock: dict, item: dict) -> str:
    if "url" in item:
        return item["url"]
    method = "raw" if item.get("raw") else "resolve"
    return (
        f"https://huggingface.co/datasets/{lock['repository']}/{method}/"
        f"{lock['revision']}/{item['path']}"
    )


def acquire(lock: dict, item: dict, raw: Path, reuse: Path | None = None) -> Path:
    existing = ROOT / item["existing"] if item.get("existing") else None
    if existing is not None and existing.is_file():
        validate_artifact(existing, item)
        return existing
    target = raw / "artifacts" / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        validate_artifact(target, item)
        return target
    if reuse is not None and (reuse / item["path"]).is_file():
        # Hard-link a checksum-verified prior download; neither copy is ever written.
        validate_artifact(reuse / item["path"], item)
        os.link(reuse / item["path"], target)
        return target
    partial = target.with_name(target.name + ".part")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item["bytes"]:
            raise ValueError(f"Oversized partial artifact: {partial}")
        if offset == item["bytes"]:
            validate_artifact(partial, item)
            partial.replace(target)
            return target
        headers = {"User-Agent": "banglatok-corpus-v2/1.0", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(
                Request(artifact_url(lock, item), headers=headers), timeout=90
            ) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError(f"Invalid range response: {content_range}")
                elif response.status == 200:
                    offset = 0  # Server ignored Range: never append a full response.
                else:
                    raise ValueError(f"Unexpected HTTP status: {response.status}")
                with partial.open("ab" if offset else "wb") as output:
                    written = offset
                    while block := response.read(min(1024 * 1024, item["bytes"] - written + 1)):
                        written += len(block)
                        if written > item["bytes"]:
                            raise ValueError(f"Download exceeds pinned size: {item['path']}")
                        output.write(block)
            if partial.stat().st_size != item["bytes"]:
                raise OSError("Incomplete download")
        except (OSError, IncompleteRead) as error:
            if attempt == 4:
                raise RuntimeError(
                    f"Download failed after five attempts: {item['path']}"
                ) from error
            time.sleep(2**attempt)
            continue
        validate_artifact(partial, item)  # A wrong checksum is fatal, never silently repinned.
        partial.replace(target)
        return target
    raise AssertionError("unreachable")


def paragraph(text: str, lang: str) -> tuple[str, str]:
    text = " ".join(unicodedata.normalize("NFC", text).split())
    if not 12 <= len(text) <= 2000:
        return "", "length"
    if len(text.split()) < 8:
        return "", "short_words"
    if "\ufffd" in text or any(unicodedata.category(c) == "Cc" for c in text):
        return "", "corrupt"
    if EMAIL.search(text) or PHONE.search(text) or SECRET.search(text):
        return "", "privacy_heuristic"
    letters = sum(c.isalpha() for c in text)
    if lang == "bn":
        script = sum("\u0980" <= c <= "\u09ff" and c.isalpha() for c in text)
    else:
        script = sum(c.isalpha() and "LATIN" in unicodedata.name(c, "") for c in text)
    if not letters or script * 2 < letters:
        return "", "script_mismatch"
    return text, ""


def write_json(path: Path, value: dict) -> None:
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def articles(path: Path, item: dict, batch_size: int) -> Iterator[dict]:
    """Yield one article/page at a time, retaining original source grouping."""
    if item.get("format") != "mediawiki-xml-bz2":
        import pyarrow.parquet as pq

        lang = "bn" if item["source"] == "bn-wikipedia" else "en"
        with pq.ParquetFile(path, pre_buffer=False) as parquet:
            for batch in parquet.iter_batches(batch_size=batch_size, columns=["id", "url", "text"]):
                for article in batch.to_pylist():
                    if not article["url"].startswith(f"https://{lang}.wikipedia.org/"):
                        raise ValueError(f"Unexpected article URL: {article['url']}")
                    yield {
                        **article,
                        "source": f"wikipedia-{lang}-20231101",
                        "group": f"wiki-{lang}:{article['id']}",
                        "attribution": article["url"] + NOTICE,
                    }
        return

    import mwparserfromhell

    project = item["project"]
    with bz2.open(path, "rb") as stream:
        context = ET.iterparse(stream, events=("start", "end"))
        _, root = next(context)
        for event, page in context:
            if event != "end" or not page.tag.endswith("}page"):
                continue
            if page.findtext("{*}ns") != "0" or page.find("{*}redirect") is not None:
                root.clear()
                continue
            title = page.findtext("{*}title", "")
            page_id = page.findtext("{*}id", "")
            revision = page.findtext("{*}revision/{*}id", "")
            wikitext = page.findtext("{*}revision/{*}text", "")
            wikitext = "\n".join(
                line
                for line in wikitext.splitlines()
                if not line.lstrip().startswith(("=", "*", "#", ";", ":"))
            )
            code = mwparserfromhell.parse(wikitext)
            for tag in code.filter_tags(recursive=False):
                if str(tag.tag).lower() in {
                    "ref",
                    "references",
                    "gallery",
                    "table",
                    "math",
                    "syntaxhighlight",
                    "source",
                    "code",
                    "pre",
                }:
                    code.remove(tag)
            plain = code.strip_code(normalize=True, collapse=False)
            # Drop headings, table/list remnants and contact-heavy lines; never
            # sentence-split a paragraph to inflate the record count.
            lines = [
                line
                for line in plain.splitlines()
                if not line.lstrip().startswith(("=", "*", "#", "|", "!", "{", "}", ";", ":"))
                and "http://" not in line
                and "https://" not in line
                and len(" ".join(line.split())) >= 80
                and line.rstrip().rstrip("\"'”’)]").endswith((".", "।", "॥", "?", "!", "…"))
            ]
            url = f"https://bn.{project}.org/wiki/{quote(title.replace(' ', '_'), safe='/')}"
            history = (
                f"https://bn.{project}.org/w/index.php?title={quote(title, safe='')}&action=history"
            )
            oldid = f"https://bn.{project}.org/w/index.php?oldid={revision}"
            group = title.split("/", 1)[0] if project == "wikibooks" else page_id
            notice = (
                f"; {project.capitalize()} contributors; CC BY-SA 4.0 "
                "(https://creativecommons.org/licenses/by-sa/4.0/); "
                f"revision {oldid}; contributors/history {history}; snapshot 20261101; "
                "markup removed; paragraphs NFC/whitespace normalized"
            )
            yield {
                "id": page_id,
                "url": url,
                "text": "\n".join(lines),
                "source": f"{project}-bn-20261101",
                "group": f"{project}-bn:{group}",
                "attribution": url + notice,
            }
            root.clear()


def emit(
    path: Path,
    item: dict,
    lock: dict,
    output: Path,
    raw: Path,
    max_paragraphs: int,
    limit: int,
    batch_size: int,
    offset: int = 0,
) -> dict:
    """Write qualifying paragraphs; the first ``offset`` are skipped, never emitted.

    Skipped rows are counted, character-counted and hashed exactly as they would
    have been written, so a continuation can prove it skipped a prior prefix.
    """
    lang = "en" if item["source"] == "en-wikipedia" else "bn"
    name = item.get("output_name", f"{item['source']}-{Path(item['path']).stem}")
    destination = output / f"{name}.jsonl"
    metadata_path = raw / "metadata" / f"{name}.json"
    settings = {
        "artifact_sha256": item["sha256"],
        "revision": item.get("revision", lock["revision"]),
        "max_paragraphs": max_paragraphs,
        "record_limit": limit,
        "normalization": "NFC then Unicode whitespace collapse",
        "filter_version": 2 if item.get("format") == "mediawiki-xml-bz2" else 1,
    }
    if offset:
        settings["record_offset"] = offset
    if not destination.exists() and offset and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["settings"] == settings and metadata["output"] is None:
            return metadata  # Completed shard wholly inside the skipped prefix.
    if destination.exists():
        if not metadata_path.exists():
            raise ValueError(f"Existing output without metadata: {destination}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["settings"] != settings or sha256(destination) != metadata["sha256"]:
            raise ValueError(f"Output/settings mismatch: {destination}; use a different --output")
        return metadata
    output.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    counts: Counter = Counter()
    digest = hashlib.sha256()
    skipped_digest = hashlib.sha256()
    stop = False
    with partial.open("wb") as stream:
        for article in articles(path, item, batch_size):
            counts["articles_scanned"] += 1
            kept = 0
            emitted = False
            for original in article["text"].split("\n"):
                text, reason = paragraph(original, lang)
                if reason:
                    counts[f"rejected_{reason}"] += 1
                    continue
                record = {
                    "text": text,
                    "bucket": "bangla" if lang == "bn" else "english",
                    "source": article["source"],
                    "group": article["group"],
                    "attribution": article["attribution"],
                    "split": "",
                }
                encoded = (
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                kept += 1
                if counts["skipped_records"] < offset:
                    skipped_digest.update(encoded)
                    counts["skipped_records"] += 1
                    counts["skipped_normalized_characters"] += len(text)
                    counts["skipped_bytes"] += len(encoded)
                    if kept >= max_paragraphs:
                        break
                    continue
                stream.write(encoded)
                digest.update(encoded)
                counts["records"] += 1
                counts["normalized_characters"] += len(text)
                if not emitted:
                    emitted = True
                    counts["articles_emitted"] += 1
                if limit and counts["records"] >= limit:
                    stop = True
                    break
                if kept >= max_paragraphs:
                    break
            if stop:
                break
    counts.setdefault("records", 0)
    counts.setdefault("normalized_characters", 0)
    if offset and not counts["records"]:
        partial.unlink()  # Entirely skipped: no candidate file, provenance only.
    metadata = {
        "source": item["source"],
        "output": str(destination) if counts["records"] or not offset else None,
        "settings": settings,
        "artifact": {**item, "local_path": str(path), "url": artifact_url(lock, item)},
        "bytes": partial.stat().st_size if partial.exists() else 0,
        "sha256": digest.hexdigest(),
        "counts": dict(counts),
        "quota_reached": stop,
        "license_evidence": item.get("license_evidence", lock["license_evidence"]),
        "sampling": (
            "pinned shard order, all articles in physical order; "
            "first qualifying paragraphs; no slicing"
        ),
    }
    if offset:
        metadata["skipped_sha256"] = skipped_digest.hexdigest()
        metadata["offset_semantics"] = (
            "record_offset = qualifying records still to skip when this shard began, "
            "in the same global English order as the default run; skipped rows are "
            "hashed/counted under skipped_* and never written or counted as records"
        )
    # Metadata first: a completed .jsonl never appears without its provenance.
    write_json(metadata_path, metadata)
    if metadata["output"] is not None:
        partial.replace(destination)
    return metadata


def verify(output: Path, raw: Path) -> dict:
    total: Counter = Counter()
    files = []
    for path in sorted(output.glob("*.jsonl")):
        name = path.stem
        meta = json.loads((raw / "metadata" / f"{name}.json").read_text(encoding="utf-8"))
        count = chars = 0
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                record = json.loads(line)
                if (
                    set(record) != FIELDS
                    or not all(isinstance(value, str) for value in record.values())
                    or record["bucket"] not in {"bangla", "english"}
                ):
                    raise ValueError(f"Invalid six-field schema: {path}:{count + 1}")
                lang = "bn" if record["bucket"] == "bangla" else "en"
                text, reason = paragraph(record["text"], lang)
                if reason or text != record["text"] or record["split"] != "":
                    raise ValueError(f"Invalid normalized candidate: {path}:{count + 1}")
                if record["source"].startswith("wikipedia-"):
                    valid = (
                        record["source"] == f"wikipedia-{lang}-20231101"
                        and record["group"].startswith(f"wiki-{lang}:")
                        and record["attribution"].startswith(f"https://{lang}.wikipedia.org/")
                        and record["attribution"].endswith(NOTICE)
                    )
                else:
                    project = record["source"].split("-")[0]
                    valid = (
                        project in {"wikibooks", "wikivoyage"}
                        and lang == "bn"
                        and record["source"] == f"{project}-bn-20261101"
                        and record["group"].startswith(f"{project}-bn:")
                        and record["attribution"].startswith(f"https://bn.{project}.org/wiki/")
                        and "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)"
                        in record["attribution"]
                        and "contributors/history https://" in record["attribution"]
                        and "revision https://" in record["attribution"]
                    )
                if not valid:
                    raise ValueError(f"Invalid provenance: {path}:{count + 1}")
                count += 1
                chars += len(text)
        if (
            digest.hexdigest() != meta["sha256"]
            or count != meta["counts"]["records"]
            or chars != meta["counts"]["normalized_characters"]
            or path.stat().st_size != meta["bytes"]
        ):
            raise ValueError(f"Output checksum/count mismatch: {path}")
        files.append(
            {
                "path": str(path),
                "records": count,
                "normalized_characters": chars,
                "sha256": digest.hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
        total["records"] += count
        total["normalized_characters"] += chars
    return {"files": files, "total": dict(total)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["bn-wikipedia", "en-wikipedia", "bn-wikibooks", "bn-wikivoyage", "all"],
        default="all",
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_RAW / "candidates")
    parser.add_argument("--bn-quota", type=int, default=0, help="0 = all pinned Bengali material")
    parser.add_argument("--en-quota", type=int, default=3_200_000)
    parser.add_argument("--books-quota", type=int, default=0)
    parser.add_argument("--voyage-quota", type=int, default=0)
    parser.add_argument(
        "--en-shards", type=int, default=8, help="first N pinned English shards, maximum 8"
    )
    parser.add_argument(
        "--max-paragraphs", type=int, default=20, help="10–20 qualifying paragraphs/article"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--en-offset",
        type=int,
        default=0,
        help=(
            "skip the first N qualifying English records of the default global order "
            "(hashed/counted, never emitted); --en-quota then counts only new records. "
            "Requires --source en-wikipedia and a separate --raw/--output"
        ),
    )
    parser.add_argument(
        "--reuse-artifacts",
        type=Path,
        help="prior artifacts directory: hard-link checksum-verified files instead of downloading",
    )
    parser.add_argument(
        "--verify", action="store_true", help="stream-verify completed candidates, without fetching"
    )
    args = parser.parse_args()
    if not 10 <= args.max_paragraphs <= 20 or not 1 <= args.en_shards <= 8 or args.batch_size < 1:
        parser.error("max-paragraphs must be 10–20, en-shards 1–8, batch-size positive")
    if min(args.bn_quota, args.en_quota, args.books_quota, args.voyage_quota, args.en_offset) < 0:
        parser.error("quotas and offsets must be nonnegative")
    if args.en_offset and (
        args.source != "en-wikipedia"
        or DEFAULT_RAW.resolve() in (args.raw.resolve(), *args.raw.resolve().parents)
        or DEFAULT_RAW.resolve() in (args.output.resolve(), *args.output.resolve().parents)
    ):
        parser.error(
            "--en-offset needs --source en-wikipedia and --raw/--output outside the default"
        )
    if args.verify:
        print(json.dumps(verify(args.output, args.raw), indent=2), flush=True)
        return
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    for artifact in lock["artifacts"]:
        if artifact["source"] == "license":
            acquire(lock, artifact, args.raw, args.reuse_artifacts)
    summary = []
    for source, quota in [
        ("bn-wikipedia", args.bn_quota),
        ("en-wikipedia", args.en_quota),
        ("bn-wikibooks", args.books_quota),
        ("bn-wikivoyage", args.voyage_quota),
    ]:
        if args.source not in ("all", source):
            continue
        records = skipped = 0
        items = [item for item in lock["artifacts"] if item["source"] == source]
        if source == "en-wikipedia":
            items = items[: args.en_shards]
        offset = args.en_offset if source == "en-wikipedia" else 0
        for item in items:
            if quota and records >= quota:
                break
            path = acquire(lock, item, args.raw, args.reuse_artifacts)
            metadata = emit(
                path,
                item,
                lock,
                args.output,
                args.raw,
                args.max_paragraphs,
                quota - records if quota else 0,
                args.batch_size,
                offset - skipped,
            )
            records += metadata["counts"]["records"]
            skipped += metadata["counts"].get("skipped_records", 0)
            summary.append(metadata)
            print(
                json.dumps(
                    {
                        "output": metadata["output"],
                        "counts": metadata["counts"],
                        "sha256": metadata["sha256"],
                    }
                ),
                flush=True,
            )
    args.raw.mkdir(parents=True, exist_ok=True)
    result = {"lock_sha256": sha256(args.lock), "outputs": summary}
    if args.en_offset:
        result["continuation"] = {
            "en_offset": args.en_offset,
            "en_quota": args.en_quota,
            "skipped_records": skipped,
            "emitted_records": records,
            "offset_satisfied": skipped == args.en_offset,
            "quota_satisfied": records == args.en_quota,
            "reuse_artifacts": str(args.reuse_artifacts) if args.reuse_artifacts else None,
        }
    write_json(args.raw / f"acquisition-{args.source}.json", result)


if __name__ == "__main__":
    main()

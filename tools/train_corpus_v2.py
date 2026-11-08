"""Train every corpus-v2 tokenizer candidate and same-size SentencePiece baseline.

Steps (research/reports/CORPUS_V2_PROTOCOL.md):

- four native BPE bases (``bntok train``): each base vocabulary with the default
  40/25/25/10 mixture and with ``-balanced`` 25/25/25/25;
- four fused models (``benchmarks/build_fused_model.py``) at each base's total
  budget, fitted on the training split only;
- SentencePiece Unigram and BPE baselines at each total budget
  (``benchmarks/train_sentencepiece_baselines.py``), one invocation per budget.

Each step writes ``logs/<step>.log`` and ``steps/<step>.json`` (command, input
and output SHA-256, wall time, peak RSS). A rerun skips a step whose recorded
command, input hashes and output hashes still match; anything else reruns.
Independent steps run concurrently up to ``--jobs`` processes and an estimated
memory budget. ``manifest.json`` and ``SHA256SUMS`` summarize the outputs.
"""

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FUSE = ROOT / "benchmarks/build_fused_model.py"
SENTENCEPIECE = ROOT / "benchmarks/train_sentencepiece_baselines.py"
# Base vocabulary -> total fused budget, matching bn-bpe-28k and bn-bpe-58k.
SIZES = {16000: 27760, 32000: 57925}
GIB = 1 << 30

# Peak memory footprint and wall seconds per GiB of training JSONL, measured with
# /usr/bin/time -l on the 1.80 GiB attempt-1 corpus (models/corpus-v2-probe/logs):
# 32k base 5.07 GB / 83 s; fusion 0.94 GB / 94 s; one SentencePiece budget
# (text export, BPE, then Unigram) 11.0 GB / 338 s. A 25% margin covers larger
# corpora and the unmeasured 16k base, which shares the word-count stage.
MARGIN = 1.25
PEAK_GIB_PER_INPUT_GIB = {
    "base": 2.63 * MARGIN,
    "fuse": 0.48 * MARGIN,
    "sentencepiece": 5.71 * MARGIN,
}
SECONDS_PER_INPUT_GIB = {"base": 46.2, "fuse": 52.0, "sentencepiece": 188.2}
# Fraction of physical memory the scheduler may commit to estimated peaks.
MEMORY_FRACTION = 0.75


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_summary(path: Path) -> dict[str, object]:
    digest, rows = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
            rows += chunk.count(b"\n")
    return {
        "path": rel(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "sha256": digest.hexdigest(),
    }


def physical_memory() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def rel(path: Path) -> str:
    """Repository-relative path when possible: steps run from the repository root."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def label(budget: int) -> str:
    return f"{round(budget / 1000)}k" if budget >= 1000 else str(budget)


@dataclass
class Step:
    name: str
    kind: str
    command: list[str]
    inputs: list[Path]
    outputs: list[Path]
    after: list[str] = field(default_factory=list)
    cleanup: list[Path] = field(default_factory=list)
    memory: float = 0.0  # estimated peak bytes
    seconds: float = 0.0  # estimated wall seconds


def plan(args: argparse.Namespace, input_gib: float) -> list[Step]:
    out, training = args.output, rel(args.training)
    python, bntok = rel(args.python), rel(args.bntok)
    tools = {"base": args.bntok, "fuse": FUSE, "sentencepiece": SENTENCEPIECE}
    steps: list[Step] = []

    def add(
        name: str, kind: str, command: list[str], inputs: list[Path], outputs: list[Path], **extra
    ) -> None:
        steps.append(
            Step(
                name,
                kind,
                command,
                [tools[kind], *inputs],
                outputs,
                memory=PEAK_GIB_PER_INPUT_GIB[kind] * input_gib * GIB,
                seconds=SECONDS_PER_INPUT_GIB[kind] * input_gib,
                **extra,
            )
        )

    for base_vocab, budget in args.sizes.items():
        for balanced in (False, True):
            variant = "-balanced" if balanced else ""
            base = out / "base" / f"bpe-{base_vocab}{variant}-v2.json"
            fused = out / f"bn-bpe-{label(budget)}{variant}-v2.json"
            add(
                f"base-{base_vocab}{variant}",
                "base",
                [bntok, "train", "-algorithm", "bpe", "-vocab", str(base_vocab)]
                + (["-balanced"] if balanced else [])
                + ["-input", training, "-out", rel(base)],
                [args.training],
                [base, Path(f"{base}.meta.json")],
            )
            add(
                f"fuse-{label(budget)}{variant}",
                "fuse",
                [python, rel(FUSE), "--base", rel(base), "--training", training]
                + ["--budget", str(budget), "--output", rel(fused)],
                [args.training, base],
                [fused, fused.with_suffix(".provenance.json")],
                after=[f"base-{base_vocab}{variant}"],
            )
    if not args.skip_sentencepiece:
        for budget in args.sizes.values():
            directory = out / f"sentencepiece-{budget}"
            prefixes = [directory / f"sentencepiece-{kind}-{budget}" for kind in ("unigram", "bpe")]
            add(
                f"sentencepiece-{budget}",
                "sentencepiece",
                [python, rel(SENTENCEPIECE), "--training", training, "--output", rel(directory)]
                + ["--metadata", rel(directory / "metadata.json"), "--vocab-size", str(budget)]
                + ["--workers", "1"],
                [args.training],
                [directory / "metadata.json"]
                + [
                    prefix.with_suffix(suffix)
                    for prefix in prefixes
                    for suffix in (".model", ".vocab")
                ],
                cleanup=[directory / "train.txt"],
            )
    return steps


class Runner:
    def __init__(self, output: Path, known: dict[Path, str]) -> None:
        self.output = output
        self.hashes = dict(known)

    def digest(self, path: Path) -> str:
        key = path.resolve()
        if key not in self.hashes:
            self.hashes[key] = sha256(path)
        return self.hashes[key]

    def record_path(self, step: Step) -> Path:
        return self.output / "steps" / f"{step.name}.json"

    def fingerprint(self, step: Step) -> dict[str, object]:
        return {
            "command": step.command,
            "inputs": {rel(path): self.digest(path) for path in step.inputs},
        }

    def completed(self, step: Step) -> dict | None:
        """Return the step record when its command, inputs and outputs still match."""
        path = self.record_path(step)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if {key: record.get(key) for key in ("command", "inputs")} != self.fingerprint(step):
            return None
        for output in step.outputs:
            if not output.exists() or sha256(output) != record["outputs"].get(rel(output)):
                return None
        return record

    def run(self, step: Step) -> dict:
        """Run one step, timing it and recording its peak resident memory."""
        fingerprint = self.fingerprint(step)
        self.record_path(step).unlink(missing_ok=True)
        log = self.output / "logs" / f"{step.name}.log"
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        started = time.perf_counter()
        with log.open("w", encoding="utf-8") as stream:
            stream.write(f"$ {shlex.join(step.command)}\n")
            stream.flush()
            process = subprocess.Popen(
                step.command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
            )
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        seconds = time.perf_counter() - started
        # ru_maxrss is in bytes on macOS and KiB on Linux; it covers the step
        # process and its waited-for descendants (SentencePiece pool workers).
        peak = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if process.returncode != 0:
            raise RuntimeError(f"{step.name} exited {process.returncode}; see {log}")
        missing = [str(path) for path in step.outputs if not path.exists()]
        if missing:
            raise RuntimeError(f"{step.name} did not write {missing}; see {log}")
        for path in step.cleanup:
            path.unlink(missing_ok=True)
        record = {
            "step": step.name,
            "kind": step.kind,
            **fingerprint,
            "outputs": {rel(path): self.digest(path) for path in step.outputs},
            "log": rel(log),
            "started": started_at,
            "wall_seconds": seconds,
            "peak_rss_bytes": peak,
            "estimated_peak_bytes": round(step.memory),
        }
        self.record_path(step).write_text(json.dumps(record, indent=2) + "\n")
        return record


def default_jobs(steps: list[Step]) -> int:
    """Processes are capped by CPUs; the scheduler separately enforces the memory budget."""
    return max(1, min(os.cpu_count() or 1, len(steps)))


def schedule(steps: list[Step], runner: Runner, jobs: int, memory_budget: float) -> dict[str, dict]:
    """Run steps after their dependencies, longest first, within jobs and memory."""
    pending = sorted(steps, key=lambda step: -step.seconds)
    records: dict[str, dict] = {}
    failed: dict[str, str] = {}
    stale: set[str] = set()
    running: dict[Future, Step] = {}
    with ThreadPoolExecutor(jobs) as pool:
        while pending or running:
            committed = sum(step.memory for step in running.values())
            for step in list(pending):
                if any(name in failed for name in step.after):
                    pending.remove(step)
                    failed[step.name] = "dependency failed"
                    continue
                if any(name not in records for name in step.after):
                    continue
                if step.name not in stale:
                    if (record := runner.completed(step)) is not None:
                        pending.remove(step)
                        records[step.name] = record | {"skipped": True}
                        print(f"skip  {step.name} (inputs and outputs unchanged)", flush=True)
                        continue
                    stale.add(step.name)
                if len(running) >= jobs or (running and committed + step.memory > memory_budget):
                    continue
                pending.remove(step)
                committed += step.memory
                print(
                    f"start {step.name} (estimated {step.memory / GIB:.1f} GiB, "
                    f"{step.seconds / 60:.0f} min)",
                    flush=True,
                )
                running[pool.submit(runner.run, step)] = step
            if not running:  # newly unblocked steps or failed dependents remain
                continue
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                step = running.pop(future)
                try:
                    records[step.name] = future.result() | {"skipped": False}
                except Exception as error:  # noqa: BLE001 - report every failed step
                    failed[step.name] = str(error)
                    print(f"FAIL  {step.name}: {error}", flush=True)
                    continue
                record = records[step.name]
                print(
                    f"done  {step.name} in {record['wall_seconds']:.1f} s, "
                    f"peak {record['peak_rss_bytes'] / GIB:.2f} GiB",
                    flush=True,
                )
    if failed:
        raise SystemExit("failed steps: " + json.dumps(failed, indent=2))
    return records


def write_summary(
    args: argparse.Namespace, training: dict, steps: list[Step], records: dict
) -> None:
    out = args.output
    artifacts = {
        str(path.relative_to(out)): records[step.name]["outputs"][rel(path)]
        for step in steps
        for path in step.outputs
    }
    (out / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(artifacts.items()))
    )
    manifest = {
        "protocol": "research/reports/CORPUS_V2_PROTOCOL.md",
        "training": training,
        "sizes": {str(base): budget for base, budget in args.sizes.items()},
        "jobs": args.jobs,
        "memory_budget_bytes": round(args.memory_budget),
        "host": {
            "machine": platform.machine(),
            "system": platform.platform(),
            "cpus": os.cpu_count(),
            "memory_bytes": physical_memory(),
            "python": platform.python_version(),
        },
        "packages": {name: version(name) for name in ("banglatok", "sentencepiece")},
        "artifacts": artifacts,
        "steps": {
            step.name: {
                key: records[step.name][key]
                for key in (
                    "command",
                    "inputs",
                    "outputs",
                    "log",
                    "started",
                    "wall_seconds",
                    "peak_rss_bytes",
                    "skipped",
                )
            }
            for step in steps
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def size_pair(text: str) -> tuple[int, int]:
    base, _, budget = text.partition(":")
    try:
        pair = int(base), int(budget)
    except ValueError:
        raise argparse.ArgumentTypeError("expected BASE:BUDGET, e.g. 32000:57925") from None
    if not 260 < pair[0] < pair[1]:
        raise argparse.ArgumentTypeError("need 260 < BASE < BUDGET")
    return pair


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--training", type=Path, required=True, help="training-split JSONL records")
    parser.add_argument(
        "--output", type=Path, required=True, help="output directory, e.g. models/corpus-v2"
    )
    parser.add_argument(
        "--size",
        type=size_pair,
        action="append",
        metavar="BASE:BUDGET",
        help=(
            "base vocabulary and fused/SentencePiece total budget "
            "(default 16000:27760 and 32000:57925)"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help="concurrent processes (default: CPU count; memory budget still applies)",
    )
    parser.add_argument(
        "--memory-gib",
        type=float,
        help=(
            "memory for concurrent estimated peaks "
            f"(default {MEMORY_FRACTION:.0%} of physical memory)"
        ),
    )
    parser.add_argument("--skip-sentencepiece", action="store_true")
    parser.add_argument("--bntok", type=Path, default=ROOT / "bin/bntok")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    args.sizes = dict(args.size or SIZES.items())
    if len(set(args.sizes.values())) != len(args.sizes):
        parser.error("budgets must be distinct")
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be positive")
    if not args.bntok.exists():
        parser.error(f"{args.bntok} missing; build with: go build -o bin/bntok ./cmd/bntok")
    args.training, args.output = args.training.resolve(), args.output.resolve()
    args.bntok, args.python = args.bntok.resolve(), args.python.absolute()
    for directory in ("logs", "steps", "base"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)

    training = training_summary(args.training)
    steps = plan(args, training["bytes"] / GIB)
    args.memory_budget = (
        (args.memory_gib * GIB) if args.memory_gib else physical_memory() * MEMORY_FRACTION
    )
    args.jobs = args.jobs or default_jobs(steps)
    print(
        f"{len(steps)} steps on {training['rows']:,} rows ({training['bytes'] / GIB:.2f} GiB); "
        f"jobs={args.jobs}, memory budget {args.memory_budget / GIB:.1f} GiB",
        flush=True,
    )
    runner = Runner(args.output, {args.training: training["sha256"]})
    records = schedule(steps, runner, args.jobs, args.memory_budget)
    write_summary(args, training, steps, records)


if __name__ == "__main__":
    main()

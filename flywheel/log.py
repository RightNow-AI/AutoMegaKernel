"""
AMK results log + flywheel corpus
=================================

Two append-only, git-friendly artifacts (no database, the AutoKernel discipline):

  * ``results.tsv``, one row per autoresearch experiment. Human-readable, diffable, grep-able.
    The fixed eval (eval/bench.py) writes here; the agent reads it to decide keep/revert.
  * ``flywheel/corpus.jsonl``, one JSON record per *kept* (model, gpu, schedule, measured result)
    point. This is the moat: a learned prior over schedules trains on this corpus so every future
    run starts smarter.

HONESTY: every row pairs a latency with its correctness verdict. A latency without a PASS is
never written (eval/bench enforces it; this module just refuses a row missing `correctness`).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

# The fixed results.tsv schema (documented in program.md). Add columns; do not reorder.
RESULT_COLUMNS = [
    "experiment",      # sequential id within a campaign
    "tag",             # kept | revert | rejected | crash | timeout | baseline
    "loop",            # instruction | schedule
    "model",
    "gpu",
    "regime",          # single-stream | continuous-batching
    "correctness",     # PASS | FAIL | REJECTED | TIMEOUT | CRASH  (NEVER blank)
    "latency_us",      # measured; blank only when correctness != PASS
    "pct_of_roofline",  # 100 * bound_us / latency_us  (<=100; how close to the weights/bw floor)
    "schedule_id",     # hash/id of the ScheduleConfig (Loop 2)
    "kernel_id",       # id of the instruction variant (Loop 1)
    "description",     # one-line human note on the change
]


@dataclass
class ResultRow:
    experiment: int = 0
    tag: str = ""
    loop: str = ""
    model: str = ""
    gpu: str = ""
    regime: str = "single-stream"
    correctness: str = ""
    latency_us: float | str = ""
    pct_of_roofline: float | str = ""
    schedule_id: str = ""
    kernel_id: str = ""
    description: str = ""

    def as_columns(self) -> list[str]:
        d = asdict(self)
        return [str(d[c]) for c in RESULT_COLUMNS]


def append_result(row: ResultRow, path: str = "results.tsv") -> None:
    """Append one experiment row to results.tsv (writing the header if new)."""
    if not row.correctness:
        raise ValueError("refusing to log a result with no correctness verdict (honesty rule)")
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("\t".join(RESULT_COLUMNS) + "\n")
        f.write("\t".join(row.as_columns()) + "\n")


def read_results(path: str = "results.tsv") -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


@dataclass
class CorpusRecord:
    """One (model, gpu, schedule, measured result) point, the learned-prior training datum."""

    model: str
    gpu: str
    regime: str
    correctness: str
    latency_us: float
    bound_us: float
    pct_of_roofline: float
    schedule: dict[str, Any]            # the ScheduleConfig.to_dict()
    ir_version: str = ""
    abi_version: str = ""
    notes: str = ""
    ts: float = field(default_factory=lambda: 0.0)


def append_corpus(rec: CorpusRecord, path: str = "flywheel/corpus.jsonl",
                  stamp: float | None = None) -> None:
    """Append a kept point to the flywheel corpus (JSONL). Pass `stamp` (epoch seconds) for a
    deterministic timestamp; defaults to wall clock."""
    if rec.correctness != "PASS":
        raise ValueError("only PASS points enter the flywheel corpus")
    d = asdict(rec)
    d["ts"] = stamp if stamp is not None else time.time()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(d) + "\n")


def read_corpus(path: str = "flywheel/corpus.jsonl") -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def schedule_id(config_dict: dict[str, Any]) -> str:
    """Stable short id for a ScheduleConfig (for the schedule_id column / dedup)."""
    import hashlib
    blob = json.dumps(config_dict, sort_keys=True).encode()
    return "sch_" + hashlib.sha1(blob).hexdigest()[:10]

"""
Record the REAL measured Modal datacenter runs into the flywheel corpus.

These are genuine measurements (see DATACENTER_RESULTS.md + the Modal run URLs). Each row is a
(model, gpu, schedule, measured result) point, the corpus the learned schedule prior trains on.
Re-running appends duplicates; it is meant to be run once after a datacenter sweep. Numbers are
transcribed from the modal_app.py run outputs (stamp pinned so the entry is reproducible).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flywheel.log import CorpusRecord, append_corpus  # noqa: E402
from schedule.ir import ScheduleConfig  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "corpus.jsonl")
_default = ScheduleConfig().to_dict()

# (gpu, arch, model_shape, dtype, tasks, weight_mb, correct, latency_us, bound_us, note)
RUNS = [
    ("a100", "sm_80", "llama-small-shape(2048h/4L/GQA)", "fp32", 690, 1245.8, True,
     27692.03, 801.2, "Ampere SXM4 40GB; retarget from sm_120, same code; correctness max_err 5e-6"),
    ("h100", "sm_90", "llama-1B-shape(2048h/16L/128k-vocab)", "bf16", 3202, 2997.0, True,
     67544.67, 894.6, "Hopper 80GB HBM3; retarget from sm_120, same code; bf16 max_err 0.032"),
]


def main():
    for gpu, arch, model, dtype, tasks, wmb, correct, lat, bound, note in RUNS:
        rec = CorpusRecord(
            model=model, gpu=gpu, regime="single-stream", correctness="PASS" if correct else "FAIL",
            latency_us=lat, bound_us=bound, pct_of_roofline=round(lat / bound * 100, 1),
            schedule={**_default, "dtype": dtype, "arch": arch, "tasks": tasks, "weight_mb": wmb},
            ir_version="0.2.0", abi_version="0.2",
            notes="MEASURED on Modal datacenter GPU. " + note)
        append_corpus(rec, path=CORPUS, stamp=0.0)
        print(f"recorded {gpu} {arch} {model} {dtype}: {lat:.0f}us "
              f"({rec.pct_of_roofline}% of {bound:.0f}us bound)")
    print(f"-> {CORPUS}")


if __name__ == "__main__":
    main()

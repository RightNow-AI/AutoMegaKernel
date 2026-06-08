"""
End-to-end product test: `amk compile toy` must import -> search -> lower -> VALIDATE ->
verify correctness vs eager -> emit artifacts -> log the flywheel row, with correctness PASS.

Runs on CPU (the authoritative correctness path) so it is GPU-independent and deterministic.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compile import amk_compile  # noqa: E402


def test_compile_toy_end_to_end(tmp_path):
    out = str(tmp_path)
    res = amk_compile("toy", gpu="rtx5090", regime="single-stream", search_budget=8,
                      device="cpu", token=7, out_dir=out, verbose=False, stamp=0.0)

    assert res["correctness"] == "PASS", res
    assert os.path.exists(res["program"]) and os.path.exists(res["report"])

    # the emitted program is a valid, loadable, re-validatable megakernel
    from schedule.ir import MegakernelProgram, validate
    prog = MegakernelProgram.load(res["program"])
    assert validate(prog).ok
    assert prog.meta or prog.tasks  # non-empty

    # the flywheel logged exactly one kept point + one results row
    with open(os.path.join(out, "results.tsv"), encoding="utf-8") as f:
        rows = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(rows) == 2  # header + 1
    assert "PASS" in rows[1]

    corpus = os.path.join(out, "..", "flywheel", "corpus.jsonl")
    if os.path.exists(corpus):
        recs = [json.loads(ln) for ln in open(corpus, encoding="utf-8") if ln.strip()]
        assert any(r["correctness"] == "PASS" for r in recs)


def test_compile_two_layer_toy(tmp_path):
    res = amk_compile("toy-2L", gpu="rtx5090", device="cpu", token=3,
                      out_dir=str(tmp_path), verbose=False, stamp=0.0)
    assert res["correctness"] == "PASS"


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        r = amk_compile("toy", gpu="rtx5090", search_budget=8, device="cpu",
                        out_dir=d, verbose=True, stamp=0.0)
        print("RESULT:", r)
        assert r["correctness"] == "PASS"
        print("COMPILE E2E OK")

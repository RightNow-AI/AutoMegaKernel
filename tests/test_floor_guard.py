"""
AMK, CPU-only acceptance test for the PHYSICAL-FLOOR HONESTY GUARD (TCG-FLOOR).

A batch-1 decode is memory bound: it must read every weight once, so its latency CANNOT fall below
the weights/bandwidth roofline floor (``bound_us = weight_bytes / HBM_bandwidth``). A sub-floor
number is physically impossible, a measured artifact (a silently-failed launch returning a
stale-but-correct buffer) or an over-optimistic cost-model fallback. The harness WITHHOLDS it
(nulls the latency) rather than present an impossible win, the same discipline as the correctness
gate. This is the honesty invariant the OSS release advertises; this test proves the guard fires.

Strategy (GPU-free): on device='cpu' the harness latency is the analytic ``cost_model.predict_us``
(imported into the harness namespace). We monkeypatch ``harness.predict_us`` so a toy eval's
predicted latency lands BELOW its own roofline floor, then assert harness.evaluate WITHHOLDS it:
``latency_us``/``latency_kind``/``pct_of_roofline`` come back None and a note explains why. We also
assert the un-patched eval is ABOVE the floor (so the guard is the only thing changing the result).

Run:  uv run python -m pytest tests/test_floor_guard.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness  # noqa: E402

GPU = "rtx5090"
MODEL = "toy"


def test_baseline_eval_is_above_the_floor():
    """Sanity anchor: the un-patched toy eval on CPU is honest and ABOVE the floor (latency present,
    pct_of_roofline > 100% of the floor). This makes the withholding in the next test attributable
    purely to the injected sub-floor latency, not to anything else."""
    v = harness.evaluate(MODEL, GPU, None, device="cpu")
    assert v["valid"] is True and v["correct"] is True
    assert v["latency_us"] is not None and v["latency_kind"] == "predicted"
    assert v["bound_us"] is not None and v["bound_us"] > 0
    # a real memory-bound decode is at/above the weights/bandwidth floor.
    assert v["latency_us"] >= v["bound_us"], "baseline latency should not already be sub-floor"
    assert v["pct_of_roofline"] is not None


def test_subfloor_predicted_latency_is_withheld(monkeypatch):
    """Force the analytic prediction BELOW the roofline floor and assert harness.evaluate withholds
    it: a physically-impossible memory-bound latency is never banked as a win."""
    orig_predict = harness.predict_us
    seen = {}

    def fake_predict(prog, target):
        # half the floor: provably impossible (you cannot stream the weights in less than the floor).
        floor = target.bandwidth_bound_us(prog.total_weight_bytes())
        seen["floor"] = floor
        seen["fake"] = floor * 0.5
        return floor * 0.5

    monkeypatch.setattr(harness, "predict_us", fake_predict)
    v = harness.evaluate(MODEL, GPU, None, device="cpu")
    # restore is automatic via monkeypatch; keep a reference so the linter sees orig_predict used.
    assert orig_predict is not None

    # the schedule is still valid + correct (the guard is orthogonal to correctness).
    assert v["valid"] is True and v["correct"] is True
    # the bound is reported, and our injected latency was genuinely below it.
    assert v["bound_us"] is not None and v["bound_us"] > 0
    assert seen["fake"] < seen["floor"], "the injected latency must actually be sub-floor"

    # THE GUARD: a sub-floor latency is WITHHELD, no latency, no kind, no roofline pct.
    assert v["latency_us"] is None, "sub-floor latency must be withheld (set to None)"
    assert v["latency_kind"] is None, "withheld latency carries no kind"
    assert v["pct_of_roofline"] is None, "withheld latency carries no roofline pct"
    # and a human-readable note explains the physical-floor withholding.
    assert any("floor" in n.lower() and "withheld" in n.lower() for n in v["notes"]), \
        f"expected a physical-floor withholding note, got: {v['notes']}"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))

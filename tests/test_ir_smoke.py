"""Smoke test for the standard IR + the deadlock-freedom validator.

This is the first proof that the load-bearing contract behaves: a valid 3-instruction DAG
passes, a cyclic / unsatisfiable schedule is REJECTED, and JSON round-trips. Run directly
(`uv run python tests/test_ir_smoke.py`) or under pytest.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schedule.ir import (  # noqa: E402
    BufferKind, DType, InstructionKind, MegakernelProgram, TARGETS, Wait, validate,
)


def build_valid_chain() -> MegakernelProgram:
    """x --rmsnorm--> h --gemv--> y --add(x)--> out  (a tiny valid forward-pass fragment)."""
    p = MegakernelProgram(meta={"model": "toy", "gpu": "rtx5090"}, target=TARGETS["rtx5090"])
    x = p.new_buffer("x", BufferKind.IO_INPUT, DType.F16, (1, 16))
    w_norm = p.new_buffer("norm.w", BufferKind.WEIGHT, DType.F16, (16,), source="norm.weight")
    w_proj = p.new_buffer("proj.w", BufferKind.WEIGHT, DType.F16, (16, 16), source="proj.weight")
    h = p.new_buffer("h", BufferKind.ACTIVATION, DType.F16, (1, 16))
    y = p.new_buffer("y", BufferKind.ACTIVATION, DType.F16, (1, 16))
    out = p.new_buffer("out", BufferKind.IO_OUTPUT, DType.F16, (1, 16))

    c_norm = p.new_counter("rmsnorm done")
    c_proj = p.new_counter("gemv done")
    c_add = p.new_counter("residual done")

    p.add_task(InstructionKind.RMSNORM, [x.id, w_norm.id], [h.id], out_counter=c_norm.id,
               params={"eps": 1e-6, "hidden": 16}, label="rmsnorm")
    p.add_task(InstructionKind.GEMV_TILE, [h.id, w_proj.id], [y.id], out_counter=c_proj.id,
               waits=[Wait(c_norm.id, 1)], params={"K": 16, "N_tile": 16, "n_off": 0}, label="gemv")
    p.add_task(InstructionKind.ADD, [y.id, x.id], [out.id], out_counter=c_add.id,
               waits=[Wait(c_proj.id, 1)], params={}, label="residual")
    return p


def test_valid_chain_passes():
    p = build_valid_chain()
    res = validate(p)
    assert res.ok, res.report()
    order = p.topological_order()
    assert order is not None and len(order) == 3
    exec_order, stuck = p.simulate_counters()
    assert stuck == [], f"stuck tasks: {stuck}"
    assert len(exec_order) == 3


def test_cycle_is_rejected():
    p = build_valid_chain()
    # Make the first task wait on the last task's counter -> cycle -> guaranteed deadlock.
    p.tasks[0].waits.append(Wait(p.counters[2].id, 1))
    res = validate(p)
    assert not res.ok
    assert any("CYCLE" in e for e in res.errors), res.report()
    _, stuck = p.simulate_counters()
    assert stuck, "cyclic program must leave tasks stuck"


def test_unsatisfiable_threshold_is_rejected():
    p = build_valid_chain()
    # Wait for 5 increments of a counter that only one task produces -> unsatisfiable.
    p.tasks[1].waits = [Wait(p.counters[0].id, 5)]
    res = validate(p)
    assert not res.ok
    assert any("unsatisfiable" in e for e in res.errors), res.report()


def test_json_roundtrip():
    p = build_valid_chain()
    s = p.to_json()
    p2 = MegakernelProgram.from_json(s)
    assert validate(p2).ok
    assert p2.to_json() == s  # stable serialization
    assert len(p2.tasks) == len(p.tasks)


if __name__ == "__main__":
    test_valid_chain_passes()
    test_cycle_is_rejected()
    test_unsatisfiable_threshold_is_rejected()
    test_json_roundtrip()
    p = build_valid_chain()
    print(validate(p).report())
    print(p.summary())
    print("bandwidth floor (us):",
          round(TARGETS["rtx5090"].bandwidth_bound_us(p.total_weight_bytes()), 4))
    print("ALL IR SMOKE TESTS PASSED")

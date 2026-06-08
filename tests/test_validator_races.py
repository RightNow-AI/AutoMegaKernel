"""
Regression tests for the validator soundness bugs the adversarial contract review found.
Each test pins a counterexample that v0's validate() WRONGLY accepted; they must now be
REJECTED (or, for additive compatibility / huge cycles, handled without crashing).

These are the teeth behind 'correctness from structure': if any of these regress, an
auto-generated schedule could race or hang a real GPU.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schedule.ir import (  # noqa: E402
    ABI_MAX_INPUTS, BufferKind, DType, InstructionKind, MegakernelProgram, TARGETS, Wait, validate,
)

DT = DType.F32


def _p():
    return MegakernelProgram(meta={"model": "race", "gpu": "rtx5090"}, target=TARGETS["rtx5090"])


def _b(p, name, kind, shape, source=None):
    return p.new_buffer(name, kind, DT, tuple(shape), source=source).id


def test_partial_wait_on_shared_counter_is_rejected():
    """(D) 4 tiles share a counter; a consumer waiting threshold=2 is a first-2-of-4 RACE."""
    p = _p()
    x = _b(p, "x", BufferKind.IO_INPUT, (1, 8))
    w = _b(p, "w", BufferKind.WEIGHT, (8, 8), source="w")
    y = _b(p, "y", BufferKind.ACTIVATION, (1, 8))
    o = _b(p, "o", BufferKind.IO_OUTPUT, (1, 8))
    c = p.new_counter().id
    cout = p.new_counter().id
    for i in range(4):
        p.add_task(InstructionKind.GEMV_TILE, [x, w], [y], out_counter=c,
                   params={"K": 8, "N_tile": 2, "n_off": i * 2}, label=f"tile{i}")
    p.add_task(InstructionKind.COPY, [y], [o], out_counter=cout, waits=[Wait(c, 2)], label="partial")
    res = validate(p)
    assert not res.ok and any("RACE" in e or "all-join" in e for e in res.errors), res.report()


def test_two_producers_one_consumer_partial_is_rejected():
    """(A) two distinct producers share a counter; consumer waits threshold=1 -> wrong producer."""
    p = _p()
    x = _b(p, "x", BufferKind.IO_INPUT, (1, 8))
    w = _b(p, "w", BufferKind.WEIGHT, (8,), source="w")
    ha = _b(p, "ha", BufferKind.ACTIVATION, (1, 8))
    hb = _b(p, "hb", BufferKind.ACTIVATION, (1, 8))
    o = _b(p, "o", BufferKind.IO_OUTPUT, (1, 8))
    c = p.new_counter().id
    cout = p.new_counter().id
    p.add_task(InstructionKind.RMSNORM, [x, w], [ha], out_counter=c, params={"eps": 1e-6, "hidden": 8}, label="pa")
    p.add_task(InstructionKind.RMSNORM, [x, w], [hb], out_counter=c, params={"eps": 1e-6, "hidden": 8}, label="pb")
    p.add_task(InstructionKind.COPY, [hb], [o], out_counter=cout, waits=[Wait(c, 1)], label="reads_hb")
    assert not validate(p).ok


def test_missing_happens_before_edge_is_rejected():
    """(B) consumer reads activation A but waits only on an unrelated counter -> data race."""
    p = _p()
    x = _b(p, "x", BufferKind.IO_INPUT, (1, 8))
    w = _b(p, "w", BufferKind.WEIGHT, (8,), source="w")
    actA = _b(p, "actA", BufferKind.ACTIVATION, (1, 8))
    actB = _b(p, "actB", BufferKind.ACTIVATION, (1, 8))
    o = _b(p, "o", BufferKind.IO_OUTPUT, (1, 8))
    c0 = p.new_counter().id
    c1 = p.new_counter().id
    cout = p.new_counter().id
    p.add_task(InstructionKind.RMSNORM, [x, w], [actA], out_counter=c0, params={"eps": 1e-6, "hidden": 8}, label="t0")
    p.add_task(InstructionKind.COPY, [x], [actB], out_counter=c1, label="t1")
    # reads actA but waits ONLY on c1 (t1), no path t0 -> t2
    p.add_task(InstructionKind.ADD, [actA, actB], [o], out_counter=cout, waits=[Wait(c1, 1)], label="t2")
    res = validate(p)
    assert not res.ok and any("RACE" in e for e in res.errors), res.report()


def test_kv_read_before_append_is_rejected():
    """(E) a read of a KV cache that is written this pass needs the KV_APPEND happens-before."""
    p = _p()
    newkv = _b(p, "newkv", BufferKind.IO_INPUT, (1, 2, 16))  # this token's k/v, computed upstream
    kv = _b(p, "kv", BufferKind.KV_CACHE, (256, 2, 16))
    o = _b(p, "o", BufferKind.IO_OUTPUT, (1, 2, 16))
    cprod = p.new_counter().id
    cout = p.new_counter().id
    # append (writer of kv), reading its own cache + an IO_INPUT is fine
    p.add_task(InstructionKind.KV_APPEND, [newkv, kv], [kv], out_counter=cprod,
               params={"pos": 5}, label="append")
    # a DIFFERENT reader of kv with NO wait on the append -> reads stale cache (race)
    p.add_task(InstructionKind.COPY, [kv], [o], out_counter=cout, label="reads_kv_unordered")
    res = validate(p)
    assert not res.ok and any("RACE" in e for e in res.errors), res.report()


def test_two_unordered_writers_one_buffer_is_rejected():
    """(WAW-1) two tasks WRITE the same buffer with no shared counter and no happens-before edge
    between them -> write-after-write RACE (the headline race-freedom hole)."""
    p = _p()
    x = _b(p, "x", BufferKind.IO_INPUT, (1, 8))
    w = _b(p, "w", BufferKind.WEIGHT, (8,), source="w")
    a = _b(p, "a", BufferKind.ACTIVATION, (1, 8))
    o = _b(p, "o", BufferKind.IO_OUTPUT, (1, 8))
    c0 = p.new_counter().id
    c1 = p.new_counter().id
    cout = p.new_counter().id
    # two whole-buffer writers of `a`, distinct counters, no edge between them
    p.add_task(InstructionKind.RMSNORM, [x, w], [a], out_counter=c0,
               params={"eps": 1e-6, "hidden": 8}, label="wa")
    p.add_task(InstructionKind.RMSNORM, [x, w], [a], out_counter=c1,
               params={"eps": 1e-6, "hidden": 8}, label="wb")
    # a reader that joins both writers (so RAW is satisfied), the only remaining flaw is WAW
    p.add_task(InstructionKind.COPY, [a], [o], out_counter=cout,
               waits=[Wait(c0, 1), Wait(c1, 1)], label="reader")
    res = validate(p)
    assert not res.ok and any("WAW" in e and "RACE" in e for e in res.errors), res.report()


def test_fully_overlapping_gemv_tiles_is_rejected():
    """(WAW-2) two GEMV tiles with the SAME n_off/N_tile under one counter write the SAME columns
    -> overlapping multi-writer footprints -> REJECTED."""
    p = _p()
    x = _b(p, "x", BufferKind.IO_INPUT, (1, 8))
    w = _b(p, "w", BufferKind.WEIGHT, (8, 8), source="w")
    y = _b(p, "y", BufferKind.ACTIVATION, (1, 8))
    o = _b(p, "o", BufferKind.IO_OUTPUT, (1, 8))
    c = p.new_counter().id
    cout = p.new_counter().id
    for i in range(2):
        p.add_task(InstructionKind.GEMV_TILE, [x, w], [y], out_counter=c,
                   params={"K": 8, "N_tile": 4, "n_off": 0}, label=f"tile{i}")  # same range!
    p.add_task(InstructionKind.COPY, [y], [o], out_counter=cout, waits=[Wait(c, 2)], label="reader")
    res = validate(p)
    assert not res.ok and any("WAW" in e and "RACE" in e for e in res.errors), res.report()


def test_disjoint_gemv_tiles_under_one_counter_is_accepted():
    """(WAW-3) the COMMON, legitimate case: disjoint column tiles under one shared counter, joined
    all-of-N by the reader -> ACCEPTED (no false reject)."""
    p = _p()
    x = _b(p, "x", BufferKind.IO_INPUT, (1, 8))
    w = _b(p, "w", BufferKind.WEIGHT, (8, 8), source="w")
    y = _b(p, "y", BufferKind.ACTIVATION, (1, 8))
    o = _b(p, "o", BufferKind.IO_OUTPUT, (1, 8))
    c = p.new_counter().id
    cout = p.new_counter().id
    for i in range(4):
        p.add_task(InstructionKind.GEMV_TILE, [x, w], [y], out_counter=c,
                   params={"K": 8, "N_tile": 2, "n_off": i * 2}, label=f"tile{i}")  # disjoint
    p.add_task(InstructionKind.COPY, [y], [o], out_counter=cout, waits=[Wait(c, 4)], label="reader")
    res = validate(p)
    assert res.ok, res.report()
    assert not any("WAW" in e for e in res.errors), res.report()


def test_toy_lowering_still_accepted_no_false_waw_reject():
    """(WAW-4) a real end-to-end toy lowering (which emits disjoint-tiled multi-writer GEMV for
    every projection) must STILL validate, the WAW check must not false-reject the compiler."""
    from models.toy import make_toy  # noqa: PLC0415, optional torch dep, imported lazily
    from schedule.graph import from_toy  # noqa: PLC0415
    from schedule.lower import lower  # noqa: PLC0415

    p = lower(from_toy(make_toy()))
    # sanity: the toy really does exercise disjoint-tiled multi-writers
    multi = {b: ws for b, ws in p.writers_by_buffer().items() if len(ws) > 1}
    assert multi, "expected the toy lowering to produce multi-writer (tiled) buffers"
    res = validate(p)
    assert res.ok, res.report()
    assert not any("WAW" in e for e in res.errors), res.report()


def test_capacity_overflow_is_rejected():
    """(H) more inputs/waits than the fixed POD record can hold must be rejected, not truncated."""
    p = _p()
    ins = [_b(p, f"in{i}", BufferKind.WEIGHT, (4,), source=f"in{i}") for i in range(ABI_MAX_INPUTS + 1)]
    o = _b(p, "o", BufferKind.IO_OUTPUT, (4,))
    c = p.new_counter().id
    p.add_task(InstructionKind.ALLREDUCE_SHARD, ins, [o], out_counter=c, label="too_wide")
    assert not validate(p).ok


def test_huge_cycle_does_not_crash_validator():
    """_describe_cycle must be iterative: a 5000-node cycle returns REJECTED, never RecursionError."""
    p = _p()
    o = _b(p, "o", BufferKind.IO_OUTPUT, (1,))
    n = 5000
    counters = [p.new_counter().id for _ in range(n)]
    for i in range(n):
        prev = counters[(i - 1) % n]
        outs = [o] if i == 0 else [_b(p, f"a{i}", BufferKind.ACTIVATION, (1,))]
        p.add_task(InstructionKind.COPY, [o], outs, out_counter=counters[i],
                   waits=[Wait(prev, 1)], label=f"n{i}")
    res = validate(p)  # must return, not raise
    assert not res.ok and any("CYCLE" in e for e in res.errors)


def test_additive_fields_load_without_crashing():
    """from_dict must tolerate unknown (newer) fields in target/config (additive compatibility)."""
    p = _p()
    o = _b(p, "o", BufferKind.IO_OUTPUT, (1,))
    c = p.new_counter().id
    p.add_task(InstructionKind.COPY, [o], [o], out_counter=c, label="x")
    d = p.to_dict()
    d["target"]["future_tensor_mem_kb"] = 999          # field a newer writer added
    d["config"] = {"tiling": {}, "made_up_search_dim": 7}
    p2 = MegakernelProgram.from_dict(d)                 # must not raise
    assert p2.target.name == "rtx5090"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK  {name}")
    print("\nALL VALIDATOR RACE/SOUNDNESS REGRESSIONS PASS")

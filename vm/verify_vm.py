"""
AMK, VM CORRECTNESS PROOF (the trust foundation)
=================================================

Proves, with REAL numerics and NO GPU required, that the megakernel runtime semantics are
correct and safe:

  1. The VM **refuses to load** a schedule the validator rejects (cycle / unsatisfiable wait).
  2. A hand-written 3-instruction DAG (rmsnorm -> tiled gemv -> residual add) executes under
     counter sync and matches a direct eager computation, including a **tiled** matvec whose
     two tiles share one counter (consumer waits threshold=2), the core fan-in pattern.
  3. A full SwiGLU MLP block (rmsnorm -> gate/up gemv tiles -> silu*up -> down gemv tiles ->
     residual) matches eager `models.toy.ToyMLP`. Proves multi-stage fusion on real weights.
  4. Deadlock-freedom is confirmed dynamically (simulate_counters leaves no task stuck) and the
     static validator agrees.

The CUDA VM (vm/scheduler.cu) is conformance-tested against THIS reference (see
instructions/verify_inst.py and tests/test_cuda_vm.py). If they ever disagree, the CUDA side
is wrong by definition.

Run:  uv run python vm/verify_vm.py        (also a pytest module)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from models.toy import ToyConfig, ToyMLP, _rmsnorm  # noqa: E402
from schedule.ir import (  # noqa: E402
    BufferKind, DType, InstructionKind, MegakernelProgram, TARGETS, Wait, validate,
)
from vm.reference_vm import ReferenceVM  # noqa: E402

DT = DType.F32
TDT = torch.float32


def _tiled_gemv(p, x_buf, w_buf, out_buf, K, N, n_tiles, wait, counter, label):
    """Emit n_tiles GEMV_TILE tasks writing disjoint column ranges of out_buf, all incrementing
    `counter`. A consumer then waits (counter, n_tiles). Returns nothing (mutates p)."""
    tile = (N + n_tiles - 1) // n_tiles
    for i in range(n_tiles):
        n_off = i * tile
        n_tile = min(tile, N - n_off)
        if n_tile <= 0:
            break
        p.add_task(InstructionKind.GEMV_TILE, [x_buf, w_buf], [out_buf], out_counter=counter,
                   waits=list(wait), params={"K": K, "N_tile": n_tile, "n_off": n_off},
                   label=f"{label}[t{i}]", est_bytes=K * n_tile * 2, est_flops=2 * K * n_tile)


def _new(p, name, kind, shape, source=None):
    return p.new_buffer(name, kind, DT, tuple(shape), source=source).id


# ======================================================================================
def test_refuses_invalid_schedule():
    p = MegakernelProgram(meta={"model": "neg", "gpu": "rtx5090"}, target=TARGETS["rtx5090"])
    x = _new(p, "x", BufferKind.IO_INPUT, (1, 8))
    w = _new(p, "w", BufferKind.WEIGHT, (8,), source="w")
    o = _new(p, "o", BufferKind.IO_OUTPUT, (1, 8))
    c0 = p.new_counter().id
    c1 = p.new_counter().id
    # cycle: t0 waits on c1 (produced by t1), t1 waits on c0 (produced by t0)
    p.add_task(InstructionKind.RMSNORM, [x, w], [o], out_counter=c0, waits=[Wait(c1, 1)],
               params={"eps": 1e-6, "hidden": 8}, label="a")
    p.add_task(InstructionKind.COPY, [o], [o], out_counter=c1, waits=[Wait(c0, 1)], label="b")
    assert not validate(p).ok
    try:
        ReferenceVM(p, weights={"w": torch.ones(8)})
        raise AssertionError("VM must refuse to load an invalid (cyclic) schedule")
    except ValueError as e:
        assert "refuses" in str(e)


def test_three_instruction_dag_matches_eager():
    torch.manual_seed(1)
    K = 16
    x = torch.randn(1, K, dtype=TDT)
    norm_w = torch.randn(K, dtype=TDT)
    proj_w = torch.randn(K, K, dtype=TDT)  # [N, K] torch Linear layout, N=K
    weights = {"norm.w": norm_w, "proj.w": proj_w}

    p = MegakernelProgram(meta={"model": "dag3", "gpu": "rtx5090"}, target=TARGETS["rtx5090"])
    bx = _new(p, "x", BufferKind.IO_INPUT, (1, K))
    bnw = _new(p, "norm.w", BufferKind.WEIGHT, (K,), source="norm.w")
    bpw = _new(p, "proj.w", BufferKind.WEIGHT, (K, K), source="proj.w")
    bh = _new(p, "h", BufferKind.ACTIVATION, (1, K))
    by = _new(p, "y", BufferKind.ACTIVATION, (1, K))
    bo = _new(p, "out", BufferKind.IO_OUTPUT, (1, K))
    c_norm = p.new_counter("norm").id
    c_proj = p.new_counter("proj").id
    c_add = p.new_counter("add").id

    p.add_task(InstructionKind.RMSNORM, [bx, bnw], [bh], out_counter=c_norm,
               params={"eps": 1e-6, "hidden": K}, label="rmsnorm")
    _tiled_gemv(p, bh, bpw, by, K=K, N=K, n_tiles=2, wait=[Wait(c_norm, 1)],
                counter=c_proj, label="proj")
    p.add_task(InstructionKind.ADD, [by, bx], [bo], out_counter=c_add,
               waits=[Wait(c_proj, 2)], label="residual")  # threshold=2: both gemv tiles

    res = validate(p)
    assert res.ok, res.report()
    _, stuck = p.simulate_counters()
    assert stuck == []

    out = ReferenceVM(p, weights).run({"x": x})
    eager = (x @ proj_w.t()) + x  # proj of rmsnorm... wait, gemv consumes h=rmsnorm(x)
    eager = (_rmsnorm(x, norm_w, 1e-6) @ proj_w.t()) + x
    torch.testing.assert_close(out["out"], eager, rtol=1e-5, atol=1e-5)


def test_mlp_block_matches_eager():
    torch.manual_seed(2)
    cfg = ToyConfig(hidden=64, intermediate=128)
    mlp = ToyMLP(cfg).to(TDT).eval()
    post_norm = torch.randn(cfg.hidden, dtype=TDT)
    x = torch.randn(1, cfg.hidden, dtype=TDT)

    with torch.no_grad():
        eager = x + mlp(_rmsnorm(x, post_norm, cfg.rms_eps))

    sd = {f"mlp.{k}": v for k, v in mlp.state_dict().items()}
    sd["post_norm"] = post_norm
    H, inter = cfg.hidden, cfg.intermediate

    p = MegakernelProgram(meta={"model": "mlp", "gpu": "rtx5090"}, target=TARGETS["rtx5090"])
    bx = _new(p, "x", BufferKind.IO_INPUT, (1, H))
    bnw = _new(p, "post_norm", BufferKind.WEIGHT, (H,), source="post_norm")
    bgw = _new(p, "gate", BufferKind.WEIGHT, (inter, H), source="mlp.gate_proj.weight")
    buw = _new(p, "up", BufferKind.WEIGHT, (inter, H), source="mlp.up_proj.weight")
    bdw = _new(p, "down", BufferKind.WEIGHT, (H, inter), source="mlp.down_proj.weight")
    bxn = _new(p, "xn", BufferKind.ACTIVATION, (1, H))
    bg = _new(p, "g", BufferKind.ACTIVATION, (1, inter))
    bu = _new(p, "u", BufferKind.ACTIVATION, (1, inter))
    bact = _new(p, "act", BufferKind.ACTIVATION, (1, inter))
    bd = _new(p, "d", BufferKind.ACTIVATION, (1, H))
    bo = _new(p, "out", BufferKind.IO_OUTPUT, (1, H))

    c_norm = p.new_counter().id
    c_gate = p.new_counter().id
    c_up = p.new_counter().id
    c_act = p.new_counter().id
    c_down = p.new_counter().id
    c_res = p.new_counter().id

    p.add_task(InstructionKind.RMSNORM, [bx, bnw], [bxn], out_counter=c_norm,
               params={"eps": cfg.rms_eps, "hidden": H}, label="post_norm")
    _tiled_gemv(p, bxn, bgw, bg, K=H, N=inter, n_tiles=4, wait=[Wait(c_norm, 1)], counter=c_gate, label="gate")
    _tiled_gemv(p, bxn, buw, bu, K=H, N=inter, n_tiles=4, wait=[Wait(c_norm, 1)], counter=c_up, label="up")
    p.add_task(InstructionKind.SILU_MUL, [bg, bu], [bact], out_counter=c_act,
               waits=[Wait(c_gate, 4), Wait(c_up, 4)], label="swiglu")
    _tiled_gemv(p, bact, bdw, bd, K=inter, N=H, n_tiles=4, wait=[Wait(c_act, 1)], counter=c_down, label="down")
    p.add_task(InstructionKind.ADD, [bd, bx], [bo], out_counter=c_res,
               waits=[Wait(c_down, 4)], label="residual")

    res = validate(p)
    assert res.ok, res.report()
    _, stuck = p.simulate_counters()
    assert stuck == []

    out = ReferenceVM(p, sd).run({"x": x})
    torch.testing.assert_close(out["out"], eager, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    test_refuses_invalid_schedule()
    print("[1/4] VM refuses invalid (cyclic) schedule .......... OK")
    test_three_instruction_dag_matches_eager()
    print("[2/4] 3-instruction DAG (tiled gemv) == eager ....... OK")
    test_mlp_block_matches_eager()
    print("[3/4] SwiGLU MLP block == eager ToyMLP .............. OK")
    # 4: deadlock-freedom already asserted via simulate_counters in each test above
    print("[4/4] deadlock-freedom (no stuck tasks) confirmed ... OK")
    print("\nVM REFERENCE SEMANTICS VERIFIED (no GPU required).")

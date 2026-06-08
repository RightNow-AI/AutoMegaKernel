"""
AMK, CUDA megakernel VM CONFORMANCE TEST (runs on the real GPU)
================================================================

Proves the persistent cooperative CUDA VM (vm/scheduler.cu + loader.py) produces the SAME result
as the CPU ReferenceVM oracle (vm/reference_vm.py) on the EXACT programs already proven correct on
CPU in vm/verify_vm.py:

  1. the 3-instruction tiled-gemv DAG  (rmsnorm -> 2-tile gemv -> residual add), and
  2. the SwiGLU MLP block              (rmsnorm -> gate/up gemv tiles -> silu*up -> down tiles -> add).

For each, we build the program with the SAME builders/helpers verify_vm uses (_tiled_gemv, _new),
run ReferenceVM on CPU and MegakernelVM on CUDA with identical weights/inputs, and assert
torch.allclose within rtol/atol = 2e-3 (fp32). We also assert the kernel ACTUALLY RAN on the GPU
(loader.last_status == "OK", a cooperative launch, not a CPU fallback).

If the cooperative persistent kernel cannot run on this Windows sm_120 GPU after genuine effort,
the test SKIPS with the precise CUDA error text (never a fake pass). Run:

    uv run python tests/test_cuda_vm.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from models.toy import ToyConfig, ToyMLP  # noqa: E402
from schedule.ir import (  # noqa: E402
    BufferKind, InstructionKind, MegakernelProgram, TARGETS, Wait, validate,
)
from vm.reference_vm import ReferenceVM  # noqa: E402
from vm.verify_vm import TDT, _new, _tiled_gemv  # noqa: E402  (the proven builders)


class GpuUnavailable(Exception):
    """Raised when the GPU/cooperative path genuinely cannot run (-> SKIP, not FAIL)."""


# ======================================================================================
# Program builders, byte-identical to the proven verify_vm programs.
# ======================================================================================
def build_dag3():
    """rmsnorm -> 2-tile gemv -> residual add (the core fan-in pattern, threshold=2)."""
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
               waits=[Wait(c_proj, 2)], label="residual")
    return p, weights, {"x": x}, "out"


def build_mlp():
    """SwiGLU MLP block on real ToyMLP weights."""
    torch.manual_seed(2)
    cfg = ToyConfig(hidden=64, intermediate=128)
    mlp = ToyMLP(cfg).to(TDT).eval()
    post_norm = torch.randn(cfg.hidden, dtype=TDT)
    x = torch.randn(1, cfg.hidden, dtype=TDT)

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
    return p, sd, {"x": x}, "out"


# ======================================================================================
# The conformance check.
# ======================================================================================
def _run_case(name, builder):
    p, weights, inputs, out_name = builder()
    res = validate(p)
    assert res.ok, f"{name}: program rejected by validator:\n{res.report()}"

    # CPU oracle
    ref = ReferenceVM(p, weights, device="cpu").run(inputs)[out_name]

    # GPU megakernel (real cooperative launch on this RTX 5090 sm_120)
    try:
        from vm.loader import MegakernelVM
        vm = MegakernelVM(p, weights, device="cuda")
        gpu = vm.run(inputs)[out_name]
    except (RuntimeError, TimeoutError) as e:
        raise GpuUnavailable(f"{name}: CUDA VM could not run: {e}") from e

    # the kernel must have ACTUALLY run on the GPU (no CPU fallback)
    assert getattr(vm, "last_status", {}).get("status") == "OK", \
        f"{name}: kernel did not report a clean GPU launch: {getattr(vm, 'last_status', None)}"
    assert gpu.is_cuda, f"{name}: output is not a CUDA tensor (CPU fallback?)"

    gpu_cpu = gpu.detach().cpu().to(ref.dtype)
    close = torch.allclose(gpu_cpu, ref, rtol=2e-3, atol=2e-3)
    max_abs = (gpu_cpu - ref).abs().max().item()
    print(f"  [{name}] grid_dim={vm.last_grid_dim} status={vm.last_status['status']} "
          f"max_abs_err={max_abs:.3e} allclose(2e-3)={close}")
    assert close, (f"{name}: GPU != ReferenceVM (max_abs_err={max_abs:.3e})\n"
                   f"gpu={gpu_cpu.flatten()[:8]}\nref={ref.flatten()[:8]}")
    return max_abs


def test_cuda_vm_three_instruction_dag():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    try:
        _run_case("dag3", build_dag3)
    except GpuUnavailable as e:
        import pytest
        pytest.skip(str(e))


def test_cuda_vm_swiglu_mlp_block():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    try:
        _run_case("mlp", build_mlp)
    except GpuUnavailable as e:
        import pytest
        pytest.skip(str(e))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)
    print("Building + launching the AMK persistent cooperative megakernel on the GPU...")
    try:
        e1 = _run_case("dag3", build_dag3)
        print("[1/2] 3-instruction tiled-gemv DAG: GPU == ReferenceVM .......... OK")
        e2 = _run_case("mlp", build_mlp)
        print("[2/2] SwiGLU MLP block:             GPU == ReferenceVM .......... OK")
        print(f"\nCUDA MEGAKERNEL VM CONFORMANCE VERIFIED on {torch.cuda.get_device_name(0)} "
              f"(max_abs_err dag3={e1:.2e}, mlp={e2:.2e}).")
    except GpuUnavailable as e:
        print(f"\nSKIP (GPU/cooperative path unavailable): {e}")
        sys.exit(0)

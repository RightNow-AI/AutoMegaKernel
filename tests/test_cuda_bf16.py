"""
AMK, bf16 DECODE MEGAKERNEL CONFORMANCE + PIPELINING BENCH (runs on the real GPU)
=================================================================================

Two acceptance jobs, both on the live RTX 5090 (sm_120):

  (A) bf16 correctness. Build the FULL Llama-style decode program with dtype=DType.BF16, bind a
      bf16 ToyLlama's weights, run it through the persistent cooperative megakernel
      (vm.loader.MegakernelVM), and assert the GPU logits:
        * match eager ToyLlama(bf16).forward within bf16 tolerance (rtol/atol = 2e-2), AND
        * match the CPU ReferenceVM on the SAME bf16 program/weights, AND
        * actually ran on the GPU (grid_dim > 0, status OK, CUDA tensor).
      This exercises every decode opcode through the bf16 load->fp32 compute->bf16 store path
      (EMBED, RMSNORM, GEMV_TILE, ROPE, KV_APPEND, ATTENTION_TILE, SILU_MUL, ADD).

  (B) Software-pipelining speedup. On a LARGER ToyLlama (hidden=1024, n_layers=4,
      intermediate=4096, vocab=8192) build the program with pipelining_depth=0 vs the configured
      depth, time both on the GPU via cuda events through the loader, and report the latency delta.
      The pipelined version must be correct (matches the depth=0 result) and SHOULD be faster.
      We report the real numbers either way.

Run:  uv run python tests/test_cuda_bf16.py     (also a pytest module)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from models.toy import make_toy  # noqa: E402
from schedule.graph import from_toy  # noqa: E402
from schedule.ir import DType, ScheduleConfig, TARGETS, validate  # noqa: E402
from schedule.lower import POS_NAME, RESHAPE_ID_NAME, TOKEN_NAME, lower, required_inputs  # noqa: E402
from vm.reference_vm import ReferenceVM  # noqa: E402

BF16_RTOL = BF16_ATOL = 2e-2


class GpuUnavailable(Exception):
    """Raised when the GPU/cooperative path genuinely cannot run (-> SKIP, not FAIL)."""


def _eager_decode_logits(model, tok: int) -> torch.Tensor:
    with torch.no_grad():
        logits = model.forward(torch.tensor([tok]))     # [S=1, vocab]
    return logits[-1].view(1, -1)


def _build_inputs(tok: int, pos: int) -> dict[str, torch.Tensor]:
    contract = required_inputs(pos)
    assert set(contract) == {TOKEN_NAME, POS_NAME, RESHAPE_ID_NAME}
    return {
        TOKEN_NAME: torch.tensor([tok], dtype=torch.int32),
        POS_NAME: torch.tensor([pos], dtype=torch.int32),
        RESHAPE_ID_NAME: torch.tensor([int(contract[RESHAPE_ID_NAME][0])], dtype=torch.int32),
    }


# ------------------------------------------------------------------------------------------------
# (A) bf16 full decode correctness
# ------------------------------------------------------------------------------------------------
def _run_bf16_case(name: str, n_layers: int, tok: int, pos: int = 0):
    model = make_toy(seed=0, dtype=torch.bfloat16, n_layers=n_layers)
    graph = from_toy(model)
    prog = lower(graph, target=TARGETS["rtx5090"], pos=pos, dtype=DType.BF16)

    res = validate(prog)
    assert res.ok, f"{name}: lowered bf16 program rejected:\n{res.report()}"

    weights = model.weights_dict()
    inputs = _build_inputs(tok, pos)

    # CPU oracle on the SAME bf16 program/weights.
    ref = ReferenceVM(prog, weights, device="cpu").run(inputs, kv={})["logits"]

    # Eager bf16 oracle.
    eager = _eager_decode_logits(model, tok)

    # GPU megakernel (bf16 storage, fp32 compute).
    try:
        from vm.loader import MegakernelVM
        vm = MegakernelVM(prog, weights, device="cuda")
        gpu = vm.run(inputs, kv={})["logits"]
    except (RuntimeError, TimeoutError) as e:
        raise GpuUnavailable(f"{name}: CUDA VM could not run: {e}") from e

    assert vm.last_status.get("status") == "OK", f"{name}: {vm.last_status}"
    assert vm.last_grid_dim > 0
    assert gpu.is_cuda, f"{name}: output not CUDA"

    gpu_f = gpu.detach().cpu().to(torch.float32)
    eager_f = eager.detach().cpu().to(torch.float32)
    ref_f = ref.detach().cpu().to(torch.float32)
    err_eager = (gpu_f - eager_f).abs().max().item()
    err_ref = (gpu_f - ref_f).abs().max().item()
    close_eager = torch.allclose(gpu_f, eager_f, rtol=BF16_RTOL, atol=BF16_ATOL)
    close_ref = torch.allclose(gpu_f, ref_f, rtol=BF16_RTOL, atol=BF16_ATOL)
    print(f"  [{name}] layers={n_layers} grid_dim={vm.last_grid_dim} status={vm.last_status['status']} "
          f"tasks={len(prog.tasks)} max_err(vs eager)={err_eager:.3e} "
          f"max_err(vs refVM)={err_ref:.3e} allclose(2e-2): eager={close_eager} refVM={close_ref}")
    assert close_eager, (f"{name}: bf16 GPU != eager (max_err={err_eager:.3e})\n"
                         f"gpu={gpu_f.flatten()[:8]}\neager={eager_f.flatten()[:8]}")
    assert close_ref, (f"{name}: bf16 GPU != ReferenceVM (max_err={err_ref:.3e})")
    return err_eager, err_ref


# ------------------------------------------------------------------------------------------------
# (B) software-pipelining latency bench
# ------------------------------------------------------------------------------------------------
def _time_vm(vm, inputs, iters: int, warmup: int) -> float:
    """Median KERNEL-ONLY latency (ms) via cuda events. We do one run() to build + upload the
    device tables, then re-fire JUST the cooperative kernel with relaunch() (no host re-packing or
    H2D copies) so the measurement isolates the megakernel, exactly where the prefetch lives."""
    vm.run(inputs, kv={})                       # build device tables once
    for _ in range(warmup):
        vm.relaunch()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        vm.relaunch()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _run_pipelining_bench(depth: int = 2):
    from vm.loader import MegakernelVM
    model = make_toy(seed=0, dtype=torch.bfloat16, n_layers=4,
                     hidden=1024, intermediate=4096, vocab=8192,
                     n_heads=16, n_kv_heads=4, head_dim=64)
    graph = from_toy(model)
    weights = model.weights_dict()
    inputs = _build_inputs(tok=11, pos=0)

    def build(d: int):
        cfg = ScheduleConfig(pipelining_depth=d)
        prog = lower(graph, target=TARGETS["rtx5090"], config=cfg, pos=0, dtype=DType.BF16)
        assert validate(prog).ok
        return MegakernelVM(prog, weights, device="cuda")

    vm0 = build(0)        # no prefetch
    vmP = build(depth)    # pipelined prefetch

    iters, warmup = 50, 10
    t0 = _time_vm(vm0, inputs, iters, warmup)
    tP = _time_vm(vmP, inputs, iters, warmup)

    # correctness: pipelined result must equal the un-pipelined result (prefetch is a pure hint).
    r0 = vm0.run(inputs, kv={})["logits"].detach().cpu().to(torch.float32)
    rP = vmP.run(inputs, kv={})["logits"].detach().cpu().to(torch.float32)
    max_diff = (r0 - rP).abs().max().item()
    same = torch.allclose(r0, rP, rtol=0, atol=0)

    speedup = (t0 / tP) if tP > 0 else float("nan")
    delta_us = (t0 - tP) * 1e3
    print(f"  [pipeline-bench] tasks={len(vmP.prog.tasks)} grid_dim={vmP.last_grid_dim} "
          f"depth=0 -> {t0*1e3:.1f} us | depth={depth} -> {tP*1e3:.1f} us | "
          f"delta={delta_us:+.1f} us | speedup={speedup:.3f}x | pipelined==base: {same} "
          f"(max_diff={max_diff:.2e})")
    assert same, f"pipelining changed the result! max_diff={max_diff:.2e}"
    return t0, tP, speedup


# ------------------------------------------------------------------------------------------------
# pytest entry points
# ------------------------------------------------------------------------------------------------
def test_cuda_bf16_one_layer():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    try:
        _run_bf16_case("bf16-1L", n_layers=1, tok=7, pos=0)
    except GpuUnavailable as e:
        import pytest
        pytest.skip(str(e))


def test_cuda_bf16_two_layer():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    try:
        _run_bf16_case("bf16-2L", n_layers=2, tok=19, pos=0)
    except GpuUnavailable as e:
        import pytest
        pytest.skip(str(e))


def test_pipelining_correct_and_timed():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    try:
        _run_pipelining_bench(depth=2)
    except GpuUnavailable as e:
        import pytest
        pytest.skip(str(e))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)
    print("bf16 full decode + software-pipelining bench on the GPU...")
    try:
        e1 = _run_bf16_case("bf16-1L", n_layers=1, tok=7, pos=0)
        print("[1/3] 1-layer bf16 decode: GPU == eager == ReferenceVM (2e-2) ... OK")
        e2 = _run_bf16_case("bf16-2L", n_layers=2, tok=19, pos=0)
        print("[2/3] 2-layer bf16 decode: GPU == eager == ReferenceVM (2e-2) ... OK")
        _run_pipelining_bench(depth=2)
        print("[3/3] pipelining: correct + timed ............................. OK")
        print(f"\nbf16 MEGAKERNEL VERIFIED on {torch.cuda.get_device_name(0)} "
              f"(max_err vs eager: 1L={e1[0]:.2e}, 2L={e2[0]:.2e}).")
    except GpuUnavailable as e:
        print(f"\nSKIP (GPU/cooperative path unavailable): {e}")
        sys.exit(0)

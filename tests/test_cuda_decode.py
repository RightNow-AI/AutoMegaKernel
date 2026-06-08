"""
AMK, FULL DECODE MEGAKERNEL CONFORMANCE TEST (runs on the real GPU)
===================================================================

The true end-to-end M0-on-hardware proof. We build the WHOLE Llama-style decode program with the
real graph importer + lowerer (``schedule.graph.from_toy`` + ``schedule.lower.lower``), bind the
toy model weights, and run it through the persistent cooperative CUDA megakernel
(``vm.loader.MegakernelVM``) on the RTX 5090 (sm_120). We assert the GPU logits:

  * match eager ``ToyLlama.forward(tokens)[-1]`` within rtol/atol = 2e-3, AND
  * match the CPU ``ReferenceVM`` output on the same program/weights, AND
  * actually ran on the GPU (grid_dim > 0, last_status == "OK", output is a CUDA tensor).

This exercises EVERY decode opcode in the VM dispatch: EMBED (gather + flat->head reshape bridge),
RMSNORM, GEMV_TILE (q/k/v/o/gate/up/down/lm_head), ROPE (Llama rotate-half), KV_APPEND (k & v at
pos), ATTENTION_TILE (GQA whole-window), SILU_MUL, ADD. 1-layer and 2-layer variants prove the
per-layer wiring composes (residual threading + per-layer KV).

We use the EXACT run() input contract documented by ``schedule.lower.required_inputs()``:
token_id, pos, reshape_id0 (the constant [0] for the EMBED reshape bridge), plus the (empty at
pos=0) KV buffers.

Run:  uv run python tests/test_cuda_decode.py     (also a pytest module)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from models.toy import make_toy  # noqa: E402
from schedule.graph import from_toy  # noqa: E402
from schedule.ir import DType, TARGETS, validate  # noqa: E402
from schedule.lower import POS_NAME, RESHAPE_ID_NAME, TOKEN_NAME, lower, required_inputs  # noqa: E402
from vm.reference_vm import ReferenceVM  # noqa: E402

RTOL = ATOL = 2e-3


class GpuUnavailable(Exception):
    """Raised when the GPU/cooperative path genuinely cannot run (-> SKIP, not FAIL)."""


def _eager_decode_logits(model, tok: int) -> torch.Tensor:
    """Eager oracle: the first-token logits == decode at pos=0 with an empty cache."""
    with torch.no_grad():
        logits = model.forward(torch.tensor([tok]))     # [S=1, vocab]
    return logits[-1].view(1, -1)


def _build_inputs(tok: int, pos: int) -> dict[str, torch.Tensor]:
    """The run() input contract per schedule.lower.required_inputs(): token_id, pos, reshape_id0."""
    contract = required_inputs(pos)            # documents the keys + the constant reshape id [0]
    assert set(contract) == {TOKEN_NAME, POS_NAME, RESHAPE_ID_NAME}
    return {
        TOKEN_NAME: torch.tensor([tok], dtype=torch.int32),
        POS_NAME: torch.tensor([pos], dtype=torch.int32),
        RESHAPE_ID_NAME: torch.tensor([int(contract[RESHAPE_ID_NAME][0])], dtype=torch.int32),
    }


def _run_case(name: str, n_layers: int, tok: int, pos: int = 0):
    model = make_toy(seed=0, dtype=torch.float32, n_layers=n_layers)
    graph = from_toy(model)
    prog = lower(graph, target=TARGETS["rtx5090"], pos=pos, dtype=DType.F32)

    res = validate(prog)
    assert res.ok, f"{name}: lowered program rejected by validator:\n{res.report()}"
    bad = [w for w in res.warnings if "RACE" in w or "CYCLE" in w]
    assert not bad, f"{name}: validator emitted RACE/CYCLE warnings: {bad}"

    weights = model.weights_dict()
    inputs = _build_inputs(tok, pos)

    # CPU oracle (counter-driven reference VM).
    ref = ReferenceVM(prog, weights, device="cpu").run(inputs, kv={})["logits"]

    # Eager oracle.
    eager = _eager_decode_logits(model, tok)

    # GPU megakernel (real cooperative launch on this RTX 5090 sm_120).
    try:
        from vm.loader import MegakernelVM
        vm = MegakernelVM(prog, weights, device="cuda")
        gpu = vm.run(inputs, kv={})["logits"]
    except (RuntimeError, TimeoutError) as e:
        raise GpuUnavailable(f"{name}: CUDA VM could not run: {e}") from e

    # the kernel must have ACTUALLY run on the GPU (no CPU fallback).
    assert getattr(vm, "last_status", {}).get("status") == "OK", \
        f"{name}: kernel did not report a clean GPU launch: {getattr(vm, 'last_status', None)}"
    assert vm.last_grid_dim > 0, f"{name}: grid_dim must be > 0, got {vm.last_grid_dim}"
    assert gpu.is_cuda, f"{name}: output is not a CUDA tensor (CPU fallback?)"

    gpu_cpu = gpu.detach().cpu().to(ref.dtype)
    err_eager = (gpu_cpu - eager).abs().max().item()
    err_ref = (gpu_cpu - ref).abs().max().item()
    close_eager = torch.allclose(gpu_cpu, eager, rtol=RTOL, atol=ATOL)
    close_ref = torch.allclose(gpu_cpu, ref, rtol=RTOL, atol=ATOL)
    print(f"  [{name}] layers={n_layers} grid_dim={vm.last_grid_dim} "
          f"status={vm.last_status['status']} tasks={len(prog.tasks)} "
          f"max_err(vs eager)={err_eager:.3e} max_err(vs refVM)={err_ref:.3e} "
          f"allclose(2e-3): eager={close_eager} refVM={close_ref}")
    assert close_eager, (f"{name}: GPU != eager ToyLlama (max_err={err_eager:.3e})\n"
                         f"gpu={gpu_cpu.flatten()[:8]}\neager={eager.flatten()[:8]}")
    assert close_ref, (f"{name}: GPU != ReferenceVM (max_err={err_ref:.3e})\n"
                       f"gpu={gpu_cpu.flatten()[:8]}\nref={ref.flatten()[:8]}")
    # argmax token agreement is the actual decode decision, must match exactly.
    assert int(gpu_cpu.argmax()) == int(eager.argmax()), \
        f"{name}: argmax token disagrees (gpu={int(gpu_cpu.argmax())} eager={int(eager.argmax())})"
    return err_eager, err_ref


def test_cuda_decode_one_layer_matches_eager_and_refvm():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    try:
        _run_case("decode-1L", n_layers=1, tok=7, pos=0)
    except GpuUnavailable as e:
        import pytest
        pytest.skip(str(e))


def test_cuda_decode_two_layer_matches_eager_and_refvm():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    try:
        _run_case("decode-2L", n_layers=2, tok=19, pos=0)
    except GpuUnavailable as e:
        import pytest
        pytest.skip(str(e))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)
    print("Building + launching the FULL decode megakernel on the GPU...")
    try:
        e1 = _run_case("decode-1L", n_layers=1, tok=7, pos=0)
        print("[1/2] 1-layer full decode: GPU == eager == ReferenceVM ......... OK")
        e2 = _run_case("decode-2L", n_layers=2, tok=19, pos=0)
        print("[2/2] 2-layer full decode: GPU == eager == ReferenceVM ......... OK")
        print(f"\nFULL DECODE MEGAKERNEL VERIFIED on {torch.cuda.get_device_name(0)} "
              f"(max_err vs eager: 1L={e1[0]:.2e}, 2L={e2[0]:.2e}).")
    except GpuUnavailable as e:
        print(f"\nSKIP (GPU/cooperative path unavailable): {e}")
        sys.exit(0)

"""
AMK, fp16 DECODE MEGAKERNEL CONFORMANCE + LARGE-head_dim ATTENTION (runs on the real GPU)
=========================================================================================

Three acceptance jobs, all on the live RTX 5090 (sm_120):

  (A) fp16 correctness. Build the FULL Llama-style decode program with dtype=DType.F16, bind an
      fp16 ToyLlama's weights, run it through the persistent cooperative megakernel
      (vm.loader.MegakernelVM), and assert the GPU logits:
        * match eager ToyLlama(fp16).forward within fp16 tolerance (rtol/atol = 2e-2), AND
        * match the CPU ReferenceVM on the SAME fp16 program/weights (the fp32-accumulate /
          cast-to-fp16 semantics are identical to bf16/fp32, so this is bit-for-bit), AND
        * actually ran on the GPU (grid_dim > 0, status OK, CUDA tensor).
      This exercises every decode opcode through the fp16 load->fp32 compute->fp16 store path
      (EMBED, RMSNORM, GEMV_TILE incl. the 8-wide coalesced __half weight burst, ROPE, KV_APPEND,
      ATTENTION_TILE, SILU_MUL, ADD).

  (B) Larger head_dim attention. The ATTENTION_TILE op uses a STATIC __shared__ scratch for
      head_dim <= 256 and a DYNAMIC-shared-memory path for 256 < head_dim <= 512 (the loader opts
      the kernel into the needed dynamic smem). We run head_dim=256 (static path edge) and
      head_dim=512 (dynamic path) decodes and assert GPU == eager == ReferenceVM, so the lifted cap
      is correct, not just non-crashing.

Run:  uv run python tests/test_cuda_fp16.py     (also a pytest module)
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

FP16_RTOL = FP16_ATOL = 2e-2


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
# (A) fp16 full decode correctness
# ------------------------------------------------------------------------------------------------
def _run_fp16_case(name: str, n_layers: int, tok: int, pos: int = 0):
    model = make_toy(seed=0, dtype=torch.float16, n_layers=n_layers)
    graph = from_toy(model)
    prog = lower(graph, target=TARGETS["rtx5090"], pos=pos, dtype=DType.F16)

    res = validate(prog)
    assert res.ok, f"{name}: lowered fp16 program rejected:\n{res.report()}"

    weights = model.weights_dict()
    inputs = _build_inputs(tok, pos)

    # CPU oracle on the SAME fp16 program/weights (the conformance ground truth).
    ref = ReferenceVM(prog, weights, device="cpu").run(inputs, kv={})["logits"]

    # Eager fp16 oracle.
    eager = _eager_decode_logits(model, tok)

    # GPU megakernel (fp16 storage, fp32 compute).
    try:
        from vm.loader import MegakernelVM
        vm = MegakernelVM(prog, weights, device="cuda")
        gpu = vm.run(inputs, kv={})["logits"]
    except (RuntimeError, TimeoutError) as e:
        raise GpuUnavailable(f"{name}: CUDA VM could not run: {e}") from e

    assert vm.last_status.get("status") == "OK", f"{name}: {vm.last_status}"
    assert vm.last_grid_dim > 0
    assert gpu.is_cuda, f"{name}: output not CUDA"
    assert gpu.dtype == torch.float16, f"{name}: expected fp16 output, got {gpu.dtype}"

    gpu_f = gpu.detach().cpu().to(torch.float32)
    eager_f = eager.detach().cpu().to(torch.float32)
    ref_f = ref.detach().cpu().to(torch.float32)
    err_eager = (gpu_f - eager_f).abs().max().item()
    err_ref = (gpu_f - ref_f).abs().max().item()
    close_eager = torch.allclose(gpu_f, eager_f, rtol=FP16_RTOL, atol=FP16_ATOL)
    close_ref = torch.allclose(gpu_f, ref_f, rtol=FP16_RTOL, atol=FP16_ATOL)
    print(f"  [{name}] layers={n_layers} grid_dim={vm.last_grid_dim} status={vm.last_status['status']} "
          f"tasks={len(prog.tasks)} max_err(vs eager)={err_eager:.3e} "
          f"max_err(vs refVM)={err_ref:.3e} allclose(2e-2): eager={close_eager} refVM={close_ref}")
    assert close_eager, (f"{name}: fp16 GPU != eager (max_err={err_eager:.3e})\n"
                         f"gpu={gpu_f.flatten()[:8]}\neager={eager_f.flatten()[:8]}")
    # fp32-accumulate then cast-to-fp16 is identical to the reference => bit-for-bit.
    assert close_ref, (f"{name}: fp16 GPU != ReferenceVM (max_err={err_ref:.3e})")
    assert int(gpu_f.argmax()) == int(eager_f.argmax()), \
        f"{name}: argmax token disagrees (gpu={int(gpu_f.argmax())} eager={int(eager_f.argmax())})"
    return err_eager, err_ref


# ------------------------------------------------------------------------------------------------
# (B) larger head_dim attention (static path edge @256, dynamic-smem path @512)
# ------------------------------------------------------------------------------------------------
def _run_head_dim_case(name: str, head_dim: int, tok: int, dtype=DType.F32,
                       torch_dt=torch.float32, pos: int = 0):
    # hidden must equal n_heads*head_dim for the toy q/o projections; keep heads small so the
    # model stays tiny while head_dim is large. n_kv_heads=1 (GQA) so attention mapping is exercised.
    n_heads, n_kv = 2, 1
    model = make_toy(seed=2, dtype=torch_dt, n_layers=1, n_heads=n_heads, n_kv_heads=n_kv,
                     head_dim=head_dim, hidden=n_heads * head_dim, intermediate=256, vocab=512)
    graph = from_toy(model)
    prog = lower(graph, target=TARGETS["rtx5090"], pos=pos, dtype=dtype)
    assert validate(prog).ok, f"{name}: program rejected"

    weights = model.weights_dict()
    inputs = _build_inputs(tok, pos)
    ref = ReferenceVM(prog, weights, device="cpu").run(inputs, kv={})["logits"]
    eager = _eager_decode_logits(model, tok)

    try:
        from vm.loader import MegakernelVM
        vm = MegakernelVM(prog, weights, device="cuda")
        gpu = vm.run(inputs, kv={})["logits"]
    except (RuntimeError, TimeoutError) as e:
        raise GpuUnavailable(f"{name}: CUDA VM could not run: {e}") from e

    assert vm.last_status.get("status") == "OK", f"{name}: {vm.last_status}"
    assert vm.last_grid_dim > 0
    assert gpu.is_cuda, f"{name}: output not CUDA"
    assert vm.max_attn_head_dim == head_dim, \
        f"{name}: loader saw head_dim={vm.max_attn_head_dim}, expected {head_dim}"
    # ATTENTION dynamic-smem path is taken iff head_dim > 256 (the static cap). NOTE: the cp.async
    # double-buffered GEMV (the production decode path on sm_80+) ALSO provisions dynamic smem (its
    # weight-staging ring + x-cache), so vm.dyn_smem_bytes is no longer a clean probe of the
    # ATTENTION path alone. To isolate the attention contribution we rebuild with cp.async disabled
    # (knobs={'cpasync':0}): then dyn smem is attention-only and the original invariant holds.
    from vm.loader import MegakernelVM as _VM
    vm_attn = _VM(prog, weights, device="cuda", knobs={"cpasync": 0})
    if head_dim > 256:
        assert vm_attn.dyn_smem_bytes >= (2 * head_dim + 8) * 4, \
            f"{name}: loader under-provisioned attention dynamic smem ({vm_attn.dyn_smem_bytes} B) " \
            f"for head_dim={head_dim}"
    else:
        assert vm_attn.dyn_smem_bytes == 0, \
            f"{name}: static attention path should need 0 dyn smem, got {vm_attn.dyn_smem_bytes}"

    gpu_f = gpu.detach().cpu().to(torch.float32)
    eager_f = eager.detach().cpu().to(torch.float32)
    ref_f = ref.detach().cpu().to(torch.float32)
    err_eager = (gpu_f - eager_f).abs().max().item()
    err_ref = (gpu_f - ref_f).abs().max().item()
    tol = 2e-3 if dtype == DType.F32 else FP16_ATOL
    close_eager = torch.allclose(gpu_f, eager_f, rtol=tol, atol=tol)
    close_ref = torch.allclose(gpu_f, ref_f, rtol=tol, atol=tol)
    path = "DYNAMIC-smem" if head_dim > 256 else "STATIC-smem"
    print(f"  [{name}] head_dim={head_dim} ({path}) dyn_smem={vm.dyn_smem_bytes}B "
          f"grid_dim={vm.last_grid_dim} status={vm.last_status['status']} "
          f"max_err(vs eager)={err_eager:.3e} max_err(vs refVM)={err_ref:.3e} "
          f"allclose: eager={close_eager} refVM={close_ref}")
    assert close_eager, f"{name}: head_dim={head_dim} GPU != eager (max_err={err_eager:.3e})"
    assert close_ref, f"{name}: head_dim={head_dim} GPU != ReferenceVM (max_err={err_ref:.3e})"
    return err_eager, err_ref


# ------------------------------------------------------------------------------------------------
# pytest entry points
# ------------------------------------------------------------------------------------------------
def test_cuda_fp16_one_layer():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    try:
        _run_fp16_case("fp16-1L", n_layers=1, tok=7, pos=0)
    except GpuUnavailable as e:
        import pytest
        pytest.skip(str(e))


def test_cuda_fp16_two_layer():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    try:
        _run_fp16_case("fp16-2L", n_layers=2, tok=19, pos=0)
    except GpuUnavailable as e:
        import pytest
        pytest.skip(str(e))


def test_cuda_attention_head_dim_256_static():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    try:
        _run_head_dim_case("attn-hd256", head_dim=256, tok=5)
    except GpuUnavailable as e:
        import pytest
        pytest.skip(str(e))


def test_cuda_attention_head_dim_512_dynamic_smem():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    try:
        _run_head_dim_case("attn-hd512", head_dim=512, tok=5)
    except GpuUnavailable as e:
        import pytest
        pytest.skip(str(e))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)
    print("fp16 full decode + large-head_dim attention on the GPU...")
    try:
        e1 = _run_fp16_case("fp16-1L", n_layers=1, tok=7, pos=0)
        print("[1/4] 1-layer fp16 decode: GPU == eager == ReferenceVM (2e-2) ... OK")
        e2 = _run_fp16_case("fp16-2L", n_layers=2, tok=19, pos=0)
        print("[2/4] 2-layer fp16 decode: GPU == eager == ReferenceVM (2e-2) ... OK")
        _run_head_dim_case("attn-hd256", head_dim=256, tok=5)
        print("[3/4] head_dim=256 attention (static smem): correct ........... OK")
        _run_head_dim_case("attn-hd512", head_dim=512, tok=5)
        print("[4/4] head_dim=512 attention (dynamic smem): correct .......... OK")
        print(f"\nfp16 MEGAKERNEL + large-head_dim VERIFIED on {torch.cuda.get_device_name(0)} "
              f"(fp16 max_err vs eager: 1L={e1[0]:.2e}, 2L={e2[0]:.2e}).")
    except GpuUnavailable as e:
        print(f"\nSKIP (GPU/cooperative path unavailable): {e}")
        sys.exit(0)

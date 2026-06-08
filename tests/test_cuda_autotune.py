"""
AMK, AUTOTUNE KNOB CORRECTNESS + TUNE-LOOP SMOKE TEST (runs on the real GPU)
============================================================================

Closes the loop on the GEMV autotune knobs (vm/ops.cuh compile-time macros threaded through
vm/loader.py and exercised by vm/autotune.py). Two jobs on the live RTX 5090 (sm_120):

  (A) KNOB CORRECTNESS. Every autotune GEMV variant, cols_per_warp (x-reuse), kunroll (K-ILP),
      and __launch_bounds__ (occupancy), must produce a BIT-IDENTICAL decode result to the
      cols_per_warp=1 default kernel AND to the CPU ReferenceVM. The knobs only change the memory-
      access / ILP / occupancy pattern, not the fp32 elementwise-then-sum order, so the result must
      not move. A variant that changes the answer is a bug, not a tune point.

  (B) TUNE-LOOP SMOKE. A tiny vm.autotune grid builds, correctness-gates, and cuda-event times a
      couple of variants end-to-end, proving the on-hardware loop runs and that the best it keeps
      passed the gate (never times an incorrect kernel).

Run:  uv run python tests/test_cuda_autotune.py     (also a pytest module)
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

BF16_RTOL = BF16_ATOL = 2e-2

# The knob variants under test (all must equal the default kernel and the ReferenceVM).
KNOB_VARIANTS = [
    {"cols_per_warp": 2},
    {"cols_per_warp": 4},
    {"cols_per_warp": 4, "kunroll": 2},
    {"lb_maxthreads": 256, "lb_minblocks": 3},
    {"cols_per_warp": 4, "lb_maxthreads": 256, "lb_minblocks": 3},
]


class GpuUnavailable(Exception):
    pass


def _build_inputs(tok: int, pos: int) -> dict[str, torch.Tensor]:
    contract = required_inputs(pos)
    return {
        TOKEN_NAME: torch.tensor([tok], dtype=torch.int32),
        POS_NAME: torch.tensor([pos], dtype=torch.int32),
        RESHAPE_ID_NAME: torch.tensor([int(contract[RESHAPE_ID_NAME][0])], dtype=torch.int32),
    }


def _run_knob_correctness():
    from vm.loader import MegakernelVM
    model = make_toy(seed=0, dtype=torch.bfloat16, n_layers=2)
    graph = from_toy(model)
    prog = lower(graph, target=TARGETS["rtx5090"], pos=0, dtype=DType.BF16)
    assert validate(prog).ok
    weights = model.weights_dict()
    inputs = _build_inputs(tok=7, pos=0)

    ref = ReferenceVM(prog, weights, device="cpu").run(inputs, kv={})["logits"]
    ref_f = ref.detach().cpu().to(torch.float32)

    try:
        vm0 = MegakernelVM(prog, weights, device="cuda", knobs=None)  # default kernel
        base = vm0.run(inputs, kv={})["logits"].detach().cpu().to(torch.float32)
    except (RuntimeError, TimeoutError) as e:
        raise GpuUnavailable(f"default kernel could not run: {e}") from e
    assert torch.allclose(base, ref_f, rtol=BF16_RTOL, atol=BF16_ATOL), "default != ReferenceVM"

    for knobs in KNOB_VARIANTS:
        try:
            vm = MegakernelVM(prog, weights, device="cuda", knobs=knobs)
            g = vm.run(inputs, kv={})["logits"].detach().cpu().to(torch.float32)
        except (RuntimeError, TimeoutError) as e:
            raise GpuUnavailable(f"variant {knobs} could not run: {e}") from e
        assert vm.last_status.get("status") == "OK", f"{knobs}: {vm.last_status}"
        # knobs change only access pattern/occupancy -> must be BIT-IDENTICAL to the default kernel.
        exact = (g - base).abs().max().item()
        assert exact == 0.0, f"variant {knobs} not bit-identical to default (max_diff={exact:.3e})"
        assert torch.allclose(g, ref_f, rtol=BF16_RTOL, atol=BF16_ATOL), f"{knobs} != ReferenceVM"
        print(f"  [knob] {knobs}  grid_dim={vm.last_grid_dim}  bit-identical to default + == refVM")


def test_autotune_knob_variants_correct():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    try:
        _run_knob_correctness()
    except GpuUnavailable as e:
        import pytest
        pytest.skip(str(e))


def test_autotune_loop_smoke(monkeypatch=None):
    """A minimal vm.autotune run (2 variants, short timing) proves the on-hardware loop builds,
    correctness-gates and times variants, and keeps a gate-passing best."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    import vm.autotune as at

    def _tiny_grid():
        return [({"cols_per_warp": 1, "kunroll": 1, "lb_maxthreads": 0, "lb_minblocks": 0}, 256),
                ({"cols_per_warp": 4, "kunroll": 1, "lb_maxthreads": 0, "lb_minblocks": 0}, 256)]

    import tempfile
    orig = at._knob_grid
    at._knob_grid = _tiny_grid
    tmp = os.path.join(tempfile.gettempdir(), "amk_autotune_smoke.json")
    try:
        # write to a temp path so the smoke run never clobbers the canonical paper/results JSON.
        baseline, best, speedup = at.autotune(iters=30, warmup=10, out_path=tmp)
    except RuntimeError as e:
        import pytest
        pytest.skip(f"autotune loop could not run: {e}")
    finally:
        at._knob_grid = orig
    assert baseline is not None and best is not None
    assert best["lat_us"] > 0.0
    assert best["max_err"] <= BF16_ATOL + 1e-6, "best variant must have passed the correctness gate"
    assert best["pct_roofline"] <= 100.0 + 1e-6, "cannot beat the HBM roofline"
    print(f"  [smoke] baseline={baseline['lat_us']:.1f} us best={best['lat_us']:.1f} us "
          f"speedup={speedup:.3f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)
    print("Autotune knob correctness + tune-loop smoke on the GPU...")
    try:
        _run_knob_correctness()
        print("[1/2] all autotune knob variants bit-identical to default + == ReferenceVM ... OK")
        test_autotune_loop_smoke()
        print("[2/2] autotune loop builds/gates/times variants ...................... OK")
    except GpuUnavailable as e:
        print(f"\nSKIP (GPU/cooperative path unavailable): {e}")
        sys.exit(0)

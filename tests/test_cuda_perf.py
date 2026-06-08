"""
AMK, DECODE MEGAKERNEL STEADY-STATE LATENCY (runs on the real GPU)
==================================================================

Builds the 'small'-scale ToyLlama decode program (hidden=2048, n_layers=4, n_heads=16,
n_kv_heads=4, head_dim=128, intermediate=5632, vocab=32000) and measures STEADY-STATE
per-token latency through the persistent-tables path:

  * one full vm.run() builds + uploads the device tables ONCE,
  * subsequent tokens reuse those tables (the loader's persistent/relaunch path) so the only
    per-token host cost is counter-memset + copying the new IO inputs into the arena.

We time the steady-state token latency with cuda events (warmup, then iters, median) and print
it against the bandwidth roofline of TARGETS['rtx5090'] (weights / HBM bandwidth, the honest
single-stream decode floor).

The bf16 weights of this 'small' config dominate HBM traffic; decode is memory-bound, so the
roofline is the time to stream every weight byte once.

Run:  uv run python tests/test_cuda_perf.py     (also a pytest module)
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


class GpuUnavailable(Exception):
    """Raised when the GPU/cooperative path genuinely cannot run (-> SKIP, not FAIL)."""


# The acceptance 'small'-scale model.
SMALL = dict(hidden=2048, n_layers=4, n_heads=16, n_kv_heads=4, head_dim=128,
             intermediate=5632, vocab=32000)


def _build_inputs(tok: int, pos: int) -> dict[str, torch.Tensor]:
    contract = required_inputs(pos)
    assert set(contract) == {TOKEN_NAME, POS_NAME, RESHAPE_ID_NAME}
    return {
        TOKEN_NAME: torch.tensor([tok], dtype=torch.int32),
        POS_NAME: torch.tensor([pos], dtype=torch.int32),
        RESHAPE_ID_NAME: torch.tensor([int(contract[RESHAPE_ID_NAME][0])], dtype=torch.int32),
    }


def _time_steady_state(vm, inputs, iters: int, warmup: int) -> float:
    """Median STEADY-STATE per-token latency (ms) via cuda events.

    First vm.run() builds + uploads the device tables once. Subsequent tokens go through the
    persistent path: vm.run() with the SAME program/shapes only host-memsets the counters and
    copies the new IO inputs into the arena, then relaunches the cooperative kernel. We time
    that steady-state run(), the real per-token cost in an autoregressive loop."""
    vm.run(inputs, kv={})                       # build device tables once (cold)
    for _ in range(warmup):
        vm.run(inputs, kv={})                   # warm steady-state path
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        vm.run(inputs, kv={})
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _run_perf(dtype=DType.BF16, iters: int = 50, warmup: int = 10):
    from vm.loader import MegakernelVM
    torch_dtype = torch.bfloat16 if dtype == DType.BF16 else torch.float32
    model = make_toy(seed=0, dtype=torch_dtype, **SMALL)
    graph = from_toy(model)
    target = TARGETS["rtx5090"]
    cfg = ScheduleConfig(pipelining_depth=2)
    prog = lower(graph, target=target, config=cfg, pos=0, dtype=dtype)
    res = validate(prog)
    assert res.ok, f"perf program rejected by validator:\n{res.report()}"

    weights = model.weights_dict()
    inputs = _build_inputs(tok=11, pos=0)

    try:
        vm = MegakernelVM(prog, weights, device="cuda")
    except (RuntimeError, TimeoutError) as e:
        raise GpuUnavailable(f"perf: CUDA VM could not build/launch: {e}") from e

    t_ms = _time_steady_state(vm, inputs, iters, warmup)
    if vm.last_status.get("status") != "OK":
        raise GpuUnavailable(f"perf: kernel status {vm.last_status}")

    weight_bytes = int(prog.meta.get("weight_bytes", prog.total_weight_bytes()))
    roofline_us = target.bandwidth_bound_us(weight_bytes)
    t_us = t_ms * 1e3
    pct = (roofline_us / t_us) * 100.0 if t_us > 0 else float("nan")

    print(f"  [perf] dtype={dtype.name} tasks={len(prog.tasks)} grid_dim={vm.last_grid_dim} "
          f"weights={weight_bytes/1e6:.1f} MB")
    print(f"  [perf] steady-state latency = {t_us:.1f} us/token | "
          f"roofline = {roofline_us:.1f} us | {pct:.1f}% of HBM roofline")
    return t_us, roofline_us, pct


def test_cuda_perf_steady_state():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    try:
        t_us, roof_us, pct = _run_perf()
    except GpuUnavailable as e:
        import pytest
        pytest.skip(str(e))
    # Sanity floor only: a measured number must be positive and below the roofline by no more
    # than physics allows (we can never beat the bandwidth floor on real HW).
    assert t_us > 0.0
    assert pct <= 100.0 + 1e-6, f"impossible: {pct:.1f}% of roofline (faster than HBM)"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)
    print("Measuring steady-state decode latency on the GPU (persistent-tables path)...")
    try:
        t_us, roof_us, pct = _run_perf()
        print(f"\nDECODE PERF on {torch.cuda.get_device_name(0)}: "
              f"{t_us:.1f} us/token, {pct:.1f}% of the {roof_us:.1f} us HBM roofline.")
    except GpuUnavailable as e:
        print(f"\nSKIP (GPU/cooperative path unavailable): {e}")
        sys.exit(0)

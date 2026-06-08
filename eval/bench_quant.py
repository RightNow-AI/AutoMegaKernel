"""
AMK, int4/int8 vs bf16 DECODE LATENCY + ROOFLINE (runs on the real GPU)
========================================================================

The bandwidth lever, measured honestly. Decode is memory-bound on the weights: a token streams the
whole weight set through HBM once, so ``time >= weight_bytes / HBM_bandwidth``. Quantizing the
weights to int4 cuts those bytes ~4x, dropping the roofline floor ~4x. This script:

  1. Builds the 'small' ToyLlama (4L, hidden 2048, inter 5632, vocab 32000) in bf16.
  2. Quantizes its Linear weights to groupwise int4 (and int8).
  3. Lowers the SAME decode graph three ways (bf16 / int8 / int4) and runs each through the
     persistent cooperative megakernel (vm.loader.MegakernelVM), timing STEADY-STATE per-token
     latency with cuda events (warmup >= 25, iters >= 100, median).
  4. Reports, for each: weight bytes streamed, the HBM roofline floor (weight_bytes / bw), the
     measured ms/token, the % of roofline achieved, and the int4/int8 speedup vs bf16.

Every latency is a real CUDA-event median; every weight-byte count is the actual buffer total the
program streams (packed int4 + fp16 scales for the quantized paths). No fabricated numbers.

Run:  uv run python eval/bench_quant.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from models.toy import make_toy  # noqa: E402
from schedule.graph import from_toy  # noqa: E402
from schedule.ir import BufferKind, DType, ScheduleConfig, TARGETS, validate  # noqa: E402
from schedule.lower import POS_NAME, RESHAPE_ID_NAME, TOKEN_NAME, lower  # noqa: E402
from schedule.quantize import quantize_weights  # noqa: E402

SMALL = dict(hidden=2048, n_layers=4, n_heads=16, n_kv_heads=4, head_dim=128,
             intermediate=5632, vocab=32000)


class GpuUnavailable(Exception):
    pass


def _inputs(tok: int, pos: int) -> dict[str, torch.Tensor]:
    return {
        TOKEN_NAME: torch.tensor([tok], dtype=torch.int32),
        POS_NAME: torch.tensor([pos], dtype=torch.int32),
        RESHAPE_ID_NAME: torch.tensor([0], dtype=torch.int32),
    }


def _time_steady_state(vm, inputs, iters: int, warmup: int) -> float:
    """Median steady-state per-token latency (ms) via cuda events through the persistent path."""
    vm.run(inputs, kv={})
    for _ in range(warmup):
        vm.run(inputs, kv={})
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        vm.run(inputs, kv={})
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def _time_kernel_only(vm, inputs, iters: int, warmup: int) -> float:
    """Median KERNEL-ONLY per-token latency (ms): build tables once, then re-fire just the
    cooperative kernel (no host re-pack / H2D). Isolates the megakernel from per-token host cost."""
    vm.run(inputs, kv={})
    for _ in range(warmup):
        vm.relaunch()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        vm.relaunch()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def _streamed_weight_bytes(prog, weights) -> int:
    """Actual HBM bytes the program streams for its WEIGHT buffers, using the BOUND tensors'
    storage (so int4 packed uint8 + fp16 scales are counted at their true byte cost, not the
    logical-numel*4 the IR would imply for an I4 buffer)."""
    total = 0
    for b in prog.buffers:
        if b.kind != BufferKind.WEIGHT:
            continue
        key = b.source or b.name
        t = weights.get(key)
        if t is not None:
            total += t.numel() * t.element_size()
        else:
            total += b.nbytes
    return total


def _bench_one(name: str, dtype: DType, weights, graph, target, quant, iters, warmup):
    from vm.loader import MegakernelVM
    cfg = ScheduleConfig(pipelining_depth=2)
    prog = lower(graph, target=target, config=cfg, pos=0, dtype=dtype, quant=quant)
    res = validate(prog)
    assert res.ok, f"{name}: program rejected:\n{res.report()}"
    try:
        vm = MegakernelVM(prog, weights, device="cuda")
    except (RuntimeError, TimeoutError) as e:
        raise GpuUnavailable(f"{name}: CUDA VM could not build/launch: {e}") from e
    t_ms = _time_steady_state(vm, _inputs(11, 0), iters, warmup)
    k_ms = _time_kernel_only(vm, _inputs(11, 0), iters, warmup)
    if vm.last_status.get("status") != "OK":
        raise GpuUnavailable(f"{name}: kernel status {vm.last_status}")
    wbytes = _streamed_weight_bytes(prog, weights)
    roof_us = target.bandwidth_bound_us(wbytes)
    t_us = t_ms * 1e3
    k_us = k_ms * 1e3
    pct = (roof_us / k_us) * 100.0 if k_us > 0 else float("nan")
    return {"name": name, "t_us": t_us, "k_us": k_us, "roof_us": roof_us, "pct": pct,
            "wbytes": wbytes, "grid": vm.last_grid_dim, "tasks": len(prog.tasks)}


def run(iters: int = 100, warmup: int = 25, group: int = 128):
    target = TARGETS["rtx5090"]
    model = make_toy(seed=0, dtype=torch.bfloat16, **SMALL)
    graph = from_toy(model)
    wd = model.weights_dict()
    q8, m8 = quantize_weights(wd, group=group, bits=8, graph=graph)
    q4, m4 = quantize_weights(wd, group=group, bits=4, graph=graph)

    rows = []
    rows.append(_bench_one("bf16", DType.BF16, wd, graph, target, None, iters, warmup))
    rows.append(_bench_one("int8", DType.F16, q8, graph, target, m8, iters, warmup))
    rows.append(_bench_one("int4", DType.F16, q4, graph, target, m4, iters, warmup))

    bf16 = rows[0]
    print(f"  model: small 4L hidden={SMALL['hidden']} inter={SMALL['intermediate']} "
          f"vocab={SMALL['vocab']} | group={group} | iters={iters} warmup={warmup}")
    print(f"  {'path':>5} | {'weights MB':>10} | {'roofline us':>11} | {'kernel us':>9} | "
          f"{'% roof':>7} | {'kSpeedup':>8} | {'token us':>9} | {'tSpeedup':>8}")
    for r in rows:
        ksp = bf16["k_us"] / r["k_us"] if r["k_us"] > 0 else float("nan")
        tsp = bf16["t_us"] / r["t_us"] if r["t_us"] > 0 else float("nan")
        print(f"  {r['name']:>5} | {r['wbytes']/1e6:>10.1f} | {r['roof_us']:>11.1f} | "
              f"{r['k_us']:>9.1f} | {r['pct']:>6.1f}% | {ksp:>7.2f}x | "
              f"{r['t_us']:>9.1f} | {tsp:>7.2f}x")
    roof_drop = bf16["roof_us"] / rows[2]["roof_us"] if rows[2]["roof_us"] > 0 else float("nan")
    print(f"  int4 roofline floor dropped {roof_drop:.2f}x vs bf16 "
          f"({bf16['roof_us']:.1f} -> {rows[2]['roof_us']:.1f} us).")
    print(f"  measured kernel-only speedup vs bf16: int8={bf16['k_us']/rows[1]['k_us']:.2f}x  "
          f"int4={bf16['k_us']/rows[2]['k_us']:.2f}x  (HONEST: dequant ALU offsets the byte savings; "
          f"the non-GEMV megakernel work, attention/norms/sync, is unchanged, capping Amdahl).")

    # Commit the run so the paper cites a real artifact (not a printed-only number). Per path:
    # token_us (full decode), kernel_us (kernel-only relaunch), weight bytes/MB streamed, the
    # weights/HBM roofline floor us, % of that floor reached, and the bf16-relative speedups.
    paths = {}
    for r in rows:
        paths[r["name"]] = {
            "token_us": round(r["t_us"], 1),
            "kernel_us": round(r["k_us"], 1),
            "weight_MB": round(r["wbytes"] / 1e6, 1),
            "weight_bytes": int(r["wbytes"]),
            "roofline_floor_us": round(r["roof_us"], 1),
            "pct_roofline": round(r["pct"], 1),
            "tasks": r["tasks"],
            "speedup_vs_bf16_token": round(bf16["t_us"] / r["t_us"], 2) if r["t_us"] > 0 else None,
            "speedup_vs_bf16_kernel": round(bf16["k_us"] / r["k_us"], 2) if r["k_us"] > 0 else None,
        }
    out = {
        "tag": "quant_decode",
        "gpu": torch.cuda.get_device_name(0),
        "cap": f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}",
        "model": {"name": "small", "n_layers": SMALL["n_layers"], "hidden": SMALL["hidden"],
                  "intermediate": SMALL["intermediate"], "vocab": SMALL["vocab"]},
        "iters": iters, "warmup": warmup, "group": group,
        "paths": paths,
        "int4_roofline_floor_drop_vs_bf16": round(roof_drop, 2),
        "method": ("per-token decode latency on the persistent megakernel; token_us = full decode "
                   "(vm.run), kernel_us = kernel-only re-fire (vm.relaunch); roofline floor = "
                   "streamed weight_bytes / HBM spec bandwidth; CUDA-event median, "
                   f"warmup={warmup} iters={iters}; int8/int4 = W8A16/W4A16 weight-only, "
                   "dequant folded into the GEMV; speedups are bf16-relative."),
    }
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, os.pardir, "paper", "results", "quant_decode.json")
    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}")
    return rows


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)
    print(f"int4/int8 vs bf16 decode latency + roofline on {torch.cuda.get_device_name(0)}...")
    try:
        run()
    except GpuUnavailable as e:
        print(f"SKIP (GPU unavailable): {e}")
        sys.exit(0)

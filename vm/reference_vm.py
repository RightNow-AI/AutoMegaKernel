"""
AMK, CPU/GPU REFERENCE MEGAKERNEL VM (Layer 0, reference model)
===============================================================

A bit-exact *reference model* of the megakernel runtime's scheduling semantics, executed with
real numerics in plain PyTorch. It is NOT a mock: it implements the same contract the CUDA VM
must implement -

  * load a schedule only if ``schedule.ir.validate`` accepts it (correctness from STRUCTURE),
  * drive execution purely by **monotonic counters**: a task fires only when every wait
    ``counter >= threshold`` holds; on completion it does ``out_counter += 1``,
  * detect deadlock dynamically: if a full sweep makes no progress, the remaining tasks are
    stuck and we raise (the watchdog the spec asks for, in software).

Why it matters: this gives a **GPU-free proof** that a generated schedule is correct (its
output equals eager PyTorch) and deadlock-free, and it is the conformance oracle the CUDA VM
is checked against. The CUDA VM in vm/scheduler.cu must produce the same result.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from instructions.reference import RefCtx, reference_for
from schedule.ir import (
    Buffer, BufferKind, DType, MegakernelProgram, validate,
)

_TORCH_DTYPE = {
    DType.F32: torch.float32, DType.F16: torch.float16, DType.BF16: torch.bfloat16,
    DType.I32: torch.int32, DType.I8: torch.int8, DType.U8: torch.uint8, DType.BOOL: torch.bool,
}
# fp8 types map to nearest torch fp8 if available, else bf16 for the reference path.
for _name, _dt in (("F8E4M3", DType.F8E4M3), ("F8E5M2", DType.F8E5M2)):
    _TORCH_DTYPE.setdefault(_dt, getattr(torch, "float8_e4m3fn", torch.bfloat16)
                            if _name == "F8E4M3" else getattr(torch, "float8_e5m2", torch.bfloat16))


def torch_dtype(dt: DType) -> torch.dtype:
    return _TORCH_DTYPE.get(dt, torch.float32)


class DeadlockError(RuntimeError):
    """Raised when the schedule makes no forward progress, the software watchdog firing."""


@dataclass
class RunStats:
    n_tasks: int = 0
    sweeps: int = 0
    exec_order: list[int] = field(default_factory=list)
    wall_s: float = 0.0


class ReferenceVM:
    """Execute a :class:`MegakernelProgram` with counter-driven scheduling and real numerics."""

    def __init__(self, program: MegakernelProgram, weights: dict[str, torch.Tensor],
                 device: str = "cpu", strict_validate: bool = True):
        self.prog = program
        self.device = torch.device(device)
        self.weights = weights
        self.ctx = RefCtx()
        if strict_validate:
            res = validate(program)
            if not res.ok:
                # The VM REFUSES to load an invalid schedule. This is the agent-safety mechanism.
                raise ValueError("ReferenceVM refuses to load an invalid schedule:\n" + res.report())

    # ------------------------------------------------------------------
    def _bind(self, b: Buffer, inputs: dict[str, torch.Tensor],
              kv: dict[str, torch.Tensor]) -> torch.Tensor:
        dt = torch_dtype(b.dtype)
        if b.kind in (BufferKind.WEIGHT, BufferKind.CONST):
            key = b.source or b.name
            if key not in self.weights:
                raise KeyError(f"weight/const buffer '{b.name}' (source='{b.source}') not in weights dict")
            return self.weights[key].to(self.device, dt)
        if b.kind == BufferKind.IO_INPUT:
            if b.name not in inputs:
                raise KeyError(f"IO_INPUT buffer '{b.name}' not provided to run()")
            return inputs[b.name].to(self.device)
        if b.kind == BufferKind.KV_CACHE:
            if b.name in kv:
                return kv[b.name].to(self.device, dt)
            return torch.zeros(b.shape, dtype=dt, device=self.device)
        # ACTIVATION / IO_OUTPUT
        return torch.zeros(b.shape, dtype=dt, device=self.device)

    # ------------------------------------------------------------------
    def run(self, inputs: dict[str, torch.Tensor], kv: dict[str, torch.Tensor] | None = None,
            timeout_s: float = 30.0) -> dict[str, torch.Tensor]:
        """Execute the forward pass. Returns {buffer_name: tensor} for all IO_OUTPUT (and any
        KV_CACHE buffers, so callers can thread state across decode steps)."""
        prog = self.prog
        kv = kv or {}
        tensors: dict[int, torch.Tensor] = {b.id: self._bind(b, inputs, kv) for b in prog.buffers}

        cval = {c.id: c.init for c in prog.counters}
        done = [False] * len(prog.tasks)
        stats = RunStats(n_tasks=len(prog.tasks))
        t0 = time.time()

        remaining = len(prog.tasks)
        while remaining > 0:
            progressed = False
            stats.sweeps += 1
            for t in prog.tasks:
                if done[t.id]:
                    continue
                if all(cval.get(w.counter, 0) >= w.threshold for w in t.waits):
                    ins = [tensors[b] for b in t.inputs]
                    outs = [tensors[b] for b in t.outputs]
                    reference_for(t.op)(ins, outs, t.params, self.ctx)
                    cval[t.out_counter] = cval.get(t.out_counter, 0) + 1
                    done[t.id] = True
                    stats.exec_order.append(t.id)
                    remaining -= 1
                    progressed = True
            if not progressed:
                stuck = [t.id for t in prog.tasks if not done[t.id]]
                raise DeadlockError(
                    f"no progress with {len(stuck)} tasks stuck: {stuck[:16]}"
                    + (" ..." if len(stuck) > 16 else "")
                    + ", counters: " + str(cval))
            if time.time() - t0 > timeout_s:
                raise DeadlockError(f"reference VM exceeded {timeout_s}s (likely a livelock)")

        stats.wall_s = time.time() - t0
        self.last_stats = stats

        out: dict[str, torch.Tensor] = {}
        for b in prog.buffers:
            if b.kind in (BufferKind.IO_OUTPUT, BufferKind.KV_CACHE):
                out[b.name] = tensors[b.id]
        return out


def run_program(program: MegakernelProgram, weights: dict[str, torch.Tensor],
                inputs: dict[str, torch.Tensor], device: str = "cpu",
                kv: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """Convenience one-shot: validate, load, run."""
    return ReferenceVM(program, weights, device=device).run(inputs, kv=kv)

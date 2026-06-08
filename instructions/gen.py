"""
AMK Layer-1, INSTRUCTION GENERATION / AUTORESEARCH LOOP
=======================================================

An AutoKernel-style autonomous tuning loop, adapted to EMIT ABI-conformant instructions instead
of free-form kernels. The search space is deliberately the *safe* knobs of an already-correct
micro-kernel, currently the block size (``AMK_THREADS``), which the kernels read as a
compile-time ``-D`` macro (absent => built-in default). New knobs slot into ``SEARCH_SPACE`` as
the kernels grow to honour them. The loop:

    for each proposed variant:
        1. BUILD  the kernel with the variant's nvcc -D flags (instructions/_build),
        2. VERIFY correctness against instructions/reference.py (instructions/verify_inst),
        3. BENCH  latency with CUDA events,
        4. KEEP-OR-REVERT: a variant is accepted only if it is *correct* AND strictly faster
           than the incumbent (correctness gates latency, exactly like AutoKernel);
        5. LOG    a row to a JSONL flywheel file.

This is a real, runnable loop over the starter kernels (it does not need to discover novel
kernels yet). The keep/revert logic and the build/verify/bench interfaces are production-shaped so
the schedule-search layer (Loop 2) can call the same gate.

Default knobs are no-ops (kernels fall back to their built-in defaults when a macro is absent), so
the default build is byte-identical to what verify_inst.py validates.

CLI:
    uv run python instructions/gen.py gemv_tile           # tune one op
    uv run python instructions/gen.py rmsnorm --iters 6
    uv run python instructions/gen.py --all
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from instructions import _build  # noqa: E402
from instructions.verify_inst import CASES, _TOL, bench_cuda  # noqa: E402

FLYWHEEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_flywheel.jsonl")


# ======================================================================================
# Search space: per-op tunable -D macros. Each value list is a candidate set; the loop
# enumerates the cartesian product (capped by --iters). The kernels read AMK_THREADS (block
# size) when defined; absent -> built-in defaults.
# ======================================================================================
SEARCH_SPACE: dict[str, dict[str, list[int]]] = {
    "gemv_tile":      {"AMK_THREADS": [128, 256, 512]},
    "rmsnorm":        {"AMK_THREADS": [128, 256, 512, 1024]},
    "silu_mul":       {"AMK_THREADS": [128, 256, 512]},
    "add":            {"AMK_THREADS": [128, 256, 512]},
    "rope":           {"AMK_THREADS": [32, 64, 128]},
    "attention_tile": {"AMK_THREADS": [32, 64, 128]},
    "embed":          {"AMK_THREADS": [128, 256, 512]},
}


@dataclass
class Variant:
    op: str
    defines: dict[str, int] = field(default_factory=dict)

    @property
    def cflags(self) -> tuple:
        return tuple(f"-D{k}={v}" for k, v in sorted(self.defines.items()))

    def label(self) -> str:
        if not self.defines:
            return "default"
        return ",".join(f"{k}={v}" for k, v in sorted(self.defines.items()))


@dataclass
class Trial:
    op: str
    variant: str
    built: bool
    correct: bool
    max_rel_err: float
    latency_us: float
    kept: bool
    note: str = ""

    def row(self) -> dict:
        return {"ts": time.time(), "op": self.op, "variant": self.variant, "built": self.built,
                "correct": self.correct, "max_rel_err": self.max_rel_err,
                "latency_us": self.latency_us, "kept": self.kept, "note": self.note}


# ======================================================================================
def _build_variant(variant: Variant, verbose: bool):
    """Build the op's kernel with the variant's -D flags. Returns the module (raises on failure).
    We clear the lru_cache key by using a per-variant build dir via extra cflags tuple identity -
    load_kernel memoises on (name, extra_cuda_cflags), so distinct flags rebuild distinctly."""
    return _build.load_kernel(variant.op, verbose=verbose, extra_cuda_cflags=variant.cflags)


def _eval_variant(variant: Variant, dtype: torch.dtype, verbose: bool) -> Trial:
    """Build + correctness + bench for a single variant. Never raises, captures failures."""
    case = CASES[variant.op]
    t = Trial(op=variant.op, variant=variant.label(), built=False, correct=False,
              max_rel_err=float("nan"), latency_us=float("nan"), kept=False)
    try:
        mod = _build_variant(variant, verbose)
        t.built = True
    except Exception as e:
        t.note = f"build failed: {type(e).__name__}: {str(e)[-160:]}"
        return t
    try:
        inp = case.build_inputs(dtype, "cuda")
        out_cuda = case.run_cuda(mod, inp)
        out_ref = case.run_ref(inp)
        torch.cuda.synchronize()
        rtol, atol = _TOL[dtype]
        a, b = out_cuda.float(), out_ref.float()
        t.correct = bool(torch.allclose(a, b, rtol=rtol, atol=atol))
        t.max_rel_err = ((a - b).abs() / b.abs().clamp_min(1e-6)).max().item()
        if t.correct:
            t.latency_us = bench_cuda(lambda: case.run_cuda(mod, inp))
        else:
            t.note = "numerics mismatch"
    except Exception as e:
        t.note = f"run failed: {type(e).__name__}: {str(e)[-160:]}"
    return t


def _proposals(op: str, max_iters: int) -> list[Variant]:
    """The default (incumbent) variant first, then a capped enumeration of the search grid."""
    grid = SEARCH_SPACE.get(op, {})
    variants = [Variant(op, {})]  # incumbent = built-in defaults
    keys = list(grid)
    combos = itertools.product(*(grid[k] for k in keys)) if keys else []
    for vals in combos:
        variants.append(Variant(op, dict(zip(keys, vals))))
    # de-dup the empty default if it also appears in the grid, cap to max_iters
    seen, uniq = set(), []
    for v in variants:
        key = v.label()
        if key not in seen:
            seen.add(key)
            uniq.append(v)
    return uniq[:max_iters]


def tune(op: str, dtype: torch.dtype, max_iters: int, verbose: bool) -> list[Trial]:
    """Run the keep-or-revert loop over `op`. Returns the list of trials; the best kept variant
    is the one with the lowest latency among correct builds (incumbent wins ties)."""
    print(f"\n=== tuning {op} (dtype={str(dtype).replace('torch.','')}, up to {max_iters} variants) ===")
    best_lat = float("inf")
    best_label = None
    trials: list[Trial] = []
    for v in _proposals(op, max_iters):
        tr = _eval_variant(v, dtype, verbose)
        # keep-or-revert: correctness GATES latency; only a strictly faster correct variant wins.
        if tr.built and tr.correct and tr.latency_us < best_lat:
            tr.kept = True
            best_lat = tr.latency_us
            best_label = tr.variant
        trials.append(tr)
        status = ("KEEP" if tr.kept else ("ok  " if tr.correct else
                  ("XBUILD" if not tr.built else "XNUM ")))
        lat = f"{tr.latency_us:8.2f}us" if tr.latency_us == tr.latency_us else "   --   "
        print(f"  [{status}] {tr.variant:24s} {lat}" + (f"  ({tr.note})" if tr.note else ""))
    with open(FLYWHEEL, "a", encoding="utf-8") as f:
        for tr in trials:
            f.write(json.dumps(tr.row()) + "\n")
    if best_label is not None:
        print(f"  -> best: {best_label} @ {best_lat:.2f}us  (logged {len(trials)} trials)")
    else:
        print("  -> no correct variant found (all reverted)")
    return trials


def main() -> int:
    ap = argparse.ArgumentParser(description="AMK instruction generation / tuning loop")
    ap.add_argument("op", nargs="?", default=None, help="op to tune (default: all)")
    ap.add_argument("--all", action="store_true", help="tune every op")
    ap.add_argument("--iters", type=int, default=4, help="max variants to try per op")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: CUDA not available.", file=sys.stderr)
        return 2
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    ops = list(CASES) if (args.all or not args.op) else [args.op]
    for op in ops:
        if op not in CASES:
            print(f"unknown op '{op}'", file=sys.stderr)
            return 2
    print(f"AMK gen loop | device={torch.cuda.get_device_name(0)} | flywheel={FLYWHEEL}")
    for op in ops:
        tune(op, dt, args.iters, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

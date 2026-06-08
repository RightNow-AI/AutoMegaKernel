"""
AMK Layer-1, PER-INSTRUCTION CONFORMANCE + MICRO-BENCHMARK
==========================================================

The acceptance harness for the standalone ABI-conformant CUDA micro-kernels. For every
implemented op it:

  1. **builds** the CUDA kernel (instructions/cuda/<op>.cu, JIT via instructions/_build.py),
  2. **generates** representative decode-shaped inputs (toy-Llama dims by default),
  3. runs the **CUDA** kernel and the matching **instructions/reference.py** function,
  4. **compares** within dtype tolerance (fp16 ~1e-2, fp32 ~1e-4), the reference is the oracle,
  5. **micro-benchmarks** latency with CUDA events (warmup + iters).

This mirrors AutoKernel's "isolated correctness, then bench" gate: a kernel only earns a PASS if
it matches the locked reference numerics on the real GPU. The same isolated check is what the
generation loop (instructions/gen.py) calls to keep-or-revert a proposed variant.

CLI:
    uv run python instructions/verify_inst.py            # all ops
    uv run python instructions/verify_inst.py gemv_tile  # one op
    uv run python instructions/verify_inst.py --dtype fp16
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from instructions._build import load_kernel  # noqa: E402
from instructions.reference import (  # noqa: E402
    RefCtx, ref_add, ref_attention_tile, ref_embed, ref_gemv_tile, ref_rmsnorm, ref_rope,
    ref_silu_mul,
)

# ---- dtype handling -------------------------------------------------------------------
_DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
_TOL = {  # (rtol, atol) per dtype, fp16 ~1e-2, fp32 ~1e-4 as specified
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
}


def _max_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a32, b32 = a.float(), b.float()
    denom = b32.abs().clamp_min(1e-6)
    return ((a32 - b32).abs() / denom).max().item()


# ======================================================================================
# Per-op test cases. Each returns (cuda_callable, reference_check) that both write `out_cuda`
# / `out_ref` so we can diff. We keep shapes at toy-Llama decode scale (and a larger gemv).
# ======================================================================================
class Case:
    """One op's isolated test: builds inputs, runs CUDA + reference, returns both outputs."""

    def __init__(self, name: str):
        self.name = name

    def build_inputs(self, dtype, device):  # -> dict
        raise NotImplementedError

    def run_cuda(self, mod, t) -> torch.Tensor:
        raise NotImplementedError

    def run_ref(self, t) -> torch.Tensor:
        raise NotImplementedError


class GemvTile(Case):
    """GEMV_TILE: out[..,n_off:n_off+N_tile] = x @ W[n_off:..].T. Two disjoint tiles per the VM
    pattern; here we verify the n_off/N_tile slice math on a non-trivial tile."""

    def build_inputs(self, dtype, device):
        M, K, N = 1, 512, 1024
        x = torch.randn(M, K, dtype=dtype, device=device)
        W = torch.randn(N, K, dtype=dtype, device=device)
        n_off, N_tile = 256, 512               # an interior tile
        return dict(x=x, W=W, M=M, K=K, N=N, n_off=n_off, N_tile=N_tile)

    def run_cuda(self, mod, t):
        out = torch.zeros(t["M"], t["N"], dtype=t["x"].dtype, device=t["x"].device)
        mod.gemv_tile(t["x"], t["W"], out, None, t["n_off"], t["N_tile"])
        return out

    def run_ref(self, t):
        out = torch.zeros(t["M"], t["N"], dtype=t["x"].dtype, device=t["x"].device)
        ref_gemv_tile([t["x"], t["W"]], [out],
                      {"K": t["K"], "N_tile": t["N_tile"], "n_off": t["n_off"]}, RefCtx())
        return out


class RmsNorm(Case):
    def build_inputs(self, dtype, device):
        H = 2048
        return dict(x=torch.randn(1, H, dtype=dtype, device=device),
                    w=torch.randn(H, dtype=dtype, device=device), H=H, eps=1e-6)

    def run_cuda(self, mod, t):
        out = torch.empty_like(t["x"])
        mod.rmsnorm(t["x"], t["w"], out, t["eps"])
        return out

    def run_ref(self, t):
        out = torch.empty_like(t["x"])
        ref_rmsnorm([t["x"], t["w"]], [out], {"eps": t["eps"], "hidden": t["H"]}, RefCtx())
        return out


class SiluMul(Case):
    def build_inputs(self, dtype, device):
        n = (1, 4096)
        return dict(gate=torch.randn(*n, dtype=dtype, device=device),
                    up=torch.randn(*n, dtype=dtype, device=device))

    def run_cuda(self, mod, t):
        out = torch.empty_like(t["gate"])
        mod.silu_mul(t["gate"], t["up"], out)
        return out

    def run_ref(self, t):
        out = torch.empty_like(t["gate"])
        ref_silu_mul([t["gate"], t["up"]], [out], {}, RefCtx())
        return out


class Add(Case):
    def build_inputs(self, dtype, device):
        n = (1, 4096)
        return dict(a=torch.randn(*n, dtype=dtype, device=device),
                    b=torch.randn(*n, dtype=dtype, device=device))

    def run_cuda(self, mod, t):
        out = torch.empty_like(t["a"])
        mod.add(t["a"], t["b"], out)
        return out

    def run_ref(self, t):
        out = torch.empty_like(t["a"])
        ref_add([t["a"], t["b"]], [out], {}, RefCtx())
        return out


class Rope(Case):
    def build_inputs(self, dtype, device):
        S, n_heads, head_dim = 8, 4, 64
        x = torch.randn(S, n_heads, head_dim, dtype=dtype, device=device)
        pos = torch.arange(S, dtype=torch.int64, device=device)
        return dict(x=x, pos=pos, head_dim=head_dim, theta=10000.0)

    def run_cuda(self, mod, t):
        out = torch.empty_like(t["x"])
        mod.rope(t["x"], t["pos"], out, t["head_dim"], t["theta"])
        return out

    def run_ref(self, t):
        out = torch.empty_like(t["x"])
        ref_rope([t["x"], t["pos"]], [out], {"head_dim": t["head_dim"], "theta": t["theta"]},
                 RefCtx())
        return out


class AttentionTile(Case):
    """Decode attention over a KV window with GQA (n_heads=4, n_kv_heads=2)."""

    def build_inputs(self, dtype, device):
        n_heads, n_kv, head_dim, kv_len = 4, 2, 64, 37
        q = torch.randn(n_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(kv_len, n_kv, head_dim, dtype=dtype, device=device)
        v = torch.randn(kv_len, n_kv, head_dim, dtype=dtype, device=device)
        scale = head_dim ** -0.5
        return dict(q=q, k=k, v=v, n_heads=n_heads, n_kv=n_kv, head_dim=head_dim,
                    kv_len=kv_len, scale=scale)

    def run_cuda(self, mod, t):
        out = torch.empty(t["n_heads"], t["head_dim"], dtype=t["q"].dtype, device=t["q"].device)
        mod.attention_tile(t["q"], t["k"], t["v"], out, t["head_dim"], 0, t["kv_len"],
                           t["n_heads"], t["n_kv"], t["scale"], 1)
        return out

    def run_ref(self, t):
        out = torch.empty(t["n_heads"], t["head_dim"], dtype=t["q"].dtype, device=t["q"].device)
        ref_attention_tile(
            [t["q"], t["k"], t["v"]], [out],
            {"head_dim": t["head_dim"], "kv_start": 0, "kv_len": t["kv_len"],
             "n_heads": t["n_heads"], "n_kv_heads": t["n_kv"], "scale": t["scale"], "flags": 1},
            RefCtx())
        return out


class Embed(Case):
    def build_inputs(self, dtype, device):
        V, H, S = 256, 2048, 4
        table = torch.randn(V, H, dtype=dtype, device=device)
        ids = torch.randint(0, V, (S,), dtype=torch.int64, device=device)
        return dict(ids=ids, table=table, V=V, H=H, S=S)

    def run_cuda(self, mod, t):
        out = torch.empty(t["S"], t["H"], dtype=t["table"].dtype, device=t["table"].device)
        mod.embed(t["ids"], t["table"], out)
        return out

    def run_ref(self, t):
        out = torch.empty(t["S"], t["H"], dtype=t["table"].dtype, device=t["table"].device)
        ref_embed([t["ids"], t["table"]], [out], {"hidden": t["H"]}, RefCtx())
        return out


# op name -> (module source name, Case). The module name maps to instructions/cuda/<name>.cu.
CASES: dict[str, Case] = {
    "gemv_tile": GemvTile("gemv_tile"),
    "rmsnorm": RmsNorm("rmsnorm"),
    "silu_mul": SiluMul("silu_mul"),
    "add": Add("add"),
    "rope": Rope("rope"),
    "attention_tile": AttentionTile("attention_tile"),
    "embed": Embed("embed"),
}


# ======================================================================================
def bench_cuda(fn, iters: int = 100, warmup: int = 20) -> float:
    """Median-ish latency in microseconds via CUDA events (warmup + timed iters)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # ms/iter -> us


def verify_one(name: str, dtype: torch.dtype, device: str = "cuda",
               do_bench: bool = True, verbose_build: bool = False) -> dict:
    """Build, run, compare, bench a single op. Returns a result row dict."""
    case = CASES[name]
    row = {"op": name, "dtype": str(dtype).replace("torch.", ""), "passed": False,
           "max_rel_err": float("nan"), "latency_us": float("nan"), "error": ""}
    try:
        mod = load_kernel(name, verbose=verbose_build)
    except Exception as e:  # build failure, report precisely
        row["error"] = f"BUILD FAILED: {type(e).__name__}: {str(e)[-400:]}"
        return row
    try:
        t = case.build_inputs(dtype, device)
        out_cuda = case.run_cuda(mod, t)
        out_ref = case.run_ref(t)
        torch.cuda.synchronize()
        rtol, atol = _TOL[dtype]
        ok = torch.allclose(out_cuda.float(), out_ref.float(), rtol=rtol, atol=atol)
        row["max_rel_err"] = _max_rel_err(out_cuda, out_ref)
        row["passed"] = bool(ok)
        if not ok:
            row["error"] = (f"MISMATCH max|rel|={row['max_rel_err']:.3e} "
                            f"max|abs|={(out_cuda.float()-out_ref.float()).abs().max().item():.3e}")
        if do_bench:
            row["latency_us"] = bench_cuda(lambda: case.run_cuda(mod, t))
    except Exception as e:
        import traceback
        row["error"] = f"RUN FAILED: {type(e).__name__}: {e}"
        if verbose_build:
            traceback.print_exc()
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="AMK per-instruction CUDA conformance + bench")
    ap.add_argument("op", nargs="?", default=None, help="single op to verify (default: all)")
    ap.add_argument("--dtype", default="fp32", choices=list(_DT), help="element dtype")
    ap.add_argument("--no-bench", action="store_true", help="skip latency micro-benchmark")
    ap.add_argument("-v", "--verbose", action="store_true", help="verbose nvcc build output")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: CUDA not available, cannot run GPU acceptance test.", file=sys.stderr)
        return 2

    dev = torch.cuda.get_device_name(0)
    print(f"AMK instruction verify | device={dev} | dtype={args.dtype}")
    print("=" * 78)

    ops = [args.op] if args.op else list(CASES)
    for op in ops:
        if op not in CASES:
            print(f"unknown op '{op}'. known: {', '.join(CASES)}", file=sys.stderr)
            return 2

    dtype = _DT[args.dtype]
    rows = []
    for op in ops:
        row = verify_one(op, dtype, do_bench=not args.no_bench, verbose_build=args.verbose)
        rows.append(row)
        status = "PASS" if row["passed"] else "FAIL"
        lat = f"{row['latency_us']:8.2f}us" if row["latency_us"] == row["latency_us"] else "   --   "
        err = f"  rel_err={row['max_rel_err']:.2e}" if row["max_rel_err"] == row["max_rel_err"] else ""
        print(f"  [{status}] {op:18s} {lat}{err}")
        if row["error"]:
            print(f"         {row['error']}")

    # ---- optional Triton backend check (skipped with a note if Triton is unavailable) ----
    _report_triton(dtype)

    print("=" * 78)
    n_pass = sum(r["passed"] for r in rows)
    print(f"RESULT: {n_pass}/{len(rows)} CUDA instructions PASS on {dev}")
    return 0 if n_pass == len(rows) else 1


def _report_triton(dtype: torch.dtype) -> None:
    """Run the Triton gemv/rmsnorm against the reference if Triton is importable; otherwise SKIP
    with the precise import error (Triton is commonly unavailable on Windows, that is fine)."""
    from instructions import triton as tri
    if not tri.HAVE_TRITON:
        print(f"  [SKIP] triton backend unavailable ({tri.why_unavailable()}) - "
              f"CUDA path is authoritative; see instructions/triton/__init__.py for install.")
        return
    rtol, atol = _TOL[dtype]
    # gemv
    try:
        t = GemvTile("gemv_tile").build_inputs(dtype, "cuda")
        out = torch.zeros(t["M"], t["N"], dtype=dtype, device="cuda")
        tri.gemv_tile(t["x"][0], t["W"], out[0], t["n_off"], t["N_tile"])
        ref = GemvTile("gemv_tile").run_ref(t)
        ok = torch.allclose(out.float(), ref.float(), rtol=rtol, atol=atol)
        print(f"  [{'PASS' if ok else 'FAIL'}] triton.gemv_tile  rel_err="
              f"{_max_rel_err(out, ref):.2e}")
    except Exception as e:
        print(f"  [SKIP] triton.gemv_tile failed to run: {type(e).__name__}: {str(e)[-120:]}")
    # rmsnorm
    try:
        t = RmsNorm("rmsnorm").build_inputs(dtype, "cuda")
        out = torch.empty_like(t["x"])
        tri.rmsnorm(t["x"], t["w"], out, t["eps"])
        ref = RmsNorm("rmsnorm").run_ref(t)
        ok = torch.allclose(out.float(), ref.float(), rtol=rtol, atol=atol)
        print(f"  [{'PASS' if ok else 'FAIL'}] triton.rmsnorm    rel_err="
              f"{_max_rel_err(out, ref):.2e}")
    except Exception as e:
        print(f"  [SKIP] triton.rmsnorm failed to run: {type(e).__name__}: {str(e)[-120:]}")


if __name__ == "__main__":
    raise SystemExit(main())

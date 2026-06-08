"""
AMK Layer-1, Triton backends (optional, behind the same op interface).
=======================================================================

Triton is the fast-iteration backend for the core ops; it sits behind the SAME numeric contract
as the CUDA kernels (instructions/reference.py is the oracle for both). On Windows the upstream
`triton` wheel is usually unavailable, so everything here is import-guarded: if Triton can't be
imported, ``HAVE_TRITON`` is False and the verifier skips the Triton path with a note rather than
failing. The CUDA path is the one that MUST pass on this machine.

INSTALL (to enable the Triton path):
  * Linux / WSL:  ``uv pip install triton``  (or the project extra: ``uv sync --extra triton``)
  * Windows:      ``uv pip install triton-windows``  (community fork; see pyproject extra note).
    Native CUDA-on-Windows Triton support is partial; if it imports but fails to compile, the
    verifier reports SKIP(triton) and the CUDA result stands.

Provided kernels: ``gemv_tile`` and ``rmsnorm`` (the two the task requires), each matching the
reference numerics (Linear [N,K] layout, tile by output columns; RMSNorm fp32-accumulate).
"""
from __future__ import annotations

HAVE_TRITON = False
_IMPORT_ERROR = ""

try:  # pragma: no cover - platform dependent
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
    import torch  # noqa: F401  (part of the import-guard probe: torch must be present too)
    HAVE_TRITON = True
except Exception as e:  # ImportError on Windows w/o triton-windows, or compile-stack missing
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"


def why_unavailable() -> str:
    return _IMPORT_ERROR or "triton present"


if HAVE_TRITON:

    @triton.jit
    def _gemv_tile_kernel(x_ptr, w_ptr, out_ptr, K, Nfull, N_tile, n_off,
                          BLOCK_K: tl.constexpr):
        """One program per output column j in [0, N_tile): out[n_off+j] = sum_k x[k]*W[n_off+j,k].
        M==1 decode gemv. fp32 accumulate (matches reference _gemv_gemm)."""
        j = tl.program_id(0)
        n = n_off + j
        acc = tl.zeros((), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + n * K + offs, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(xv * wv, axis=0)
        tl.store(out_ptr + n, acc.to(out_ptr.dtype.element_ty))

    def gemv_tile(x, W, out, n_off: int, N_tile: int):
        """out[0, n_off:n_off+N_tile] = x @ W[n_off:n_off+N_tile, :].T  (M==1)."""
        assert x.is_cuda and W.is_cuda and out.is_cuda
        K = x.shape[-1]
        Nfull = out.shape[-1]
        BLOCK_K = 256
        _gemv_tile_kernel[(N_tile,)](x.contiguous(), W.contiguous(), out, K, Nfull, N_tile, n_off,
                                     BLOCK_K=BLOCK_K)
        return out

    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, H, eps, BLOCK: tl.constexpr):
        """One program per row. out = x * rsqrt(mean(x^2)+eps) * w, fp32 accumulate."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < H
        x = tl.load(x_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
        ssum = tl.sum(x * x, axis=0)
        rms = 1.0 / tl.sqrt(ssum / H + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * rms * w
        tl.store(out_ptr + row * H + offs, y.to(out_ptr.dtype.element_ty), mask=mask)

    def rmsnorm(x, w, out, eps: float):
        assert x.is_cuda and w.is_cuda and out.is_cuda
        H = x.shape[-1]
        rows = x.numel() // H
        BLOCK = triton.next_power_of_2(H)
        _rmsnorm_kernel[(rows,)](x.contiguous(), w.contiguous(), out, H, eps, BLOCK=BLOCK)
        return out

else:  # graceful stubs so callers can import unconditionally

    def gemv_tile(*a, **k):  # noqa: D401
        raise RuntimeError(f"Triton unavailable ({why_unavailable()})")

    def rmsnorm(*a, **k):
        raise RuntimeError(f"Triton unavailable ({why_unavailable()})")

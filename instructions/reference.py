"""
AMK, REFERENCE INSTRUCTION SEMANTICS (the per-op ground truth)
==============================================================

This is the single source of truth for *what each opcode computes*. Three consumers depend
on it being correct:

  * ``vm/reference_vm.py`` drives these to execute a whole megakernel on CPU/GPU with the exact
    counter-sync semantics of the CUDA VM, giving a GPU-free correctness proof of a schedule.
  * ``instructions/verify_inst.py`` checks each generated Triton/CUDA micro-kernel against the
    matching function here, in isolation, before it is allowed into a megakernel.
  * ``eval/oracle.py`` ultimately compares the whole-model megakernel output to eager PyTorch;
    if the lowering is correct, the reference VM output equals eager within tolerance.

CONVENTIONS (frozen, the Triton/CUDA backends MUST match these exactly):
  * Each function has signature ``op(inputs, outputs, params, ctx)`` and writes results *into*
    the pre-allocated ``outputs`` tensors (mirrors the ABI's output_page_ptrs). It returns None.
  * Linear/weight layout follows torch ``nn.Linear``: weight is ``[N_out, K_in]`` and a GEMV/GEMM
    computes ``x @ W.T``. A tile writes the slice ``out[..., n_off : n_off+N_tile]``.
  * Reductions/matmuls accumulate in fp32 then cast to the output dtype (matches tensor-core
    fp32-accumulate behavior of real kernels and eager torch).
  * Tiling is by output column range ``(n_off, N_tile)``; disjoint tiles of one buffer are
    written by sibling tasks sharing one counter (threshold = number of tiles).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F

from schedule.ir import InstructionKind


@dataclass
class RefCtx:
    """Side state the reference ops may consult (rope tables, etc.). Kept tiny on purpose."""

    extras: dict[str, Any] = field(default_factory=dict)


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float32)


# --------------------------------------------------------------------------------------
# Elementwise / movement
# --------------------------------------------------------------------------------------
def ref_copy(inputs, outputs, params, ctx):
    outputs[0].copy_(inputs[0].to(outputs[0].dtype))


def ref_add(inputs, outputs, params, ctx):
    outputs[0].copy_((_f32(inputs[0]) + _f32(inputs[1])).to(outputs[0].dtype))


def ref_mul(inputs, outputs, params, ctx):
    if len(inputs) == 2:
        out = _f32(inputs[0]) * _f32(inputs[1])
    else:
        out = _f32(inputs[0]) * float(params.get("scale", 1.0))
    outputs[0].copy_(out.to(outputs[0].dtype))


def ref_gelu(inputs, outputs, params, ctx):
    outputs[0].copy_(F.gelu(_f32(inputs[0])).to(outputs[0].dtype))


def ref_silu_mul(inputs, outputs, params, ctx):
    """SwiGLU: silu(gate) * up. inputs = [gate, up]."""
    gate, up = _f32(inputs[0]), _f32(inputs[1])
    outputs[0].copy_((F.silu(gate) * up).to(outputs[0].dtype))


def ref_softmax(inputs, outputs, params, ctx):
    dim = int(params.get("dim", -1))
    outputs[0].copy_(torch.softmax(_f32(inputs[0]), dim=dim).to(outputs[0].dtype))


# --------------------------------------------------------------------------------------
# Norms
# --------------------------------------------------------------------------------------
def ref_rmsnorm(inputs, outputs, params, ctx):
    """RMSNorm: x / sqrt(mean(x^2) + eps) * weight. inputs = [x, weight]."""
    x, w = _f32(inputs[0]), _f32(inputs[1])
    eps = float(params.get("eps", 1e-6))
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    outputs[0].copy_((x * rms * w).to(outputs[0].dtype))


def ref_layernorm(inputs, outputs, params, ctx):
    x = _f32(inputs[0])
    w = _f32(inputs[1])
    b = _f32(inputs[2]) if len(inputs) > 2 else None
    eps = float(params.get("eps", 1e-5))
    hidden = int(params.get("hidden", x.shape[-1]))
    outputs[0].copy_(F.layer_norm(x, (hidden,), w, b, eps).to(outputs[0].dtype))


# --------------------------------------------------------------------------------------
# Embedding / sampling
# --------------------------------------------------------------------------------------
def ref_embed(inputs, outputs, params, ctx):
    """inputs = [token_ids(int), embedding_table[V,H]] -> outputs[0] = rows."""
    ids = inputs[0].to(torch.long).view(-1)
    table = inputs[1]
    outputs[0].copy_(table.index_select(0, ids).to(outputs[0].dtype).view_as(outputs[0]))


def ref_sample_argmax(inputs, outputs, params, ctx):
    """Greedy: logits[..., V] -> argmax token id (int) into outputs[0]."""
    outputs[0].copy_(torch.argmax(_f32(inputs[0]), dim=-1).to(outputs[0].dtype))


# --------------------------------------------------------------------------------------
# Matmuls (tiled by output columns)
# --------------------------------------------------------------------------------------
# Quantized-weight dtype codes (mirror schedule.ir.DType / vm/abi.h). A GEMV/GEMM tile is the
# weight-only quantized path iff params['qdtype'] is one of these AND a scales input is present.
_QDTYPE_I4 = 7   # DType.I4, packed two nibbles/byte
_QDTYPE_I8 = 6   # DType.I8


def _dequant_weight_rows(qw: torch.Tensor, scales: torch.Tensor, qdtype: int, group: int,
                         zeros: torch.Tensor | None, K: int) -> torch.Tensor:
    """Dequantize a groupwise weight-only-quantized matrix to fp32 ``[N, K]``.

    Storage convention (frozen, the CUDA kernel matches it bit-for-bit):
      * I8: ``qw`` is int8 ``[N, K]`` (signed, symmetric: real = q * scale).
      * I4: ``qw`` is uint8 ``[N, K//2]``; nibble layout is little-endian within a byte -
        byte ``b`` holds columns ``2b`` (low nibble) and ``2b+1`` (high nibble). Symmetric int4
        stores ``q+8`` as an unsigned nibble in ``[0,15]``; we subtract 8 to recover ``q∈[-8,7]``.
      * scales: fp ``[N, K//group]``; column ``k`` uses group ``k // group``.
      * zeros (asymmetric only): fp ``[N, K//group]`` integer-valued zero-points; real =
        ``(q - zero) * scale``. Symmetric (zeros is None) uses ``real = q * scale``.
    """
    N = qw.shape[0]
    if qdtype == _QDTYPE_I4:
        packed = qw.to(torch.int32)                       # [N, K//2] in [0,255]
        lo = packed & 0xF                                 # even columns
        hi = (packed >> 4) & 0xF                          # odd columns
        q = torch.stack([lo, hi], dim=-1).reshape(N, -1)  # [N, K] interleaved lo,hi
        q = q[:, :K].to(torch.float32) - 8.0              # symmetric int4: recover q∈[-8,7]
    else:                                                 # int8
        q = qw[:, :K].to(torch.float32)                   # already signed
    s = _f32(scales)                                      # [N, n_groups]
    n_groups = s.shape[-1]
    # expand scales to [N, K] by repeating each group `group` times (last group may be short)
    s_full = s.repeat_interleave(group, dim=-1)[:, :K]
    if s_full.shape[-1] < K:                              # ragged tail (K not multiple of group)
        pad = s[:, -1:].expand(N, K - s_full.shape[-1])
        s_full = torch.cat([s_full, pad], dim=-1)
    if zeros is not None:
        z = _f32(zeros).repeat_interleave(group, dim=-1)[:, :K]
        if z.shape[-1] < K:
            z = torch.cat([z, _f32(zeros)[:, -1:].expand(N, K - z.shape[-1])], dim=-1)
        q = q - z
    _ = n_groups
    return q * s_full                                     # [N, K] fp32


def _gemv_gemm(inputs, outputs, params):
    """Shared core: out_tile = x @ W[n_off:n_off+N_tile, :].T  (+ optional bias tile).

    Two modes, selected by params:
      * FP weight (default): inputs = [x, W(, bias)]; W is fp ``[N, K]`` (torch Linear layout).
      * WEIGHT-ONLY QUANTIZED: params['qdtype'] in {I4,I8} and inputs = [x, qW, scales(, zeros)].
        qW is the packed/int8 weight; we dequantize the tiled rows to fp32 then do the same
        ``x @ W_tile.T`` matvec. This is the dequant-fused GEMV's reference numerics (the CUDA
        kernel unpacks+scales in registers and must match this exactly)."""
    x = _f32(inputs[0])                      # [M, K]  (M=1 for decode gemv)
    qdtype = int(params.get("qdtype", 0))
    quantized = qdtype in (_QDTYPE_I4, _QDTYPE_I8) and len(inputs) >= 3
    n_off = int(params.get("n_off", 0))
    K = int(params.get("K", x.shape[-1]))

    if quantized:
        qw = inputs[1]                       # int4(packed uint8)/int8 [N, K(/2)]
        scales = inputs[2]
        zeros = inputs[3] if len(inputs) > 3 and inputs[3] is not None else None
        group = int(params.get("group", K))
        N_full = scales.shape[0]
        n_tile = int(params.get("N_tile", N_full))
        # dequant ONLY the rows this tile needs (rows == output columns n_off..n_off+n_tile)
        qw_tile = qw[n_off:n_off + n_tile]
        s_tile = scales[n_off:n_off + n_tile]
        z_tile = zeros[n_off:n_off + n_tile] if zeros is not None else None
        w_tile = _dequant_weight_rows(qw_tile, s_tile, qdtype, group, z_tile, K)  # [N_tile, K]
        out = x @ w_tile.t()                 # [M, N_tile]
        outputs[0][..., n_off:n_off + n_tile] = out.to(outputs[0].dtype)
        return

    w = _f32(inputs[1])                      # [N, K]  (torch Linear layout)
    n_tile = int(params.get("N_tile", w.shape[0]))
    w_tile = w[n_off:n_off + n_tile, :]      # [N_tile, K]
    out = x @ w_tile.t()                     # [M, N_tile]
    if len(inputs) > 2 and inputs[2] is not None:
        out = out + _f32(inputs[2])[n_off:n_off + n_tile]
    outputs[0][..., n_off:n_off + n_tile] = out.to(outputs[0].dtype)


def ref_gemv_tile(inputs, outputs, params, ctx):
    _gemv_gemm(inputs, outputs, params)


def ref_gemm_tile(inputs, outputs, params, ctx):
    _gemv_gemm(inputs, outputs, params)


def ref_dequant(inputs, outputs, params, ctx):
    """Dequantize a packed int4/int8 weight tile with per-group scales.
    inputs = [q_weight, scales(, zeros)] -> outputs[0] fp tile. Group size in params['group']."""
    q = inputs[0].to(torch.float32)
    scales = _f32(inputs[1])
    group = int(params.get("group", q.shape[-1]))
    zeros = _f32(inputs[2]) if len(inputs) > 2 else None
    # q already unpacked to int values by the caller for the reference path; apply scale/zero.
    qg = q.view(*q.shape[:-1], -1, group)
    s = scales.view(*scales.shape[:-1], -1, 1)
    deq = (qg - (zeros.view(*zeros.shape[:-1], -1, 1) if zeros is not None else 0.0)) * s
    outputs[0].copy_(deq.view_as(outputs[0]).to(outputs[0].dtype))


# --------------------------------------------------------------------------------------
# Positional / attention / KV cache
# --------------------------------------------------------------------------------------
def _rope_tables(seq_pos: torch.Tensor, head_dim: int, theta: float, device, dtype):
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    ang = seq_pos.to(torch.float32)[:, None] * inv_freq[None, :]   # [S, half]
    return torch.cos(ang), torch.sin(ang)


def ref_rope(inputs, outputs, params, ctx):
    """Apply rotary embedding (Llama rotate-half). inputs = [x, positions(int)].
    Supported x shapes (decode-first; prefill batched is a documented TODO):
      * [n_heads, head_dim]          with pos length 1  (single decode token)
      * [S, n_heads, head_dim]       with pos length S  (per-position)
    cos/sin are placed on the sequence axis so they broadcast over heads correctly."""
    x = _f32(inputs[0])
    head_dim = int(params["head_dim"])
    theta = float(params.get("theta", 10000.0))
    pos = inputs[1].to(torch.long).view(-1)
    half = head_dim // 2
    cos, sin = _rope_tables(pos, head_dim, theta, x.device, x.dtype)  # [S, half]

    if x.dim() == 2:                       # [n_heads, head_dim], single position
        c = cos[0].view(1, half)           # broadcast over heads
        s = sin[0].view(1, half)
    elif x.dim() == 3:                     # [S, n_heads, head_dim]
        assert cos.shape[0] == x.shape[0], f"rope pos len {cos.shape[0]} != seq {x.shape[0]}"
        c = cos.view(x.shape[0], 1, half)  # [S,1,half] broadcasts over heads
        s = sin.view(x.shape[0], 1, half)
    else:
        raise NotImplementedError(f"ref_rope supports rank 2 or 3, got {tuple(x.shape)}")

    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    outputs[0].copy_(rot.to(outputs[0].dtype))


def ref_allreduce_shard(inputs, outputs, params, ctx):
    """Single-node reference for a tensor-parallel shard reduce: sum the shard inputs into the
    output (on one device this is the identity of the real cross-GPU all-reduce result)."""
    acc = _f32(inputs[0]).clone()
    for t in inputs[1:]:
        acc += _f32(t)
    outputs[0].copy_(acc.to(outputs[0].dtype))


def ref_kv_append(inputs, outputs, params, ctx):
    """Append new k or v into a KV cache buffer at position `pos`.
    inputs = [new_kv[.., n_kv_heads, head_dim], kv_cache]; outputs[0] = kv_cache (in place)."""
    pos = int(params["pos"])
    new = inputs[0].to(outputs[0].dtype)
    cache = outputs[0]
    # BATCHED DECODE: cache is [B, max_seq, n_kv_heads, head_dim] (one history per sequence) and the
    # new k/v is [B, n_kv_heads, head_dim] (roped k) or [B, kv_dim] (flat v); write every sequence's
    # position `pos` slot at once. Reached only for a batch>1 program (rank-4 cache); the rank-3
    # single-sequence path below is byte-identical.
    if cache.dim() == 4:
        Bsz = cache.shape[0]
        cache[:, pos].copy_(new.reshape(Bsz, *cache.shape[2:]))
        return
    # cache layout: [max_seq, n_kv_heads, head_dim]
    cache[pos:pos + new.shape[0]].copy_(new.view(-1, *cache.shape[1:]))


def ref_attention_tile(inputs, outputs, params, ctx):
    """Single-instruction attention over a KV window (whole-window flash; the CUDA backend may
    tile internally). inputs = [q[.., n_heads, head_dim], k_cache, v_cache] -> outputs[0].

    q is the current step query; k_cache/v_cache hold [kv_len, n_kv_heads, head_dim]. Supports
    grouped-query attention (n_heads % n_kv_heads == 0) and causal masking via flags bit0."""
    q = _f32(inputs[0])                              # [n_heads, head_dim] (decode: one token)
    k = _f32(inputs[1])
    v = _f32(inputs[2])
    head_dim = int(params["head_dim"])
    kv_start = int(params.get("kv_start", 0))
    kv_len = int(params.get("kv_len", k.shape[0]))
    n_heads = int(params.get("n_heads", q.shape[-2]))
    n_kv = int(params.get("n_kv_heads", n_heads))
    scale = float(params.get("scale", 1.0 / (head_dim ** 0.5)))
    causal = bool(int(params.get("flags", 0)) & 1)

    # BATCHED DECODE (throughput path): q is [B, n_heads, head_dim] and the caches are
    # [B, max_seq, n_kv, head_dim] (one independent KV history per sequence). Each of the B queries
    # is a single token attending to its OWN cached window [kv_start, kv_start+kv_len) -- it is B
    # independent decode-attentions done in one task. This branch is reached ONLY when the lowerer
    # emits a batch>1 program (rank-3 q); the rank-2 single-token path below is byte-identical.
    if q.dim() == 3:
        Bsz = q.shape[0]
        rep = n_heads // n_kv
        kk = k[:, kv_start:kv_start + kv_len].repeat_interleave(rep, dim=2)   # [B,kv_len,n_heads,hd]
        vv = v[:, kv_start:kv_start + kv_len].repeat_interleave(rep, dim=2)
        scores = torch.einsum("bhd,bkhd->bhk", q, kk) * scale                # [B,n_heads,kv_len]
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhk,bkhd->bhd", probs, vv)                       # [B,n_heads,head_dim]
        outputs[0].copy_(out.reshape_as(outputs[0]).to(outputs[0].dtype))
        _ = Bsz
        return

    k = k[kv_start:kv_start + kv_len]                # [kv_len, n_kv, head_dim]
    v = v[kv_start:kv_start + kv_len]
    rep = n_heads // n_kv
    k = k.repeat_interleave(rep, dim=1)             # [kv_len, n_heads, head_dim]
    v = v.repeat_interleave(rep, dim=1)
    # q: [n_heads, head_dim] -> scores [n_heads, kv_len]
    scores = torch.einsum("hd,khd->hk", q, k) * scale
    if causal:
        # decode single query attends to all cached keys (all <= current pos): no masking needed
        pass
    if bool(int(params.get("flags", 0)) & 2):       # PARTIAL_WRITE: split-KV shard -> flash partials
        # Write the UN-normalized flash partial for this kv shard: [n_heads, head_dim+2] =
        # [acc (sum exp(s-m)*v) | m (running max) | l (running denom)]. ATTENTION_COMBINE merges
        # the P shards' partials with the online-softmax reduction. Empty shard (kv_len==0) ->
        # acc=0, m=-inf, l=0 so the combine gives it zero weight.
        ob = outputs[0].view(n_heads, head_dim + 2)
        if kv_len <= 0:
            ob[:, :head_dim].zero_(); ob[:, head_dim].fill_(float("-inf")); ob[:, head_dim + 1].zero_()
            return
        m = scores.max(dim=-1).values               # [n_heads]
        e = torch.exp(scores - m[:, None])          # [n_heads, kv_len]
        l = e.sum(dim=-1)                            # [n_heads]
        acc = torch.einsum("hk,khd->hd", e, v)      # [n_heads, head_dim] (un-normalized)
        ob[:, :head_dim].copy_(acc.to(ob.dtype))
        ob[:, head_dim].copy_(m.to(ob.dtype))
        ob[:, head_dim + 1].copy_(l.to(ob.dtype))
        return
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hk,khd->hd", probs, v)      # [n_heads, head_dim]
    outputs[0].copy_(out.view_as(outputs[0]).to(outputs[0].dtype))


def ref_attention_combine(inputs, outputs, params, ctx):
    """Merge P split-KV flash partials into the normalized attention output (flash-decoding combine).
    Each input p is a partial [n_heads, head_dim+2] from ref_attention_tile's PARTIAL_WRITE mode:
    [:, :head_dim]=acc_p, [:, head_dim]=m_p, [:, head_dim+1]=l_p. Online-softmax reduction:
    m = max_p m_p ; l = sum_p l_p*exp(m_p-m) ; out = (sum_p acc_p*exp(m_p-m)) / l. A shard with
    l_p==0 (empty window) contributes nothing (explicit zero-weight, never exp(-inf))."""
    head_dim = int(params["head_dim"])
    parts = [_f32(x) for x in inputs]
    n_heads = int(params.get("n_heads", parts[0].numel() // (head_dim + 2)))
    parts = [p.view(n_heads, head_dim + 2) for p in parts]
    m = torch.stack([p[:, head_dim] for p in parts], dim=0).max(dim=0).values     # [n_heads]
    acc = torch.zeros(n_heads, head_dim, dtype=torch.float32, device=parts[0].device)
    l = torch.zeros(n_heads, dtype=torch.float32, device=parts[0].device)
    for p in parts:
        mp, lp = p[:, head_dim], p[:, head_dim + 1]
        w = torch.where(lp > 0, torch.exp(mp - m), torch.zeros_like(mp))          # empty shard -> 0
        l = l + lp * w
        acc = acc + p[:, :head_dim] * w[:, None]
    out = acc / l[:, None].clamp_min(1e-20)
    outputs[0].copy_(out.view_as(outputs[0]).to(outputs[0].dtype))


# --------------------------------------------------------------------------------------
# Fused instruction: a recipe of primitive ops run over on-chip scratch
# --------------------------------------------------------------------------------------
# Recipe dtype allowlist (recipe out_dtype is data; never getattr an arbitrary torch attribute).
_RECIPE_DTYPES = {
    "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
    "float64": torch.float64, "int8": torch.int8, "int32": torch.int32,
}


def validate_recipe(recipe, n_inputs):
    """Structural validation of a FUSED recipe. Returns a list of error strings ([] if well-formed).
    Run before execution so a malformed recipe is a clear error, never an uninitialized read or an
    opaque IndexError/KeyError deep inside a primitive."""
    if not isinstance(recipe, dict):
        return ["recipe must be a dict"]
    steps = recipe.get("steps")
    if not isinstance(steps, list) or not steps:
        return ["recipe.steps must be a non-empty list"]
    errs, produced, wrote_final = [], set(), False
    for i, st in enumerate(steps):
        if not isinstance(st, dict):
            errs.append(f"step {i} must be a dict")
            continue
        try:
            op = InstructionKind(int(st.get("op")))
        except Exception:
            errs.append(f"step {i}: bad op {st.get('op')!r}")
            op = None
        if op is InstructionKind.FUSED:
            errs.append(f"step {i}: nested FUSED is not allowed")
        for a in (st.get("args") or []):
            if not (isinstance(a, dict) and len(a) == 1):
                errs.append(f"step {i}: bad arg {a!r}")
            elif "in" in a:
                if not (isinstance(a["in"], int) and 0 <= a["in"] < n_inputs):
                    errs.append(f"step {i}: input index {a['in']!r} out of range [0,{n_inputs})")
            elif "s" in a:
                if a["s"] not in produced:
                    errs.append(f"step {i}: reads scratch {a['s']!r} not written by an earlier step")
            else:
                errs.append(f"step {i}: arg must be {{'in':k}} or {{'s':j}}, got {a!r}")
        out = st.get("out")
        if out == "final" or out == -1:
            wrote_final = True
        elif isinstance(out, int):
            produced.add(out)
        else:
            errs.append(f"step {i}: out must be 'final' or an int scratch index, got {out!r}")
        od = st.get("out_dtype")
        if od is not None and od not in _RECIPE_DTYPES:
            errs.append(f"step {i}: out_dtype {od!r} not in the allowlist {sorted(_RECIPE_DTYPES)}")
    if not wrote_final:
        errs.append("recipe never writes its final output (no step with out=='final')")
    return errs


def ref_fused(inputs, outputs, params, ctx):
    """Execute a FUSED instruction: a recipe of primitive op steps run over transient scratch,
    writing the net result into ``outputs[0]``. Intermediates live only in scratch (that IS the
    fusion - they never round-trip a global buffer). The result is bit-identical to the unfused op
    sequence PROVIDED each step FULLY overwrites its scratch output AND the scratch dtype matches the
    unfused intermediate buffer's dtype - so the recipe carries ``out_shape`` + ``out_dtype`` to
    guarantee both. Scratch is zero-initialized, so an (invalid) partial write is deterministic
    rather than reading uninitialized memory; the recipe is structurally validated up front and a
    malformed recipe raises ValueError instead of crashing deep inside a primitive.

    ``params['recipe']`` is ``{"steps": [step, ...]}`` where each step is::

        {"op": <int opcode>,
         "args": [<ref>, ...],          # <ref> = {"in": k} -> inputs[k] | {"s": j} -> scratch[j]
         "out":  <int j> | "final",     # scratch index, or "final" -> outputs[0]
         "params": {...},               # the primitive op's params
         "out_shape": [..], "out_dtype": "<name>"}  # scratch shape/dtype (default: zeros_like(args[0]))
    """
    recipe = params.get("recipe") or {}
    errs = validate_recipe(recipe, len(inputs))
    if errs:
        raise ValueError("malformed FUSED recipe: " + "; ".join(errs[:4]))
    scratch: dict[int, torch.Tensor] = {}

    def resolve(ref):
        return inputs[int(ref["in"])] if "in" in ref else scratch[int(ref["s"])]

    for step in recipe["steps"]:
        op = InstructionKind(int(step["op"]))
        args = [resolve(a) for a in step["args"]]
        out_ref = step["out"]
        if out_ref == "final" or out_ref == -1:
            out_t = outputs[0]
        else:
            shape = step.get("out_shape")
            dtype = _RECIPE_DTYPES[step["out_dtype"]] if step.get("out_dtype") else args[0].dtype
            out_t = (torch.zeros(tuple(shape), dtype=dtype, device=args[0].device)
                     if shape is not None else torch.zeros_like(args[0], dtype=dtype))
            scratch[int(out_ref)] = out_t
        reference_for(op)(args, [out_t], step.get("params", {}), ctx)


# --------------------------------------------------------------------------------------
# The registry (opcode -> reference fn). The VM and verifiers dispatch through this.
# --------------------------------------------------------------------------------------
REFERENCE: dict[InstructionKind, Callable] = {
    InstructionKind.NOP: lambda i, o, p, c: None,
    InstructionKind.COPY: ref_copy,
    InstructionKind.EMBED: ref_embed,
    InstructionKind.RMSNORM: ref_rmsnorm,
    InstructionKind.LAYERNORM: ref_layernorm,
    InstructionKind.GEMV_TILE: ref_gemv_tile,
    InstructionKind.GEMM_TILE: ref_gemm_tile,
    InstructionKind.ATTENTION_TILE: ref_attention_tile,
    InstructionKind.ATTENTION_COMBINE: ref_attention_combine,
    InstructionKind.ROPE: ref_rope,
    InstructionKind.SILU_MUL: ref_silu_mul,
    InstructionKind.GELU: ref_gelu,
    InstructionKind.ADD: ref_add,
    InstructionKind.MUL: ref_mul,
    InstructionKind.DEQUANT: ref_dequant,
    InstructionKind.SOFTMAX: ref_softmax,
    InstructionKind.KV_APPEND: ref_kv_append,
    InstructionKind.SAMPLE_ARGMAX: ref_sample_argmax,
    InstructionKind.ALLREDUCE_SHARD: ref_allreduce_shard,
    InstructionKind.FUSED: ref_fused,
}


def reference_for(op: InstructionKind) -> Callable:
    if op not in REFERENCE:
        raise NotImplementedError(f"no reference semantics for opcode {op.name}")
    return REFERENCE[op]

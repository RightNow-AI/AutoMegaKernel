"""
AMK, WEIGHT-ONLY GROUPWISE QUANTIZATION (the bandwidth lever)
=============================================================

Decode is bandwidth-bound on the weights: every token streams the whole weight set through HBM
once, so ``time >= weight_bytes / HBM_bandwidth``. Quantizing the weights to int4 cuts those bytes
~4x, dropping the roofline floor ~4x. This module quantizes a model's **Linear projection
weights** to GROUPWISE int4 (or int8), symmetric, leaving everything else (norms, embedding table,
biases) in its original dtype. Activations stay fp, this is *weight-only* quantization.

THE PACKING CONVENTION (frozen, instructions/reference.py and vm/ops.cuh match it bit-for-bit):

  A projection weight ``W`` has torch ``nn.Linear`` layout ``[N_out, K_in]``. We quantize along the
  reduction axis ``K`` in contiguous groups of ``group`` (default 128). For row ``n`` and group
  ``gk`` (columns ``[gk*group, (gk+1)*group)``):

      scale[n, gk] = max(|W[n, cols]|) / qmax           (symmetric, per-group, per-row)
      q[n, k]      = round(W[n, k] / scale[n, gk])       clamped to [-qmax-? , qmax]

    * int8 : ``qmax = 127``, q in ``[-127, 127]`` stored as signed int8 ``[N, K]``.
    * int4 : levels in ``[-8, 7]``; we store ``q + 8`` (an unsigned nibble in ``[0,15]``) and PACK
             two nibbles per byte little-endian: byte ``b`` of a row holds column ``2b`` in its low
             nibble and column ``2b+1`` in its high nibble. Storage is ``uint8 [N, K//2]``.

  Dequant (the GEMV uses this): ``real[n,k] = (q[n,k] - zero[n,gk]) * scale[n,gk]``; symmetric sets
  ``zero = 0`` (no zeros tensor). Asymmetric int4/int8 (optional) stores an unsigned ``q`` in
  ``[0, 2^bits-1]`` plus a per-group integer ``zero`` so ``real = (q - zero) * scale``.

OUTPUT of :func:`quantize_weights`:  ``(qweights, meta)`` where ``qweights`` is a flat name->tensor
dict ready to bind to the VM exactly like a normal ``weights_dict()``:
    <key>            packed int4 (uint8 [N,K//2]) or int8 ([N,K])  for each quantized projection
    <key>.scales     fp16 [N, K//group]
    <key>.zeros      fp16 [N, K//group]                            (asymmetric only)
    <every other key>  passed through unchanged (norms, embed table, lm_head if not quantized)
``meta`` records ``group``, ``bits``, ``symmetric``, the set of quantized keys, and per-key shapes -
the lowerer reads it to emit the quantized GEMV tiles and size the scales/zeros buffers.

This is pure post-training round-to-nearest quantization (no calibration/GPTQ). It is deliberately
simple and CORRECT; the point of the module is the end-to-end *bandwidth* path, and the honest
quality story is "typical int4 RTN loss", measured as greedy-token agreement vs fp16, not claimed
to be lossless.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

# Names appended to a quantized weight key for its companion tensors.
SCALES_SUFFIX = ".scales"
ZEROS_SUFFIX = ".zeros"

# Roles in a ModelGraph that are Linear projections we quantize. Embedding/norms are NOT quantized.
DEFAULT_QUANT_ROLES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head",
})


@dataclass
class QuantMeta:
    """Everything the lowerer/loader needs to wire a quantized program."""

    group: int = 128
    bits: int = 4
    symmetric: bool = True
    # quantized weight key -> {"N":, "K":, "n_groups":, "packed_shape":, "scales_shape":}
    keys: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def qdtype_name(self) -> str:
        return "I4" if self.bits == 4 else "I8"

    def is_quantized(self, key: str) -> bool:
        return key in self.keys

    def to_dict(self) -> dict[str, Any]:
        return {"group": self.group, "bits": self.bits, "symmetric": self.symmetric,
                "keys": self.keys}


def _quantize_matrix(W: torch.Tensor, group: int, bits: int, symmetric: bool):
    """Quantize one Linear weight ``[N, K]`` groupwise along K.

    Returns ``(packed, scales, zeros|None)`` where ``packed`` is int8 ``[N,K]`` (bits==8) or
    uint8 ``[N,K//2]`` (bits==4, two nibbles/byte), ``scales`` fp32 ``[N, n_groups]``, and
    ``zeros`` fp32 ``[N, n_groups]`` for asymmetric (else None)."""
    assert W.dim() == 2, f"expected a 2D Linear weight, got {tuple(W.shape)}"
    N, K = W.shape
    Wf = W.detach().to(torch.float32)
    n_groups = (K + group - 1) // group
    Kpad = n_groups * group
    if Kpad != K:                                   # pad K up to a whole number of groups with 0s
        Wf = torch.cat([Wf, torch.zeros(N, Kpad - K, dtype=torch.float32)], dim=1)
    Wg = Wf.view(N, n_groups, group)                # [N, n_groups, group]

    if symmetric:
        qmax = (1 << (bits - 1)) - 1                # 7 (int4) / 127 (int8)
        amax = Wg.abs().amax(dim=-1, keepdim=True)  # [N, n_groups, 1]
        scales = (amax / qmax).clamp_min(1e-8)
        q = torch.round(Wg / scales).clamp(-qmax - (0 if bits == 8 else 1), qmax)
        zeros = None
    else:
        qmax = (1 << bits) - 1                       # 15 (int4) / 255 (int8) unsigned levels
        wmin = Wg.amin(dim=-1, keepdim=True)
        wmax = Wg.amax(dim=-1, keepdim=True)
        scales = ((wmax - wmin) / qmax).clamp_min(1e-8)
        zeros = torch.round(-wmin / scales)          # integer zero-point
        q = torch.round(Wg / scales + zeros).clamp(0, qmax)

    q = q.view(N, Kpad)[:, :K].to(torch.int32)       # back to [N, K], drop pad
    scales = scales.view(N, n_groups).to(torch.float32)
    if zeros is not None:
        zeros = zeros.view(N, n_groups).to(torch.float32)

    if bits == 8:
        if symmetric:
            packed = q.to(torch.int8)                # signed
        else:
            packed = q.to(torch.uint8)               # unsigned (with zero-point)
        return packed, scales, zeros

    # int4: encode levels as unsigned nibbles, then pack two per byte (little-endian within byte).
    if symmetric:
        nib = (q + 8).clamp(0, 15).to(torch.int32)   # [-8,7] -> [0,15]
    else:
        nib = q.clamp(0, 15).to(torch.int32)
    if K % 2 != 0:                                    # pad odd K with a zero nibble
        nib = torch.cat([nib, torch.zeros(N, 1, dtype=torch.int32)], dim=1)
    lo = nib[:, 0::2]                                 # even columns -> low nibble
    hi = nib[:, 1::2]                                 # odd columns  -> high nibble
    packed = (lo | (hi << 4)).to(torch.uint8)        # [N, K//2]
    return packed, scales, zeros


def quantize_weights(weights_dict: dict[str, torch.Tensor], group: int = 128, bits: int = 4,
                     symmetric: bool = True,
                     quant_keys: set[str] | frozenset[str] | None = None,
                     quant_roles: set[str] | frozenset[str] = DEFAULT_QUANT_ROLES,
                     graph: Any = None) -> tuple[dict[str, torch.Tensor], QuantMeta]:
    """Quantize a model's Linear weights to groupwise int4/int8 (weight-only).

    Args:
        weights_dict: flat key->tensor (e.g. ``model.weights_dict()``).
        group:    quantization group size along the reduction axis K (default 128).
        bits:     4 (packed nibbles) or 8.
        symmetric: symmetric (no zero-point) if True; asymmetric (with zeros) if False.
        quant_keys: explicit set of weight keys to quantize. If None, all 2D weights whose key
                    matches a projection role are quantized (see ``graph`` / ``quant_roles``).
        quant_roles: role names treated as quantizable Linear projections (used with ``graph`` or
                    a name heuristic when ``quant_keys`` is None).
        graph:    optional ModelGraph; if given, its weight ``role`` tags select which keys to
                  quantize (the precise path). Otherwise a substring heuristic on the key is used.

    Returns ``(qweights, meta)``: a flat dict ready to bind to the VM, plus the :class:`QuantMeta`
    the lowerer consumes. Non-quantized weights are passed through unchanged."""
    assert bits in (4, 8), "only int4 / int8 supported"
    if quant_keys is None:
        quant_keys = _select_quant_keys(weights_dict, quant_roles, graph)

    out: dict[str, torch.Tensor] = {}
    meta = QuantMeta(group=group, bits=bits, symmetric=symmetric)
    for key, W in weights_dict.items():
        if key in quant_keys and W.dim() == 2:
            packed, scales, zeros = _quantize_matrix(W, group, bits, symmetric)
            N, K = W.shape
            out[key] = packed
            out[key + SCALES_SUFFIX] = scales.to(torch.float16)
            if zeros is not None:
                out[key + ZEROS_SUFFIX] = zeros.to(torch.float16)
            meta.keys[key] = {
                "N": int(N), "K": int(K), "n_groups": int(scales.shape[1]),
                "packed_shape": list(packed.shape), "scales_shape": list(scales.shape),
                "has_zeros": zeros is not None,
            }
        else:
            out[key] = W
    return out, meta


def _select_quant_keys(weights_dict: dict[str, torch.Tensor],
                       quant_roles: set[str] | frozenset[str], graph: Any) -> set[str]:
    """Pick which keys to quantize: prefer a ModelGraph's role tags; else a name heuristic.

    CRITICAL (tied embeddings): a weight key that is ALSO read by a non-GEMV op, e.g. the
    embedding table in a tied-lm_head checkpoint, where ``model.embed_tokens.weight`` is the source
    for BOTH the ``embed`` op (gather, needs fp rows) AND the ``lm_head`` GEMV, must NOT be
    quantized. Packing it to int4 would corrupt the EMBED gather (it would read packed bytes as fp).
    We exclude any key used by a non-``linear`` graph node."""
    keys: set[str] = set()
    excluded: set[str] = set()
    if graph is not None and getattr(graph, "nodes", None) is not None:
        for n in graph.nodes:
            if getattr(n, "op", "") != "linear":
                for wk in getattr(n, "weights", []) or []:
                    excluded.add(wk)   # used by embed/etc. -> never quantize (shared tensor)
    if graph is not None and getattr(graph, "weights", None):
        for key, w in graph.weights.items():
            if getattr(w, "role", "") in quant_roles and key in weights_dict \
                    and weights_dict[key].dim() == 2 and key not in excluded:
                keys.add(key)
        if keys:
            return keys
    # heuristic: any 2D weight whose key contains a projection-role substring (not excluded).
    for key, t in weights_dict.items():
        if t.dim() != 2 or key in excluded:
            continue
        if any(r in key for r in quant_roles):
            keys.add(key)
    return keys


def quantized_size_bytes(meta: QuantMeta, weights_dict: dict[str, torch.Tensor]) -> int:
    """Total HBM bytes of the QUANTIZED weight set (packed weights + fp16 scales + zeros), the
    roofline numerator for the int4 decode floor."""
    total = 0
    for key, t in weights_dict.items():
        total += t.numel() * t.element_size()
    return total


def dequantize_to(weights_dict: dict[str, torch.Tensor], meta: QuantMeta,
                  dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Reconstruct an fp ``weights_dict`` from a quantized one (drops .scales/.zeros). Useful to
    measure quantization error directly and to give eager a dequantized reference."""
    from instructions.reference import _dequant_weight_rows  # local import (avoid cycle)
    qcode = 7 if meta.bits == 4 else 6
    out: dict[str, torch.Tensor] = {}
    for key, t in weights_dict.items():
        if key.endswith(SCALES_SUFFIX) or key.endswith(ZEROS_SUFFIX):
            continue
        if meta.is_quantized(key):
            info = meta.keys[key]
            scales = weights_dict[key + SCALES_SUFFIX]
            zeros = weights_dict.get(key + ZEROS_SUFFIX)
            W = _dequant_weight_rows(t, scales, qcode, meta.group, zeros, info["K"])
            out[key] = W.to(dtype)
        else:
            out[key] = t.to(dtype) if t.is_floating_point() else t
    return out


__all__ = [
    "quantize_weights", "dequantize_to", "quantized_size_bytes",
    "QuantMeta", "SCALES_SUFFIX", "ZEROS_SUFFIX", "DEFAULT_QUANT_ROLES",
]

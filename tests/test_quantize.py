"""
AMK, CPU-only acceptance test for the WEIGHT-ONLY GROUPWISE QUANTIZER (TCG-QUANT).

The quantizer (schedule/quantize.py) is the "bandwidth lever": it packs Linear projection weights
to groupwise symmetric int4/int8 so decode streams ~4x fewer bytes through HBM. This test pins the
ADVERTISED invariants of that module, GPU-free (pure torch on CPU), so the honesty story -
"correct round-to-nearest quantization with a known RTN error bound, tied embeddings never
quantized", is enforced in code, not just prose:

  * dequantize_to(quantize(W)) round-trips within the RTN error bound (|W - deq| <= scale/2 for
    pure symmetric RTN; the full quantize_weights path stores fp16 scales, so we add the fp16
    relative term);
  * the per-group scales / packed shapes are exactly right (int4 packs two nibbles/byte -> [N,K//2];
    int8 is [N,K]; scales are fp16 [N, ceil(K/group)]);
  * the quantized_size accounting (quantized_size_bytes) sums the actual HBM bytes of the packed
    set, and an int4 set is materially smaller than the fp32 source;
  * the TIED-EMBEDDING key (a tensor read by BOTH a non-linear `embed` op and the `lm_head` GEMV)
    is EXCLUDED from the quant keys, packing it would corrupt the embed gather.

Run:  uv run python -m pytest tests/test_quantize.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from schedule.graph import ModelConfig, ModelGraph  # noqa: E402
from schedule.quantize import (  # noqa: E402
    SCALES_SUFFIX,
    QuantMeta,
    _quantize_matrix,
    dequantize_to,
    quantize_weights,
    quantized_size_bytes,
)

# fp16 has a 10-bit mantissa; storing the per-group scale in fp16 adds at most ~2^-10 relative
# error on the reconstructed value on top of the pure round-to-nearest step.
_FP16_REL = 2.0 ** -10


def _known_matrix(N: int, K: int) -> torch.Tensor:
    """A small, fixed (seeded) weight with a comfortable dynamic range so groups have non-trivial
    scales (a deterministic 'known matrix' rather than a hand-typed one, reproducible + dense)."""
    torch.manual_seed(1234)
    return torch.randn(N, K) * 1.5


# ----------------------------------------------------------------------------------------
# 1) Pure RTN round-trip bound + shapes (via _quantize_matrix, fp32 scales -> exact scale/2 bound)
# ----------------------------------------------------------------------------------------
def test_rtn_roundtrip_within_scale_over_two():
    """Symmetric round-to-nearest: every reconstructed element is within scale/2 of the original
    (the textbook RTN bound). We hold the scales in fp32 here to assert the PURE quantization
    error, isolating it from fp16 scale-storage rounding (covered in the next test)."""
    N, K = 8, 128
    group = 32
    n_groups = (K + group - 1) // group
    W = _known_matrix(N, K)

    for bits in (4, 8):
        packed, scales, zeros = _quantize_matrix(W, group, bits, symmetric=True)
        assert zeros is None, "symmetric quantization must not emit a zero-point tensor"

        # shapes: int4 packs two nibbles per byte (uint8 [N, K//2]); int8 is signed [N, K].
        if bits == 4:
            assert packed.dtype == torch.uint8
            assert tuple(packed.shape) == (N, K // 2)
        else:
            assert packed.dtype == torch.int8
            assert tuple(packed.shape) == (N, K)
        assert tuple(scales.shape) == (N, n_groups)
        assert scales.dtype == torch.float32

        # round-trip with fp32 scales -> the dequant math is the frozen instructions/reference path.
        meta = QuantMeta(group=group, bits=bits, symmetric=True)
        meta.keys["w"] = {"N": N, "K": K, "n_groups": n_groups, "has_zeros": False}
        deq = dequantize_to({"w": packed, "w" + SCALES_SUFFIX: scales}, meta)["w"]
        assert tuple(deq.shape) == (N, K)

        # |W - deq| <= scale/2 (per element; group gk owns columns [gk*group,(gk+1)*group)).
        s_full = scales.repeat_interleave(group, dim=-1)[:, :K]
        err = (W - deq).abs()
        bound = s_full * 0.5 + 1e-6
        assert bool((err <= bound).all()), (
            f"int{bits} RTN error exceeded scale/2: max ratio "
            f"{(err / s_full).max().item():.4f}")


# ----------------------------------------------------------------------------------------
# 2) Full quantize_weights path: fp16 scales, companion-key shapes, meta accounting
# ----------------------------------------------------------------------------------------
def test_quantize_weights_shapes_meta_and_roundtrip():
    """The public entry point quantizes the selected 2D keys to fp16-scaled groupwise int4/int8,
    passes everything else through unchanged, records exact shapes in QuantMeta, and round-trips
    within the RTN bound widened by the fp16 scale-storage term."""
    N, K = 16, 128
    group = 32
    n_groups = (K + group - 1) // group
    W = _known_matrix(N, K)

    for bits in (4, 8):
        wd = {"q_proj.weight": W.clone(), "input_norm": torch.ones(K)}
        qwd, meta = quantize_weights(wd, group=group, bits=bits, symmetric=True,
                                     quant_keys={"q_proj.weight"})

        assert isinstance(meta, QuantMeta)
        assert meta.bits == bits and meta.group == group and meta.symmetric is True
        assert meta.is_quantized("q_proj.weight")

        # non-quantized weight passes through byte-identical (norms are never quantized).
        assert torch.equal(qwd["input_norm"], wd["input_norm"])

        # companion scales tensor exists, is fp16, and has the [N, n_groups] shape; no zeros (sym).
        scales = qwd["q_proj.weight" + SCALES_SUFFIX]
        assert scales.dtype == torch.float16
        assert tuple(scales.shape) == (N, n_groups)
        assert "q_proj.weight.zeros" not in qwd

        # meta records the exact packed/scales shapes the lowerer sizes buffers from.
        info = meta.keys["q_proj.weight"]
        assert info["N"] == N and info["K"] == K and info["n_groups"] == n_groups
        assert info["has_zeros"] is False
        if bits == 4:
            assert info["packed_shape"] == [N, K // 2]
        else:
            assert info["packed_shape"] == [N, K]
        assert info["scales_shape"] == [N, n_groups]

        # round-trip via the public path: scale/2 + |deq|*fp16_rel covers the fp16-scale storage.
        deq = dequantize_to(qwd, meta)["q_proj.weight"]
        s_full = scales.to(torch.float32).repeat_interleave(group, dim=-1)[:, :K]
        err = (W - deq).abs()
        bound = s_full * 0.5 + deq.abs() * _FP16_REL + 1e-4
        assert bool((err <= bound).all()), (
            f"int{bits} full-path error exceeded the RTN+fp16 bound "
            f"(max_err={err.max().item():.6f})")


# ----------------------------------------------------------------------------------------
# 3) quantized_size accounting: int4 set is materially smaller than the fp32 source
# ----------------------------------------------------------------------------------------
def test_quantized_size_accounting():
    """quantized_size_bytes sums the real HBM bytes of the quantized weight set (packed weights +
    fp16 scales). For a single quantized matrix the count must equal packed.nbytes + scales.nbytes,
    and an int4 set must be much smaller than the fp32 weights it replaced (the bandwidth win)."""
    N, K = 32, 256
    group = 64
    W = _known_matrix(N, K)
    wd = {"q_proj.weight": W.clone()}
    fp32_bytes = W.numel() * W.element_size()  # 4 bytes / element

    qwd, meta = quantize_weights(wd, group=group, bits=4, symmetric=True,
                                 quant_keys={"q_proj.weight"})
    total = quantized_size_bytes(meta, qwd)

    packed = qwd["q_proj.weight"]
    scales = qwd["q_proj.weight" + SCALES_SUFFIX]
    expected = packed.numel() * packed.element_size() + scales.numel() * scales.element_size()
    assert total == expected, f"quantized_size_bytes {total} != packed+scales {expected}"

    # int4 (0.5 byte/wt + tiny fp16 scales) must be well under half the fp32 weight bytes.
    assert total < fp32_bytes * 0.5, (
        f"int4 set ({total} B) is not materially smaller than fp32 source ({fp32_bytes} B)")


# ----------------------------------------------------------------------------------------
# 4) Tied-embedding exclusion: a key read by a non-linear op is NEVER quantized
# ----------------------------------------------------------------------------------------
def test_tied_embedding_key_excluded_from_quant():
    """CRITICAL honesty invariant: in a tied-lm_head checkpoint the embedding table is the source
    for BOTH the `embed` gather (needs fp rows) AND the `lm_head` GEMV. Packing it to int4 would
    corrupt the embed gather (it would read packed bytes as fp). _select_quant_keys must EXCLUDE
    any key used by a non-`linear` graph node. We build a tiny graph that ties them and assert the
    shared key is left in fp while an ordinary projection is quantized."""
    K = 128
    N = 16
    cfg = ModelConfig(hidden=K, n_heads=4, n_kv_heads=2, head_dim=K // 4,
                      intermediate=K, vocab=N, n_layers=1)
    g = ModelGraph(config=cfg)
    # Shared, TIED tensor: role lm_head (a quantizable role) but ALSO consumed by the embed op.
    g.add_weight("embed_tokens.weight", (N, K), role="lm_head")
    g.add_weight("q_proj.weight", (N, K), role="q_proj")
    g.add("embed", inputs=["token_id"], outputs=["h"], weights=["embed_tokens.weight"])
    g.add("linear", inputs=["h"], outputs=["logits"], weights=["embed_tokens.weight"],
          attrs={"kind": "lm_head"})          # tied lm_head reuses the embedding table
    g.add("linear", inputs=["h"], outputs=["q"], weights=["q_proj.weight"],
          attrs={"kind": "q_proj"})

    wd = {"embed_tokens.weight": _known_matrix(N, K), "q_proj.weight": _known_matrix(N, K)}
    qwd, meta = quantize_weights(wd, group=32, bits=4, graph=g)

    # the tied embedding/lm_head tensor is excluded; the plain projection is quantized.
    assert "embed_tokens.weight" not in meta.keys, \
        "tied embedding key must be EXCLUDED from quantization (shared with the embed gather)"
    assert "q_proj.weight" in meta.keys, "an ordinary projection should be quantized"

    # and the excluded tensor passes through untouched (still fp, full precision rows).
    assert torch.equal(qwd["embed_tokens.weight"], wd["embed_tokens.weight"])
    assert "embed_tokens.weight" + SCALES_SUFFIX not in qwd


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))

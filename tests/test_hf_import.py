"""
AMK, M3 GENERALITY PROOF: import a REAL HuggingFace Llama and round-trip it
===========================================================================

This is the "any model" milestone. It instantiates a genuine ``transformers.LlamaForCausalLM``
(real HF module, real HF forward semantics) from a *config* with random init, no pretrained
download, imports it via :func:`schedule.graph.from_hf`, lowers ONE decode step (pos=0, empty
KV), runs the resulting megakernel under the reference VM, and asserts the AMK logits equal the
HuggingFace ``model(input_ids).logits[0, -1]`` within fp32 tolerance (rtol/atol = 2e-3).

It exercises two shapes that together prove the importer is real:
  * a plain 2-layer model (per-layer wiring composes), and
  * a GQA model with ``num_key_value_heads < num_attention_heads`` (grouped-query replication),
  * plus a tied-embeddings model (``tie_word_embeddings=True`` -> lm_head sourced from embed).

WHY THIS IS A FAIR ORACLE: AMK's reference ops use the Llama rotate-half RoPE, RMSNorm, GQA,
SiLU SwiGLU, no bias, exactly the math a standard ``LlamaConfig`` selects (attention_bias=False,
hidden_act='silu', default/full RoPE). :func:`from_hf` rejects any config that deviates, so a
pass here means AMK genuinely reproduced HuggingFace's forward pass, not a lookalike.

Run:  uv run python tests/test_hf_import.py     (also a pytest module)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from schedule.graph import from_hf, weights_from_hf  # noqa: E402
from schedule.ir import DType, TARGETS, validate  # noqa: E402
from schedule.lower import POS_NAME, RESHAPE_ID_NAME, TOKEN_NAME, lower  # noqa: E402
from vm.reference_vm import ReferenceVM  # noqa: E402

RTOL = ATOL = 2e-3


def _make_hf(seed: int = 0, **overrides):
    """Build a real, random-init LlamaForCausalLM (fp32) from config, no weight download."""
    from transformers import LlamaConfig, LlamaForCausalLM

    base = dict(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=8, num_key_value_heads=4, vocab_size=320,
        max_position_embeddings=512, rope_theta=10000.0, rms_norm_eps=1e-6,
        attention_bias=False, mlp_bias=False, hidden_act="silu",
        tie_word_embeddings=False,
    )
    base.update(overrides)
    torch.manual_seed(seed)
    m = LlamaForCausalLM(LlamaConfig(**base)).eval().float()
    return m


def _amk_decode_logits(model, tok: int, pos: int = 0):
    """Import -> lower -> validate -> run the decode step under the reference VM. Returns
    (logits[1,vocab], program)."""
    graph = from_hf(model)
    prog = lower(graph, target=TARGETS["rtx5090"], pos=pos, dtype=DType.F32)

    res = validate(prog)
    assert res.ok, "lowered HF program must validate:\n" + res.report()
    bad = [w for w in res.warnings if "RACE" in w or "CYCLE" in w]
    assert not bad, f"validator emitted RACE/CYCLE warnings: {bad}"
    _, stuck = prog.simulate_counters()
    assert stuck == [], f"deadlock: stuck tasks {stuck}"

    inputs = {
        TOKEN_NAME: torch.tensor([tok], dtype=torch.int32),
        POS_NAME: torch.tensor([pos], dtype=torch.int32),
        RESHAPE_ID_NAME: torch.tensor([0], dtype=torch.int32),
    }
    out = ReferenceVM(prog, weights_from_hf(model), device="cpu").run(inputs, kv={})
    return out["logits"], prog


def _hf_decode_logits(model, tok: int) -> torch.Tensor:
    """HuggingFace oracle: first-token logits == decode at pos=0 with an empty cache."""
    with torch.no_grad():
        out = model(input_ids=torch.tensor([[tok]]))
    return out.logits[0, -1].view(1, -1).float()


def test_hf_two_layer_decode_matches_hf():
    """Plain 2-layer Llama (MHA, n_kv == n_heads) round-trips against the HF forward."""
    model = _make_hf(seed=0, num_attention_heads=8, num_key_value_heads=8)
    tok = 7
    amk, prog = _amk_decode_logits(model, tok)
    ref = _hf_decode_logits(model, tok)
    torch.testing.assert_close(amk, ref, rtol=RTOL, atol=ATOL)
    assert sum(t.op.name == "ATTENTION_TILE" for t in prog.tasks) == 2
    assert sum(t.op.name == "SILU_MUL" for t in prog.tasks) == 2


def test_hf_gqa_decode_matches_hf():
    """GQA config: num_key_value_heads (4) < num_attention_heads (8). Proves the grouped-query
    replication the lowerer + attention reference implement matches HF's GQA."""
    model = _make_hf(seed=1, num_attention_heads=8, num_key_value_heads=4)
    assert model.config.num_key_value_heads < model.config.num_attention_heads
    tok = 19
    amk, prog = _amk_decode_logits(model, tok)
    ref = _hf_decode_logits(model, tok)
    torch.testing.assert_close(amk, ref, rtol=RTOL, atol=ATOL)


def test_hf_tied_embeddings_decode_matches_hf():
    """tie_word_embeddings=True: from_hf sources the lm_head buffer from model.embed_tokens.weight
    (HF may keep ``lm_head.weight`` as a shared view of the embedding or omit it, depending on
    version, either way the output projection IS the embedding). The logits must match HF."""
    model = _make_hf(seed=2, tie_word_embeddings=True,
                     num_attention_heads=8, num_key_value_heads=4)
    sd = model.state_dict()
    # If HF still exposes lm_head.weight under tying, it must alias the embedding storage.
    if "lm_head.weight" in sd:
        assert sd["lm_head.weight"].data_ptr() == sd["model.embed_tokens.weight"].data_ptr(), \
            "tied lm_head.weight must share storage with the embedding"
    graph = from_hf(model)
    assert graph.meta.get("lm_head_tied_to_embed") is True
    tok = 11
    amk, _ = _amk_decode_logits(model, tok)
    ref = _hf_decode_logits(model, tok)
    torch.testing.assert_close(amk, ref, rtol=RTOL, atol=ATOL)


def test_hf_odd_head_dim_and_theta_matches_hf():
    """A non-square head layout (head_dim != hidden//heads via explicit head_dim) and a
    non-default rope_theta, to prove config fields are read exactly (not hardcoded defaults)."""
    model = _make_hf(seed=3, hidden_size=128, num_attention_heads=8, num_key_value_heads=2,
                     head_dim=24, rope_theta=50000.0)
    # confirm theta survived into the graph regardless of transformers version layout
    g = from_hf(model)
    assert abs(g.meta["rope_theta"] - 50000.0) < 1e-6, g.meta["rope_theta"]
    assert g.config.head_dim == 24
    tok = 5
    amk, _ = _amk_decode_logits(model, tok)
    ref = _hf_decode_logits(model, tok)
    torch.testing.assert_close(amk, ref, rtol=RTOL, atol=ATOL)


def test_hf_rejects_unsupported_variant():
    """attention_bias=True is a feature the template does not model, from_hf must refuse it
    loudly rather than emit a silently-wrong graph."""
    model = _make_hf(seed=4, attention_bias=True)
    try:
        from_hf(model)
    except NotImplementedError as e:
        assert "attention_bias" in str(e)
    else:
        raise AssertionError("from_hf should reject attention_bias=True")


if __name__ == "__main__":
    test_hf_two_layer_decode_matches_hf()
    print("[1/5] 2-layer Llama (MHA) decode == HuggingFace forward ........ OK")
    test_hf_gqa_decode_matches_hf()
    print("[2/5] GQA (n_kv<n_heads) decode == HuggingFace forward ......... OK")
    test_hf_tied_embeddings_decode_matches_hf()
    print("[3/5] tied embeddings decode == HuggingFace forward ............ OK")
    test_hf_odd_head_dim_and_theta_matches_hf()
    print("[4/5] explicit head_dim + non-default rope_theta == HF ......... OK")
    test_hf_rejects_unsupported_variant()
    print("[5/5] unsupported variant (attention_bias) rejected loudly ..... OK")
    print("\nM3 GENERALITY PROOF PASSED: AMK imports + correctly lowers a real "
          "HuggingFace LlamaForCausalLM.")

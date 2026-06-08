"""
AMK, MULTI-TOKEN GENERATION CONFORMANCE (the proof AMK is a real decoder)
=========================================================================

``test_cuda_decode.py`` proves AMK decodes ONE token correctly. This proves the harder thing:
AMK greedily generates a *sequence* of tokens, threading a persistent KV cache across decode
steps (one launch == one token, ``kv_len = pos+1``, KV_APPEND at ``pos``), and the resulting
token sequence is IDENTICAL to eager greedy decode, token for token, across >= 32 tokens.

We exercise three models through :func:`generate.generate`:

  * the toy decoder, 1-layer and 2-layer (per-layer KV threading composes), and
  * a real from-config ``transformers.LlamaForCausalLM`` (no download), matched against HF's own
    greedy loop.

For each, ``divergence_index == max_tokens`` is the pass condition: AMK never diverged from eager.

Backend: ``MegakernelVM`` on a CUDA device (the real cooperative megakernel), else the CPU
``ReferenceVM`` (the bit-exact scheduling oracle). Both must match eager exactly.

Run:  uv run python tests/test_generate.py     (also a pytest module)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from generate import generate  # noqa: E402
from schedule.ir import DType  # noqa: E402

MAX_TOKENS = 32


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------------------
# Toy models (1L, 2L)
# --------------------------------------------------------------------------------------
def _run_toy(model_id: str, prompt_ids: list[int], max_tokens: int = MAX_TOKENS) -> dict:
    from models.toy import make_toy
    n_layers = 2 if model_id == "toy-2L" else 1
    model = make_toy(seed=0, dtype=torch.float32, n_layers=n_layers)
    # Pass the pre-built object so generate() and the eager oracle share the SAME weights.
    out = generate(model, "rtx5090", prompt_ids, max_tokens,
                   device=_device(), dtype=DType.F32, verify=True)
    return out


def test_generate_toy_1L_matches_eager():
    out = _run_toy("toy-1L", prompt_ids=[7, 11, 3], max_tokens=MAX_TOKENS)
    assert len(out["generated"]) == MAX_TOKENS, out
    assert out["divergence_index"] == MAX_TOKENS, (
        f"toy-1L diverged from eager at index {out['divergence_index']}/{MAX_TOKENS}; "
        f"AMK={out['generated']} eager={out['eager_tokens']}")


def test_generate_toy_2L_matches_eager():
    out = _run_toy("toy-2L", prompt_ids=[19, 5], max_tokens=MAX_TOKENS)
    assert len(out["generated"]) == MAX_TOKENS, out
    assert out["divergence_index"] == MAX_TOKENS, (
        f"toy-2L diverged from eager at index {out['divergence_index']}/{MAX_TOKENS}; "
        f"AMK={out['generated']} eager={out['eager_tokens']}")


# --------------------------------------------------------------------------------------
# Real HuggingFace LlamaForCausalLM (from config, no download)
# --------------------------------------------------------------------------------------
def _make_hf(seed: int = 0, **overrides):
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
    return LlamaForCausalLM(LlamaConfig(**base)).eval().float()


def _hf_greedy(model, prompt_ids: list[int], max_tokens: int) -> list[int]:
    """HuggingFace's OWN greedy loop (do_sample=False), as the independent oracle."""
    with torch.no_grad():
        gen = model.generate(
            input_ids=torch.tensor([prompt_ids], dtype=torch.long),
            max_new_tokens=max_tokens, do_sample=False, num_beams=1,
            pad_token_id=model.config.eos_token_id or 0,
        )
    return gen[0, len(prompt_ids):].tolist()


def test_generate_hf_llama_matches_hf_greedy():
    model = _make_hf(seed=0)
    prompt_ids = [7, 11, 3]
    out = generate(model, "rtx5090", prompt_ids, MAX_TOKENS,
                   device=_device(), dtype=DType.F32, verify=True)
    assert len(out["generated"]) == MAX_TOKENS, out

    # 1) AMK == its own eager forward (the verify path in generate()).
    assert out["divergence_index"] == MAX_TOKENS, (
        f"HF Llama: AMK diverged from eager forward at {out['divergence_index']}/{MAX_TOKENS}; "
        f"AMK={out['generated']} eager={out['eager_tokens']}")

    # 2) AMK == HuggingFace's OWN model.generate(do_sample=False) greedy loop (independent oracle).
    hf_tokens = _hf_greedy(model, prompt_ids, MAX_TOKENS)
    assert out["generated"] == hf_tokens, (
        f"HF Llama: AMK tokens != HF model.generate greedy.\n"
        f"AMK={out['generated']}\nHF ={hf_tokens}")


if __name__ == "__main__":
    dev = _device()
    print(f"== AMK multi-token generation conformance (backend={'MegakernelVM' if dev=='cuda' else 'ReferenceVM'}, "
          f"device={dev}) ==")

    o1 = _run_toy("toy-1L", [7, 11, 3], MAX_TOKENS)
    assert o1["divergence_index"] == MAX_TOKENS, o1
    print(f"[1/3] toy-1L: generated {MAX_TOKENS} tokens, divergence_index="
          f"{o1['divergence_index']} (== max) ... OK  tokens={o1['generated'][:8]}...")

    o2 = _run_toy("toy-2L", [19, 5], MAX_TOKENS)
    assert o2["divergence_index"] == MAX_TOKENS, o2
    print(f"[2/3] toy-2L: generated {MAX_TOKENS} tokens, divergence_index="
          f"{o2['divergence_index']} (== max) ... OK  tokens={o2['generated'][:8]}...")

    model = _make_hf(seed=0)
    prompt = [7, 11, 3]
    o3 = generate(model, "rtx5090", prompt, MAX_TOKENS, device=dev, dtype=DType.F32, verify=True)
    assert o3["divergence_index"] == MAX_TOKENS, o3
    hf_tokens = _hf_greedy(model, prompt, MAX_TOKENS)
    assert o3["generated"] == hf_tokens, (o3["generated"], hf_tokens)
    print(f"[3/3] HF LlamaForCausalLM: AMK == eager == model.generate(do_sample=False) "
          f"for {MAX_TOKENS} tokens ... OK")

    print(f"\nMULTI-TOKEN GENERATION VERIFIED: AMK is a real decoder "
          f"(>= {MAX_TOKENS} tokens, identical to eager greedy) on device={dev}.")

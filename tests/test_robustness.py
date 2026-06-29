"""Production-robustness tests: every bad input must FAIL FAST with a clear error - never hang,
crash, or silently miscompute. The 'stable, zero-error, dynamic' contract made executable.

Each test feeds an adversarial input (an unsupported model, an absurd knob, a deadlocking config) and
asserts the system refuses it loudly. Grown from the production-hardening audit; add a case here for
every new guard so the guarantee stays regression-proof.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from schedule.graph import from_hf


def _cfg(**over):
    """A minimal VALID Llama-family config; override one field to make it adversarial."""
    base = dict(hidden_size=64, num_attention_heads=4, num_key_value_heads=4,
                intermediate_size=128, vocab_size=100, num_hidden_layers=1, rms_norm_eps=1e-6,
                attention_bias=False, mlp_bias=False, hidden_act="silu", rope_theta=10000.0,
                max_position_embeddings=512, tie_word_embeddings=False)
    base.update(over)
    return SimpleNamespace(**base)


class _Model:
    def __init__(self, cfg):
        self.config = cfg


# ---- DYNAMIC robustness: unsupported HF models must be REFUSED, never miscomputed ----

def test_from_hf_rejects_moe():
    with pytest.raises(NotImplementedError, match="mixture-of-experts"):
        from_hf(_Model(_cfg(num_local_experts=8)))


def test_from_hf_rejects_mla():
    with pytest.raises(NotImplementedError, match="(?i)mla|latent"):
        from_hf(_Model(_cfg(kv_lora_rank=512)))


def test_from_hf_rejects_indivisible_hidden():
    with pytest.raises(NotImplementedError, match="divisible"):
        from_hf(_Model(_cfg(hidden_size=65, num_attention_heads=4)))


def test_from_hf_still_rejects_bias_and_nonsilu():
    with pytest.raises(NotImplementedError):
        from_hf(_Model(_cfg(attention_bias=True)))
    with pytest.raises(NotImplementedError):
        from_hf(_Model(_cfg(hidden_act="gelu")))


# ---- OOM/CRASH: absurd or unknown knobs must FAIL FAST, never overflow SMEM ----

def test_normalize_knobs_rejects_absurd_value():
    from vm.loader import _normalize_knobs
    with pytest.raises(ValueError, match="out of the validated range"):
        _normalize_knobs({"cpa_stages": 1_000_000})
    with pytest.raises(ValueError, match="out of the validated range"):
        _normalize_knobs({"cpa_cols": -1})
    out = _normalize_knobs({"cpa_stages": 4, "cols_per_warp": 2})   # valid passes unchanged
    assert out["cpa_stages"] == 4 and out["cols_per_warp"] == 2


def test_normalize_knobs_rejects_unknown():
    from vm.loader import _normalize_knobs
    with pytest.raises(KeyError):
        _normalize_knobs({"not_a_real_knob": 1})


# ---- HANG: a config that deadlocks the cooperative kernel must be refused, not launched ----

def test_loader_refuses_deadlocking_tpb():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from models.toy import make_toy
    from schedule.graph import from_toy
    from schedule.ir import DType, TARGETS, replace
    from schedule.lower import lower
    from schedule.search import default_config
    from vm.loader import MegakernelVM
    m = make_toy(seed=0, n_layers=1)
    t = TARGETS["rtx5090"]
    prog = lower(from_toy(m), target=t, config=replace(default_config(t), threads_per_block=512),
                 pos=0, dtype=DType.F32)
    with pytest.raises(ValueError, match="(?i)deadlock|validated maximum"):
        MegakernelVM(prog, m.weights_dict(), device="cuda")

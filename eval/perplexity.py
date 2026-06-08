"""
AMK, TEACHER-FORCED PERPLEXITY (real generation FIDELITY vs HuggingFace)
========================================================================

``compile.py`` proves AMK decodes ONE token bit-close to eager; ``generate.py`` proves AMK
greedy-decodes a string token-for-token. This module proves the *quantitative* fidelity claim
that underwrites both: over a real run of text, AMK assigns the SAME next-token probabilities as
HuggingFace eager, so its **perplexity matches HF's to a few decimals in fp32**.

It is correctness evidence, not a speed lever, everything here runs on the GPU-free, fp32
``ReferenceVM`` (the bit-exact scheduling oracle), so the numbers are reproducible on any machine.

Teacher-forced NLL, the AMK way
-------------------------------
Given a token sequence ``ids = [t0, t1, ..., t_{N-1}]`` we score each *next* token under the model
that has seen all tokens before it::

    nll = -1/(N-1) * sum_{i=1..N-1}  log p(t_i | t_0..t_{i-1})
    ppl = exp(nll)

AMK is a single-token decoder (one launch == one position), so we feed the sequence ONE position
at a time, threading a persistent KV cache exactly like the real decode loop::

    kv = {}
    for pos in range(N-1):
        prog   = lower(graph, target, pos=pos)         # this position's window: kv_len = pos+1
        out    = ReferenceVM(prog, weights).run(token=ids[pos], pos=pos, kv=kv)
        kv     = {KV_CACHE buffers from out}            # cache grew at index `pos`
        logp   = log_softmax(out['logits'])            # next-token distribution after seeing ids[:pos+1]
        nll   += -logp[ ids[pos+1] ]                    # score the actual next token

HF eager scores the same sequence in a single causal forward pass (``model(input_ids).logits``),
taking ``logits[i]`` as the distribution for predicting ``ids[i+1]``. The two NLLs, and hence the
two perplexities, should agree to within fp32 round-off because AMK is *bit-close to eager*.

The headline number is ``abs(ppl_amk - ppl_hf)`` (and the relative gap). A tiny gap over a few
hundred real tokens is strong evidence AMK is faithful over whole sequences, not just the first
token. Any drift is reported honestly, not hidden.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from schedule.ir import BufferKind, DType, GpuTarget, TARGETS, validate
from schedule.lower import POS_NAME, RESHAPE_ID_NAME, TOKEN_NAME, lower
from vm.reference_vm import ReferenceVM


# ======================================================================================
# Result record
# ======================================================================================
@dataclass
class PerplexityResult:
    """Teacher-forced perplexity of one model over one token sequence.

    * ``nll`` / ``ppl``: mean next-token negative-log-likelihood (nats) and its exp = perplexity.
    * ``n_scored``: number of next-token predictions averaged over (``len(ids) - 1``).
    * ``backend``: 'amk_reference_vm' or 'hf_eager'.
    """

    nll: float
    ppl: float
    n_scored: int
    backend: str
    dtype: str = "f32"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nll": self.nll,
            "ppl": self.ppl,
            "n_scored": self.n_scored,
            "backend": self.backend,
            "dtype": self.dtype,
            "notes": self.notes,
        }


def _kv_names(prog) -> list[str]:
    return [b.name for b in prog.buffers if b.kind == BufferKind.KV_CACHE]


def _step_inputs(token: int, pos: int) -> dict[str, torch.Tensor]:
    """The frozen ReferenceVM run() input contract (CPU, int32)."""
    return {
        TOKEN_NAME: torch.tensor([int(token)], dtype=torch.int32),
        POS_NAME: torch.tensor([int(pos)], dtype=torch.int32),
        RESHAPE_ID_NAME: torch.tensor([0], dtype=torch.int32),
    }


# ======================================================================================
# AMK perplexity (ReferenceVM, KV-threaded, one position at a time)
# ======================================================================================
def amk_perplexity(graph, weights, ids: list[int], *,
                   target: GpuTarget | None = None,
                   dtype: DType = DType.F32,
                   validate_each: bool = False) -> PerplexityResult:
    """Teacher-forced perplexity through AMK's bit-exact ReferenceVM.

    Feeds ``ids`` one position at a time, threading a persistent KV cache, and at each position
    scores the *actual* next token under ``log_softmax(logits)``. Returns the mean NLL (nats) and
    perplexity over the ``len(ids) - 1`` next-token predictions.

    The lowered program is validated at the first step (and at every step if ``validate_each``);
    AMK refuses to score with an invalid (deadlock/race) schedule.
    """
    if len(ids) < 2:
        raise ValueError("need >= 2 tokens to score a next-token perplexity")
    tgt = target if target is not None else TARGETS["rtx5090"]

    kv: dict[str, torch.Tensor] = {}
    total_nll = 0.0
    n_scored = 0
    n = len(ids)

    # We only need positions 0..n-2 to predict tokens 1..n-1.
    for pos in range(n - 1):
        prog = lower(graph, target=tgt, pos=pos, dtype=dtype)
        if pos == 0 or validate_each:
            vr = validate(prog)
            if not vr.ok:
                raise RuntimeError("AMK refuses to score an invalid schedule:\n" + vr.report())
        out = ReferenceVM(prog, weights, device="cpu").run(_step_inputs(ids[pos], pos), kv=kv)
        kv = {nm: out[nm] for nm in _kv_names(prog)}

        logits = out["logits"].detach().to(torch.float32).view(-1)
        logp = torch.log_softmax(logits, dim=-1)
        total_nll += float(-logp[int(ids[pos + 1])].item())
        n_scored += 1

    nll = total_nll / n_scored
    return PerplexityResult(nll=nll, ppl=math.exp(nll), n_scored=n_scored,
                            backend="amk_reference_vm", dtype=dtype.name.lower())


# ======================================================================================
# HF eager perplexity (single causal forward pass), the authoritative reference
# ======================================================================================
def hf_perplexity(model, ids: list[int]) -> PerplexityResult:
    """Teacher-forced perplexity of a HuggingFace ``*ForCausalLM`` over ``ids`` (one forward pass).

    ``logits[i]`` is the distribution for predicting ``ids[i+1]``; we average the NLL of the actual
    next tokens. Run in fp32 for an apples-to-apples comparison with AMK's fp32 ReferenceVM.
    """
    if len(ids) < 2:
        raise ValueError("need >= 2 tokens to score a next-token perplexity")
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids], dtype=torch.long)).logits[0].to(torch.float32)
    # logits[:-1] predicts ids[1:]
    logp = torch.log_softmax(logits[:-1], dim=-1)
    targets = torch.tensor(ids[1:], dtype=torch.long)
    nll_vec = -logp.gather(1, targets.view(-1, 1)).view(-1)
    nll = float(nll_vec.mean().item())
    return PerplexityResult(nll=nll, ppl=math.exp(nll), n_scored=len(ids) - 1,
                            backend="hf_eager", dtype="f32")


def toy_perplexity(model, ids: list[int]) -> PerplexityResult:
    """Teacher-forced perplexity of a ToyLlama (``forward(ids[S]) -> [S, vocab]``)."""
    if len(ids) < 2:
        raise ValueError("need >= 2 tokens to score a next-token perplexity")
    with torch.no_grad():
        logits = model.forward(torch.tensor(ids, dtype=torch.long)).to(torch.float32)
    logp = torch.log_softmax(logits[:-1], dim=-1)
    targets = torch.tensor(ids[1:], dtype=torch.long)
    nll_vec = -logp.gather(1, targets.view(-1, 1)).view(-1)
    nll = float(nll_vec.mean().item())
    return PerplexityResult(nll=nll, ppl=math.exp(nll), n_scored=len(ids) - 1,
                            backend="toy_eager", dtype="f32")


__all__ = ["PerplexityResult", "amk_perplexity", "hf_perplexity", "toy_perplexity"]

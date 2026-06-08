"""
AMK, THE CORRECTNESS ORACLE (full-model logit equivalence vs eager PyTorch)
===========================================================================

This is the *fixed* correctness gate of the autoresearch loop. The whole point of AMK is to
emit a megakernel whose output is **bit-for-bit-within-tolerance** identical to eager PyTorch.
Latency without correctness is a lie; this module is what makes the benchmark honest (see
``eval/bench.py``, which refuses to report a latency unless an oracle :class:`Verdict` says
``correct``).

Two complementary checks, both returning a single :class:`Verdict`:

  * :func:`logit_equivalence`, a *static* numerical comparison of two logit tensors. It reports
    ``max_abs_err``, ``max_rel_err``, ``top1_agreement`` (fraction of positions whose argmax
    matches), and the softmax ``kl`` divergence, then renders a PASS/FAIL using
    **dtype-appropriate tolerances** (fp32 tight ~1e-4; fp16 ~1e-2; bf16 ~2e-2). These are the
    right order of magnitude for a *correct* fused kernel that merely reorders fp32-accumulate
    reductions, a real numerical bug (wrong layout, missing scale, swapped halves) blows past
    them by orders of magnitude.

  * :func:`token_divergence`, a *behavioral* check: greedy-decode ``n_tokens`` from both a
    candidate ``run_fn`` and the eager ``ref_model`` and return the index of the first token at
    which they disagree (``n_tokens`` == they never diverged == perfect). This catches drift that
    a single-step logit compare can miss, because errors compound autoregressively.

Design notes
------------
  * Dependency-light: torch only (numpy optional, unused). No GPU required, runs on whatever
    device the tensors already live on; it only moves things to fp32 on CPU for the final
    scalar reductions so the verdict is deterministic and device-independent.
  * Pure / side-effect-free. The oracle never mutates the model or the tensors it inspects.
  * The :class:`Verdict` is the single currency between oracle, bench, roofline and the flywheel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

# ----------------------------------------------------------------------------------------
# Dtype-appropriate tolerances. Keyed by the *compute/output* dtype of the candidate logits.
# (abs, rel), a position passes if |a-b| <= atol + rtol*|b|, same convention as torch.allclose.
# These are deliberately generous enough to absorb legitimate reduction-order differences of a
# correctly-fused kernel, and tight enough that a real op bug fails. fp32 is the strict gate.
# ----------------------------------------------------------------------------------------
_TOL: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
}
# Fallback for exotic dtypes (fp8 etc.): treat like bf16-ish but a touch looser.
_TOL_DEFAULT = (3e-2, 3e-2)
# A correct kernel must still mostly agree on the actual decoded token. Below this fraction of
# matching argmaxes we fail even if the elementwise error happens to sneak under tolerance.
_TOP1_FLOOR = 0.99


def tolerances_for(dtype: torch.dtype) -> tuple[float, float]:
    """Return (atol, rtol) for a candidate-logit dtype. Public so bench/flywheel can log them."""
    return _TOL.get(dtype, _TOL_DEFAULT)


# ----------------------------------------------------------------------------------------
# The single currency: a Verdict.
# ----------------------------------------------------------------------------------------
@dataclass
class Verdict:
    """The result of one correctness check. ``.correct`` is the only thing downstream gates
    (bench, flywheel) are allowed to trust. Everything else is for humans and the report."""

    correct: bool
    check: str = ""                      # "logit_equivalence" | "token_divergence" | ...
    dtype: str = ""                      # candidate dtype name, e.g. "float16"
    # logit metrics (NaN when not applicable to the check)
    max_abs_err: float = float("nan")
    max_rel_err: float = float("nan")
    top1_agreement: float = float("nan")  # fraction of positions whose argmax matches
    kl: float = float("nan")             # mean KL(softmax(ref) || softmax(test)), nats
    # token metrics
    first_divergence: int = -1           # token index of first mismatch (-1 = n/a)
    n_tokens: int = -1                   # how many tokens were compared
    # tolerances actually applied (for an auditable report)
    atol: float = float("nan")
    rtol: float = float("nan")
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def report(self) -> str:
        head = "PASS" if self.correct else "FAIL"
        lines = [f"[{head}] oracle.{self.check} (dtype={self.dtype})"]
        if self.max_abs_err == self.max_abs_err:  # not NaN
            lines.append(
                f"  max_abs_err={self.max_abs_err:.3e}  max_rel_err={self.max_rel_err:.3e}  "
                f"(atol={self.atol:.1e} rtol={self.rtol:.1e})")
            lines.append(
                f"  top1_agreement={self.top1_agreement:.4f}  kl={self.kl:.3e} nats")
        if self.first_divergence >= 0 or self.n_tokens >= 0:
            verdict = ("identical for all " if self.first_divergence == self.n_tokens
                       else f"diverged at token {self.first_divergence} of ")
            lines.append(f"  token decode: {verdict}{self.n_tokens} tokens")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)

    # convenience for grep-friendly single lines (bench reuses this style)
    def oneline(self) -> str:
        return (f"AutoKernel oracle:{self.check} correctness:{'PASS' if self.correct else 'FAIL'} "
                f"max_abs_err:{self.max_abs_err:.3e} max_rel_err:{self.max_rel_err:.3e} "
                f"top1:{self.top1_agreement:.4f} kl:{self.kl:.3e} dtype:{self.dtype}")


# ----------------------------------------------------------------------------------------
# Static logit equivalence.
# ----------------------------------------------------------------------------------------
def _as_2d_fp32(t: torch.Tensor) -> torch.Tensor:
    """Move to CPU fp32 and flatten to [rows, vocab] so reductions are deterministic and
    device-independent. The last dim is treated as the vocabulary axis."""
    t = t.detach().to(torch.float32).cpu()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    elif t.dim() > 2:
        t = t.reshape(-1, t.shape[-1])
    return t


def logit_equivalence(test_logits: torch.Tensor, ref_logits: torch.Tensor,
                      dtype: torch.dtype | None = None) -> Verdict:
    """Compare candidate logits against the eager reference and render a PASS/FAIL Verdict.

    Args:
      test_logits: logits from the AMK path (reference VM / CUDA megakernel). Any shape whose
        last dim is the vocab axis; [S, V] or [V] are typical.
      ref_logits:  the eager ``model.forward`` logits, same shape.
      dtype:       the dtype whose tolerances to apply. Defaults to ``test_logits.dtype`` so the
        gate matches the precision the candidate actually ran in (a bf16 kernel is judged by
        bf16 tolerances even after we upcast for the comparison math).

    Metrics (all computed in fp32 on CPU for determinism):
      max_abs_err     = max |test - ref|
      max_rel_err     = max |test - ref| / (|ref| + tiny)
      top1_agreement  = fraction of rows whose argmax matches (the decoded-token agreement)
      kl              = mean KL(softmax(ref) || softmax(test)) in nats (distributional distance)

    PASS iff (allclose at the dtype's (atol,rtol)) AND (top1_agreement >= 0.99). Both must hold:
    allclose alone can pass a kernel that is numerically close but flips the argmax on a tie, and
    top1 alone can pass a kernel that agrees on the winner but has the magnitudes wrong.
    """
    cand_dtype = dtype if dtype is not None else test_logits.dtype
    atol, rtol = tolerances_for(cand_dtype)
    notes: list[str] = []

    a = _as_2d_fp32(test_logits)
    b = _as_2d_fp32(ref_logits)
    v = Verdict(correct=False, check="logit_equivalence", dtype=str(cand_dtype).replace("torch.", ""),
                atol=atol, rtol=rtol)

    if a.shape != b.shape:
        v.notes.append(f"SHAPE MISMATCH test{tuple(a.shape)} vs ref{tuple(b.shape)}")
        v.correct = False
        return v

    # NaN/Inf are an automatic, unambiguous failure (a real kernel bug, e.g. unnormalized softmax).
    if not torch.isfinite(a).all():
        notes.append("candidate logits contain NaN/Inf")
    finite = torch.isfinite(a) & torch.isfinite(b)

    diff = (a - b).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    rel = diff / (b.abs() + 1e-12)
    max_rel = float(rel[finite].max()) if finite.any() else float("inf")

    # top-1 (argmax) agreement per row.
    top1 = float((a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean()) if a.numel() else 1.0

    # KL(softmax(ref) || softmax(test)), averaged over rows. Robust to constant logit shifts.
    log_p = torch.log_softmax(b, dim=-1)          # ref as the "true" distribution
    log_q = torch.log_softmax(a, dim=-1)
    p = log_p.exp()
    kl = float((p * (log_p - log_q)).sum(dim=-1).mean()) if a.numel() else 0.0

    allclose = torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False) and torch.isfinite(a).all()
    passed = bool(allclose and top1 >= _TOP1_FLOOR)

    v.max_abs_err, v.max_rel_err, v.top1_agreement, v.kl = max_abs, max_rel, top1, kl
    v.correct = passed
    if not allclose:
        notes.append(f"not allclose at atol={atol:.1e} rtol={rtol:.1e} (max_abs={max_abs:.3e})")
    if top1 < _TOP1_FLOOR:
        notes.append(f"top1_agreement {top1:.4f} < floor {_TOP1_FLOOR}")
    v.notes = notes
    return v


# ----------------------------------------------------------------------------------------
# Behavioral token-divergence check (greedy decode).
# ----------------------------------------------------------------------------------------
def _greedy_next(logits: torch.Tensor) -> int:
    """Argmax of the LAST position's logits -> next token id (int). Accepts [S,V] or [V]."""
    if logits.dim() >= 2:
        logits = logits[-1]
    return int(torch.argmax(logits.detach().to(torch.float32)).item())


def token_divergence(run_fn: Callable[[torch.Tensor], torch.Tensor],
                     ref_model: Any,
                     prompt: torch.Tensor,
                     n_tokens: int,
                     dtype: torch.dtype | None = None) -> Verdict:
    """Greedy-decode ``n_tokens`` continuation tokens from BOTH the candidate and the eager
    reference, starting from ``prompt``, and report the index of the first token they disagree on.

    Both decoders use the *same* prefill+append-and-rerun strategy (so the comparison is apples
    to apples and works for any ``run_fn`` that maps ``input_ids[S] -> logits[S, V]``, including a
    counter-driven ReferenceVM-backed prefill). This is intentionally simple and KV-cache-free at
    the harness level: it tests *output equivalence*, not the candidate's internal caching.

    Args:
      run_fn:    candidate. ``run_fn(input_ids: LongTensor[S]) -> logits[S, V]`` (or [V]).
      ref_model: eager oracle exposing ``ref_model.forward(input_ids) -> logits[S, V]`` (a
                 ``ToyLlama`` or any nn.Module / callable with that contract).
      prompt:    LongTensor[S0] of seed token ids.
      n_tokens:  number of continuation tokens to compare.

    Returns a :class:`Verdict` with ``first_divergence`` in [0, n_tokens]; ``== n_tokens`` means
    the two decoders produced identical tokens for the whole horizon (``correct=True``).
    """
    ref_forward = getattr(ref_model, "forward", ref_model)
    prompt = prompt.detach().to(torch.long).view(-1)

    seq_test = prompt.clone()
    seq_ref = prompt.clone()
    first_div = n_tokens
    notes: list[str] = []
    cand_dtype = dtype

    with torch.no_grad():
        for step in range(n_tokens):
            test_logits = run_fn(seq_test)
            ref_logits = ref_forward(seq_ref)
            if cand_dtype is None:
                cand_dtype = test_logits.dtype
            tok_test = _greedy_next(test_logits)
            tok_ref = _greedy_next(ref_logits)
            if tok_test != tok_ref:
                first_div = step
                notes.append(f"token {step}: candidate->{tok_test} ref->{tok_ref}")
                break
            # advance BOTH with the (agreed) reference token so prefixes stay identical; this
            # isolates per-step model error from compounding off a single early disagreement.
            nxt = torch.tensor([tok_ref], dtype=torch.long, device=seq_ref.device)
            seq_test = torch.cat([seq_test, nxt])
            seq_ref = torch.cat([seq_ref, nxt])

    v = Verdict(correct=(first_div == n_tokens), check="token_divergence",
                dtype=str(cand_dtype).replace("torch.", "") if cand_dtype else "",
                first_divergence=first_div, n_tokens=n_tokens, notes=notes)
    return v


__all__ = ["Verdict", "logit_equivalence", "token_divergence", "tolerances_for"]

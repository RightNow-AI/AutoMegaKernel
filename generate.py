"""
AMK, AUTOREGRESSIVE GENERATION (the real multi-token decoder)
=============================================================

``compile.py`` proves AMK decodes ONE token correctly; this module proves AMK is a real
**decoder**: it greedily generates many tokens, threading a persistent KV cache through the
frozen megakernel VM exactly the way the decode-loop model in :mod:`schedule.ir` prescribes -

    ONE kernel launch == one forward pass == one decoded token.

This host-driven one-launch-per-token loop is AMK's **default, shipped** multi-token decoder. There
is also a separate, EXPERIMENTAL/research path, :class:`vm.loader_persist.PersistentDecodeVM` -
that runs an entire K-token decode loop inside a SINGLE cooperative launch (no per-token host
relaunch). That single-launch path is NOT the default decode and is reachable only from its tests
(``tests/test_persist_decode.py``); ``generate()`` here always uses the per-token relaunch path.

The loop (host-driven, KV in HBM across launches):

    kv = {}                                    # empty cache at pos 0
    for t in range(n_steps):
        prog = lower(graph, target, pos=t)     # this step's window: kv_len = pos+1
        vm   = MegakernelVM | ReferenceVM(prog, weights)
        out  = vm.run(inputs(token_t, pos=t, reshape_id0=[0]), kv=kv)
        kv   = {KV_CACHE buffers from out}      # the cache grew at index `pos`
        next = argmax(out['logits'])            # greedy
        append(next)

Per-layer KV buffers are named ``L{layer}.kcache`` / ``L{layer}.vcache`` (see
:mod:`schedule.lower`); their shape ``(max_seq, n_kv_heads, head_dim)`` is FIXED across positions,
so the same dict threads cleanly and the only thing that advances between launches is the scalar
``pos`` (which moves the KV_APPEND write index and the attention ``kv_len = pos+1``). This is the
host driving the autoregressive loop, keeping each launch well under the WDDM TDR watchdog.

Correctness is proven against eager: :func:`generate` can verify that the AMK token sequence is
*identical* to eager greedy decode (HF ``model.generate(do_sample=False)`` or the toy's own greedy
loop). ``divergence_index == max_tokens`` means a perfect match, AMK decoded the same string as
eager, token for token, across the whole horizon. A real decoder, not a one-step demo.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from schedule.ir import (  # noqa: E402
    BufferKind, DType, GpuTarget, MegakernelProgram, TARGETS, validate,
)
from schedule.lower import POS_NAME, RESHAPE_ID_NAME, TOKEN_NAME, lower  # noqa: E402


# ======================================================================================
# Result record
# ======================================================================================
@dataclass
class GenerateResult:
    """The structured result of an AMK generation run.

    * ``tokens``: the FULL token sequence (prompt + generated).
    * ``generated``: only the newly decoded tokens (``tokens[len(prompt):]``).
    * ``per_step_latency_us``: wall time of each decode step (one launch == one token), µs.
    * ``divergence_index``: when verified against eager, the index of the first generated token
      that disagrees, in ``[0, max_tokens]``; ``== max_tokens`` means a PERFECT match (no
      divergence). ``-1`` when verification was not requested.
    """

    tokens: list[int]
    generated: list[int]
    per_step_latency_us: list[float]
    divergence_index: int = -1
    max_tokens: int = 0
    device: str = ""
    backend: str = ""               # "MegakernelVM" | "ReferenceVM"
    model: str = ""
    gpu: str = ""
    eager_tokens: list[int] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def matches_eager(self) -> bool:
        return self.divergence_index == self.max_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": list(self.tokens),
            "generated": list(self.generated),
            "per_step_latency_us": [round(x, 3) for x in self.per_step_latency_us],
            "divergence_index": self.divergence_index,
            "max_tokens": self.max_tokens,
            "matches_eager": self.matches_eager if self.divergence_index >= 0 else None,
            "device": self.device,
            "backend": self.backend,
            "model": self.model,
            "gpu": self.gpu,
            "eager_tokens": list(self.eager_tokens) if self.eager_tokens is not None else None,
            "notes": self.notes,
        }


# ======================================================================================
# Helpers
# ======================================================================================
def _resolve_target(gpu: str | GpuTarget) -> GpuTarget:
    if isinstance(gpu, GpuTarget):
        return gpu
    if gpu not in TARGETS:
        raise KeyError(f"unknown gpu {gpu!r}; known targets: {', '.join(sorted(TARGETS))}")
    return TARGETS[gpu]


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _torch_dt(dtype: DType) -> torch.dtype:
    return {DType.F32: torch.float32, DType.F16: torch.float16,
            DType.BF16: torch.bfloat16}.get(dtype, torch.float32)


def _load(model_id_or_obj: Any, dtype: torch.dtype):
    """Resolve (model, importer, label) for a model id string, or a pre-built object.

    A string goes through :func:`compile.load_model` (the toy / HF path). A pre-built object is
    used directly: a ``ToyLlama`` (has ``.cfg`` + ``weights_dict``) imports via ``from_toy``; a
    HuggingFace ``*ForCausalLM`` (has ``.config``) imports via ``from_hf`` + ``weights_from_hf``.
    """
    if isinstance(model_id_or_obj, str):
        import compile as _compile
        model, importer, _eager, label = _compile.load_model(model_id_or_obj, dtype=dtype)
        weights = model.weights_dict()
        return model, importer(model), weights, label

    obj = model_id_or_obj
    if hasattr(obj, "cfg") and hasattr(obj, "weights_dict"):       # ToyLlama
        from schedule.graph import from_toy
        return obj, from_toy(obj), obj.weights_dict(), "toy(obj)"
    if hasattr(obj, "config"):                                     # HuggingFace module
        from schedule.graph import from_hf, weights_from_hf
        label = type(obj).__name__
        return obj, from_hf(obj), weights_from_hf(obj), label
    raise TypeError(f"cannot import model object of type {type(obj).__name__}: expected a "
                    f"ToyLlama (.cfg/.weights_dict) or a HF *ForCausalLM (.config)")


def _kv_buffer_names(prog: MegakernelProgram) -> list[str]:
    return [b.name for b in prog.buffers if b.kind == BufferKind.KV_CACHE]


def _step_inputs(token: int, pos: int, device: str) -> dict[str, torch.Tensor]:
    """The frozen run() input contract: token id, absolute position, the constant reshape id [0]."""
    return {
        TOKEN_NAME: torch.tensor([token], dtype=torch.int32, device=device),
        POS_NAME: torch.tensor([pos], dtype=torch.int32, device=device),
        RESHAPE_ID_NAME: torch.tensor([0], dtype=torch.int32, device=device),
    }


def _argmax_logits(logits: torch.Tensor) -> int:
    """Greedy next token from a [1, vocab] (or [vocab]) logits tensor."""
    flat = logits.detach().to(torch.float32).reshape(-1)
    return int(torch.argmax(flat).item())


# ======================================================================================
# Eager greedy oracle (for verification)
# ======================================================================================
def _eager_greedy(model: Any, prompt_ids: list[int], max_tokens: int) -> list[int]:
    """Greedy-decode ``max_tokens`` continuation tokens with the EAGER model (full prefill+rerun).

    Works for both a ``ToyLlama`` (``forward(ids[S]) -> [S, vocab]``) and a HuggingFace
    ``*ForCausalLM`` (``model(input_ids=[[..]]).logits``). This is the authoritative reference the
    AMK KV-cached decode must reproduce token-for-token.
    """
    is_hf = hasattr(model, "config") and not hasattr(model, "cfg")
    seq = list(prompt_ids)
    out: list[int] = []
    with torch.no_grad():
        for _ in range(max_tokens):
            if is_hf:
                logits = model(input_ids=torch.tensor([seq], dtype=torch.long)).logits[0, -1]
            else:
                logits = model.forward(torch.tensor(seq, dtype=torch.long))[-1]
            nxt = int(torch.argmax(logits.to(torch.float32)).item())
            out.append(nxt)
            seq.append(nxt)
    return out


# ======================================================================================
# The generation loop
# ======================================================================================
def generate(model_id_or_obj: Any, gpu: str | GpuTarget, prompt_ids: list[int],
             max_tokens: int, *, device: str = "auto", dtype: DType = DType.F32,
             verify: bool = False, eager_model: Any | None = None) -> dict[str, Any]:
    """Autoregressively greedy-decode ``max_tokens`` tokens from ``prompt_ids`` through AMK.

    Threads a persistent KV cache across decode steps: each step re-lowers the frozen graph at the
    current ``pos`` (so ``kv_len = pos+1`` and KV_APPEND writes index ``pos``), runs the megakernel
    (``MegakernelVM`` on cuda, ``ReferenceVM`` on cpu), reads back the grown KV_CACHE buffers, feeds
    them to the next step, and greedily appends ``argmax(logits)``.

    Args:
      model_id_or_obj:  'toy' / 'toy-2L' / a HF id, OR a pre-built ToyLlama / HF ``*ForCausalLM``.
      gpu:              a registered GpuTarget name (e.g. 'rtx5090') or a GpuTarget.
      prompt_ids:       seed token ids (>= 1 token). Positions 0..len-1 are prefilled one-by-one.
      max_tokens:       number of NEW tokens to generate.
      device:           'auto' (cuda if available else cpu) | 'cuda' | 'cpu'.
      dtype:            element type for the lowered program (F32 = the exact oracle).
      verify:           if True, also eager-greedy-decode and set ``divergence_index`` (== max_tokens
                        on a perfect match). Uses ``eager_model`` if given, else the loaded model.

    Returns a :class:`GenerateResult` as a dict: tokens, generated, per_step_latency_us,
    divergence_index, and context.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one token")
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")

    target = _resolve_target(gpu)
    dev = _resolve_device(device)
    tdt = _torch_dt(dtype)

    model, graph, weights, label = _load(model_id_or_obj, tdt)

    max_seq = graph.config.max_seq
    total_positions = len(prompt_ids) + max_tokens - 1   # last token's logits are unused as input
    if total_positions > max_seq:
        raise ValueError(f"sequence length {total_positions} exceeds model max_seq {max_seq}; "
                         f"shorten the prompt or max_tokens")

    backend = "MegakernelVM" if dev == "cuda" else "ReferenceVM"
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but no CUDA device is available")

    if backend == "MegakernelVM":
        from vm.loader import MegakernelVM as _VM
    else:
        from vm.reference_vm import ReferenceVM as _VM

    notes: list[str] = []
    tokens: list[int] = list(prompt_ids)
    generated: list[int] = []
    per_step_us: list[float] = []

    # Persistent KV cache, threaded across launches. Names are stable across positions; the dict
    # carries the grown cache from one step to the next. Starts empty (zeros) at pos 0.
    kv: dict[str, torch.Tensor] = {}

    # Walk positions 0..total_positions: feed token at each pos, take the LAST step's argmax as the
    # first generated token, then keep feeding the generated token at the next position.
    pos = 0
    cur_token = tokens[0]
    n_prompt = len(prompt_ids)
    n_steps = n_prompt + max_tokens - 1   # forward passes to run (prefill steps + generate steps)

    for step in range(n_steps):
        # Lower THIS step's program at the current position; the only thing that changes per step.
        prog = lower(graph, target=target, config=None, pos=pos, dtype=dtype)
        if step == 0:
            vres = validate(prog)
            if not vres.ok:
                raise RuntimeError("AMK refuses to run an invalid schedule:\n" + vres.report())

        vm = _VM(prog, weights, device=dev)
        ins = _step_inputs(cur_token, pos, dev)

        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = vm.run(ins, kv=kv)
        if dev == "cuda":
            torch.cuda.synchronize()
        per_step_us.append((time.perf_counter() - t0) * 1e6)

        # Thread the grown KV cache forward (only the KV_CACHE outputs; keep on-device for cuda).
        kv = {name: out[name] for name in _kv_buffer_names(prog) if name in out}

        # Decide whether this step's logits feed the next input token.
        next_pos = pos + 1
        if step < n_prompt - 1:
            # still prefilling the prompt: next input is the next prompt token (ignore logits)
            cur_token = tokens[next_pos]
        else:
            # generation phase: greedy-pick from the logits, append, and feed it next.
            nxt = _argmax_logits(out["logits"])
            generated.append(nxt)
            tokens.append(nxt)
            cur_token = nxt
        pos = next_pos

    result = GenerateResult(
        tokens=tokens, generated=generated, per_step_latency_us=per_step_us,
        max_tokens=max_tokens, device=dev, backend=backend, model=label, gpu=target.name,
        notes=notes,
    )

    # ---- optional verification against eager greedy decode (authoritative) ----
    if verify:
        ref_model = eager_model if eager_model is not None else model
        eager = _eager_greedy(ref_model, prompt_ids, max_tokens)
        result.eager_tokens = eager
        div = max_tokens
        for i in range(max_tokens):
            if i >= len(generated) or generated[i] != eager[i]:
                div = i
                notes.append(f"token {i}: AMK->{generated[i] if i < len(generated) else None} "
                             f"eager->{eager[i]}")
                break
        result.divergence_index = div

    return result.to_dict()


__all__ = ["generate", "GenerateResult"]

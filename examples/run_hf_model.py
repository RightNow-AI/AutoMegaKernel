"""
AMK, RUN A REAL HUGGINGFACE CHECKPOINT END-TO-END (the "any model" proof, spec M3)
==================================================================================

This script loads a *real, trained, open* HuggingFace Llama-style checkpoint (default
``HuggingFaceTB/SmolLM2-135M``, Llama arch, ~135M params, no bias, SiLU MLP, full
rotate-half RoPE, tied embeddings) together with its *real tokenizer*, then greedily
decodes K tokens TWO ways and compares them token-by-token:

  1. **Through AMK**: import the module with :func:`schedule.graph.from_hf` -> lower ONE
     decode step per token with :func:`schedule.lower.lower` -> execute under
     :class:`vm.reference_vm.ReferenceVM` (the GPU-independent fp32 correctness oracle),
     threading the persistent KV cache across steps and taking the greedy ``argmax``.
  2. **Through HuggingFace**: ``model.generate(..., do_sample=False)`` (HF's own greedy).

It prints both decoded strings and whether the generated token ids match exactly.

WHY THE REFERENCE VM (not the CUDA megakernel) here: correctness is what the "any model"
milestone proves, and the ReferenceVM is the bit-for-bit conformance oracle the CUDA VM is
checked against (tests/test_cuda_decode.py shows GPU == ReferenceVM to ~1e-7 fp32). Running
the reference VM makes this proof reproducible on any machine, GPU or not. To run the SAME
program on the GPU, swap ``ReferenceVM`` for ``vm.loader.MegakernelVM`` (identical call
surface), see ``--vm cuda``.

THE DECODE LOOP (matches the frozen IR decode model: one launch == one token):
  * AMK lowers a *single-token* forward at absolute position ``pos`` against a persistent KV
    cache. We re-lower per step at the new ``pos`` (cheap, pure Python) and bind the prior
    step's KV tensors so the cache grows by one row each step, exactly the host-driven
    autoregressive loop the IR docstring describes.
  * The KV buffers are named ``L{L}.kcache`` / ``L{L}.vcache`` by the lowerer; ``run()``
    returns them so we feed them straight back in as ``kv=`` on the next step.

Run:
    uv run python examples/run_hf_model.py
    uv run python examples/run_hf_model.py --model HuggingFaceTB/SmolLM2-135M \
        --prompt "The capital of France is" --tokens 16 --vm reference
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from schedule.graph import from_hf, weights_from_hf  # noqa: E402
from schedule.ir import BufferKind, DType, TARGETS  # noqa: E402
from schedule.lower import POS_NAME, RESHAPE_ID_NAME, TOKEN_NAME, lower  # noqa: E402

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_TOKENS = 12


def _kv_buffer_names(prog) -> list[str]:
    """The KV_CACHE buffer names the lowerer emits (``L{L}.kcache`` / ``L{L}.vcache``); these
    are the keys :meth:`ReferenceVM.run` returns and re-accepts as ``kv=`` for the next step."""
    return [b.name for b in prog.buffers if b.kind == BufferKind.KV_CACHE]


def amk_greedy_decode(model, prompt_ids: list[int], n_new: int, *,
                      vm_kind: str = "reference", dtype: DType = DType.F32,
                      verbose: bool = False) -> list[int]:
    """Greedy-decode ``n_new`` tokens through AMK, threading a persistent KV cache.

    We import the HF module once into a ModelGraph, then for EACH position lower a single
    decode step and run it. The first ``len(prompt_ids)`` steps consume the prompt (prefill,
    one token at a time, which is correct because the KV cache accumulates the full context);
    after the prompt we greedily append ``argmax`` of the logits. Returns the new token ids
    (the generated continuation, not including the prompt).

    The math is bit-for-bit AMK: import -> lower -> validate-on-load -> execute. The CUDA VM
    path (``vm_kind='cuda'``) loads the *same program* on the GPU; the reference path runs it
    on the CPU as the GPU-independent oracle.
    """
    graph = from_hf(model)
    weights = weights_from_hf(model)
    target = TARGETS["rtx5090"]

    if vm_kind == "cuda":
        from vm.loader import MegakernelVM as VM  # GPU; loads the identical program
        device = "cuda"
    else:
        from vm.reference_vm import ReferenceVM as VM
        device = "cpu"

    # The full sequence the model has "seen": prompt then greedily-appended tokens. We feed it
    # one token per decode step at the matching absolute position.
    seq: list[int] = list(prompt_ids)
    generated: list[int] = []
    kv: dict[str, torch.Tensor] = {}

    n_steps = len(prompt_ids) + n_new
    for pos in range(n_steps):
        cur_tok = seq[pos]
        prog = lower(graph, target=target, pos=pos, dtype=dtype)
        inputs = {
            TOKEN_NAME: torch.tensor([cur_tok], dtype=torch.int32),
            POS_NAME: torch.tensor([pos], dtype=torch.int32),
            RESHAPE_ID_NAME: torch.tensor([0], dtype=torch.int32),
        }
        vm = VM(prog, weights, device=device)
        out = vm.run(inputs, kv=kv)
        # Re-thread the KV cache for the next step (the lowerer wrote pos `pos`; next step
        # appends pos+1). Detach/keep on whatever device the VM ran on.
        kv = {name: out[name] for name in _kv_buffer_names(prog)}

        # Only sample once we have consumed the whole prompt (the logits at the last prompt
        # token predict the first generated token).
        if pos >= len(prompt_ids) - 1:
            logits = out["logits"].detach().float().view(-1)
            nxt = int(torch.argmax(logits).item())
            generated.append(nxt)
            if len(seq) <= pos + 1:
                seq.append(nxt)
            if verbose:
                print(f"  [amk] pos={pos:>3} token_in={cur_tok:>6} -> argmax={nxt:>6}")
            if len(generated) >= n_new:
                break
    return generated[:n_new]


def hf_greedy_decode(model, tokenizer, prompt_ids: list[int], n_new: int) -> list[int]:
    """HuggingFace's own greedy generation (``do_sample=False``). Returns the new token ids
    only (HF returns prompt + continuation; we slice off the prompt)."""
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=n_new,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    full = out[0].tolist()
    return full[len(prompt_ids):len(prompt_ids) + n_new]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a real HF checkpoint end-to-end through AMK.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace checkpoint id.")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt.")
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS, help="Tokens to generate.")
    ap.add_argument("--vm", choices=["reference", "cuda"], default="reference",
                    help="Which AMK VM to run the lowered program on.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading checkpoint {args.model!r} (real trained weights + tokenizer)...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    graph = from_hf(model)   # fails loudly if the config is outside the supported family
    print("Imported into AMK graph:", graph.summary())
    print(f"  tied_embeddings={graph.meta.get('lm_head_tied_to_embed', False)} "
          f"rope_theta={graph.meta.get('rope_theta')}")

    prompt_ids = tokenizer(args.prompt, return_tensors=None)["input_ids"]
    print(f"\nPrompt: {args.prompt!r}  ->  {len(prompt_ids)} tokens {prompt_ids}")
    print(f"Generating {args.tokens} tokens via AMK ({args.vm} VM) and HF greedy...\n")

    amk_ids = amk_greedy_decode(model, prompt_ids, args.tokens,
                                vm_kind=args.vm, verbose=args.verbose)
    hf_ids = hf_greedy_decode(model, tokenizer, prompt_ids, args.tokens)

    amk_text = tokenizer.decode(amk_ids, skip_special_tokens=True)
    hf_text = tokenizer.decode(hf_ids, skip_special_tokens=True)

    match = amk_ids == hf_ids
    n_match = sum(int(a == b) for a, b in zip(amk_ids, hf_ids))

    print("=" * 78)
    print(f"AMK  ids : {amk_ids}")
    print(f"HF   ids : {hf_ids}")
    print(f"AMK  text: {args.prompt!r} + {amk_text!r}")
    print(f"HF   text: {args.prompt!r} + {hf_text!r}")
    print("-" * 78)
    print(f"token-by-token match: {n_match}/{len(hf_ids)}  ->  "
          f"{'EXACT MATCH' if match else 'MISMATCH'}")
    print("=" * 78)
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())

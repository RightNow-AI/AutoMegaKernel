"""
AMK, BASELINE BENCH CLI (apples-to-apples per-token decode: AMK vs eager vs vLLM)
=================================================================================

Runs the honest per-token DECODE comparison from :mod:`eval.baselines` on the SAME model + SAME
GPU and prints a table. This is the M2 thesis evidence: where the megakernel wins (one cooperative
launch, no per-op kernel-launch/HBM-bubble overhead), and, reported truthfully, where it does
not yet (absolute latency vs PyTorch's highly-optimized per-op kernels on tiny models).

What is measured (all CUDA-event timed, all CORRECTNESS-GATED, no latency without a PASS):
  * eager , the same model doing ONE per-op KV-cached decode step (a stream of kernel launches),
  * amk   , the AMK megakernel decode/token at steady state (persistent tables; one launch/token),
  * vllm  , ATTEMPTED; on this Windows/dev box vLLM has no wheels, so it is recorded status=
             'not_run' with the exact reason + the exact Linux command. NEVER fabricated.

Models:
  --model toy    -> the self-contained toy Llama (default; small, always available),
  --model llama  -> a small from-config transformers.LlamaForCausalLM (NO download), the
                    "small from-config Llama" datapoint. Requires `transformers`.

Run:
  uv run python eval/bench_baselines.py --model toy   --gpu rtx5090
  uv run python eval/bench_baselines.py --model llama --gpu rtx5090
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from eval.baselines import BaselineRecord, decode_comparison  # noqa: E402


# ======================================================================================
# Model factories (both share weights between eager and AMK by passing the SAME object).
# ======================================================================================
def _make_toy(dtype: torch.dtype):
    from models.toy import make_toy
    # A slightly larger toy than the unit-test fixture so the GEMV/HBM work is non-trivial,
    # while staying tiny enough to stay well under the WDDM TDR watchdog.
    return make_toy(seed=0, dtype=dtype, n_layers=2, hidden=256, intermediate=512,
                    n_heads=8, n_kv_heads=4, head_dim=32, vocab=512, max_seq=512)


def _make_llama(dtype: torch.dtype):
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as e:  # pragma: no cover
        raise SystemExit(f"--model llama needs transformers: uv pip install transformers ({e})")
    cfg = LlamaConfig(
        hidden_size=256, intermediate_size=512, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=4, vocab_size=512,
        max_position_embeddings=512, rope_theta=10000.0, rms_norm_eps=1e-6,
        attention_bias=False, mlp_bias=False, hidden_act="silu", tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval().to(dtype)


def make_model(name: str, dtype: torch.dtype):
    if name == "toy":
        return _make_toy(dtype)
    if name == "llama":
        return _make_llama(dtype)
    raise SystemExit(f"unknown --model {name!r}; choose from: toy, llama")


# ======================================================================================
# Table rendering
# ======================================================================================
def _fmt(x, spec="{:.3f}", none="-"):
    return none if x is None else spec.format(x)


def print_table(model_name: str, gpu: str, device: str,
                table: dict[str, BaselineRecord]) -> None:
    eager = table["eager"]
    amk = table["amk"]
    vllm = table["vllm"]

    print()
    print("=" * 84)
    print(f"  AMK PER-TOKEN DECODE BASELINES   model={model_name}  gpu={gpu}  device={device}")
    print("=" * 84)
    hdr = f"  {'baseline':<14}{'status':<9}{'correct':<9}{'ms/token':>12}{'tokens/s':>12}{'%roofline':>12}"
    print(hdr)
    print("  " + "-" * 80)
    for key in ("eager", "amk", "vllm"):
        r = table[key]
        print(f"  {r.name:<14}{r.status:<9}{r.correctness:<9}"
              f"{_fmt(r.ms_per_token, '{:.4f}'):>12}"
              f"{_fmt(r.tokens_per_s, '{:.1f}'):>12}"
              f"{_fmt(r.pct_of_roofline, '{:.1f}'):>12}")
    print("  " + "-" * 80)

    # interpretation
    if eager.status == "ok" and amk.status == "ok" and eager.latency_us and amk.latency_us:
        ratio = eager.latency_us / amk.latency_us
        faster = "AMK FASTER" if ratio > 1.0 else "eager faster"
        print(f"\n  wall-clock per-token:  eager={eager.ms_per_token:.4f} ms   "
              f"amk={amk.ms_per_token:.4f} ms   ->  {ratio:.2f}x ({faster})")
        ko = amk.extra.get("kernel_only_us")
        if ko:
            print(f"  AMK kernel-only (relaunch, no host marshalling): {ko/1e3:.4f} ms/token")
        if amk.pct_of_roofline:
            ach = amk.extra.get("achieved_gbs")
            achs = f"  (~{ach:.0f} GB/s achieved)" if ach else ""
            print(f"  AMK distance to HBM roofline: {amk.pct_of_roofline:.1f}% of bound"
                  f"  =>  {100.0/ (amk.pct_of_roofline/100.0):.1f}% HBM util{achs}"
                  if amk.pct_of_roofline else "")
    else:
        if amk.status != "ok":
            print(f"\n  amk: {amk.status}, {amk.note}")
        if eager.status != "ok":
            print(f"  eager: {eager.status}, {eager.note}")

    print(f"\n  vllm: {vllm.status}, {vllm.note}")
    if vllm.command:
        print(f"        reproduce on Linux: {vllm.command.splitlines()[0]} ...")

    print("\n  grep lines:")
    for key in ("eager", "amk", "vllm"):
        print("    " + table[key].grep_line())
    print()


# ======================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AMK vs eager (+ vLLM) per-token decode comparison")
    ap.add_argument("--model", default="toy", choices=["toy", "llama"],
                    help="toy = self-contained toy Llama; llama = small from-config HF Llama")
    ap.add_argument("--gpu", default="rtx5090", help="GpuTarget name (rtx5090, h100, a100, b200)")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--context-len", type=int, default=16, help="decode position / KV context length")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--dtype", default="f32", choices=["f32", "f16", "bf16"])
    args = ap.parse_args(argv)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    tdt = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    if device == "cpu":
        print("WARNING: no CUDA device, the AMK megakernel cannot run on CPU, so its row will be "
              "status='error' (CPU has only the ReferenceVM oracle, not a perf target). The eager "
              "row will be a CPU REFERENCE timing, not a GPU performance number.")

    model = make_model(args.model, tdt)
    table = decode_comparison(model, args.gpu, device=device,
                              context_len=args.context_len, warmup=args.warmup, iters=args.iters)
    print_table(args.model, args.gpu, device, table)

    # exit non-zero only if a row that SHOULD have measured (eager always; amk on cuda) errored.
    bad = table["eager"].status == "error"
    if device == "cuda":
        bad = bad or table["amk"].status == "error"
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

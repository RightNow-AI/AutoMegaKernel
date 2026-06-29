"""
AMK command-line entry point.

THE UNIFIED AGENT HARNESS - one command, any HuggingFace model: import -> generate a
correctness-gated megakernel -> let the agent (Forge, or the builtin search) optimize it -> report.
Everything below is a component of this one flow.

    uv run python amk_cli.py optimize <hf-model> --gpu <arch>    # the whole flywheel, any model

    uv run python amk_cli.py compile toy --gpu rtx5090 --regime single-stream
    uv run python amk_cli.py doctor          # environment + GPU + toolchain check
    uv run python amk_cli.py verify          # run the VM correctness proofs
    uv run python amk_cli.py corpus          # summarize the flywheel corpus

The coding-agent harness, Loop 1 (instruction tuning; see HARNESS.md):

    uv run python amk_cli.py tune-instruction gemv_tile --gpu rtx5090 --budget 6  # AutoKernel loop

The coding-agent harness (Loop 2, schedule search; see HARNESS.md):

    uv run python amk_cli.py propose toy --gpu rtx5090            # incumbent config + search space
    uv run python amk_cli.py eval toy --gpu rtx5090 --config cfg.json   # prints the JSON verdict
    uv run python amk_cli.py loop toy --gpu rtx5090 --budget 8    # keep/revert autoresearch loop

The UNATTENDED "run it and sleep" autoresearch driver (Loop 2; see HARNESS.md / program.md):

    uv run python amk_cli.py autoresearch toy --gpu rtx5090 --iters 20 --device cpu
    uv run python amk_cli.py autoresearch toy --gpu rtx5090 --minutes 120          # sleep on it
    uv run python amk_cli.py autoresearch toy --gpu rtx5090 --iters 50 --cold      # ignore prior

It proposes -> lowers -> validates -> correctness-gated-evaluates -> keeps/reverts unattended,
logs every experiment to results.tsv + the flywheel corpus, checkpoints each iter (RESUMES if
re-run), and the flywheel prior makes every future run start smarter.

The real autoregressive decoder (multi-token generation, KV cache threaded across steps):

    uv run python amk_cli.py generate toy --gpu rtx5090 --prompt-ids "1,2,3" --max-tokens 32

(The conventional product surface is ``amk compile <model> --gpu <arch>``; with this research
monorepo it is spelled ``uv run python amk_cli.py compile ...``, same thing.)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _doctor() -> int:
    import torch
    from schedule.ir import TARGETS
    print("AMK doctor")
    print(f"  python      : {sys.version.split()[0]}")
    print(f"  torch       : {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"  cuda avail  : {cuda}")
    if cuda:
        cap = torch.cuda.get_device_capability(0)
        print(f"  device      : {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
        p = torch.cuda.get_device_properties(0)
        print(f"  SMs / mem   : {p.multi_processor_count} SMs, {p.total_memory/1024**3:.1f} GB")
    nvcc = _which("nvcc")
    print(f"  nvcc        : {nvcc or 'NOT FOUND (CUDA megakernel build needs it)'}")
    print(f"  targets     : {', '.join(TARGETS)}")
    print("  status      : ready" if cuda and nvcc else
          "  status      : CPU-only (reference VM works; GPU megakernel needs CUDA+nvcc)")
    return 0


def _which(exe: str) -> str | None:
    from shutil import which
    return which(exe)


def _verify() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    rc = 0
    for script in (os.path.join("vm", "verify_vm.py"),):
        print(f"--- {script} ---")
        rc |= subprocess.call([sys.executable, os.path.join(root, script)])
    return rc


def _corpus() -> int:
    from flywheel.log import read_corpus, read_results
    _root = os.path.dirname(os.path.abspath(__file__))
    rows = read_results(os.path.join(_root, "workspace", "results.tsv"))
    corpus = read_corpus(os.path.join(_root, "flywheel", "corpus.jsonl"))
    print(f"results.tsv : {len(rows)} experiment rows")
    for r in rows[-10:]:
        print(f"  [{r.get('tag')}] {r.get('model')}/{r.get('gpu')} "
              f"corr={r.get('correctness')} lat={r.get('latency_us')}us "
              f"({r.get('pct_of_roofline')}% bound) {r.get('description', '')}")
    print(f"flywheel    : {len(corpus)} kept (model,gpu,schedule,result) points")
    return 0


class _StdoutToStderr:
    """Context manager that redirects OS-level fd 1 (stdout) to fd 2 (stderr) so that build
    chatter from the JIT extension (ninja, nvcc), which writes to the raw fd, bypassing Python's
    sys.stdout, does not pollute the machine-readable JSON the CLI prints. The JSON itself is
    written AFTER the context exits, to the real stdout."""

    def __enter__(self):
        sys.stdout.flush()
        self._saved = os.dup(1)
        os.dup2(2, 1)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        os.dup2(self._saved, 1)
        os.close(self._saved)
        return False


def _propose(rest: list[str]) -> int:
    import argparse
    import harness
    ap = argparse.ArgumentParser(prog="amk propose")
    ap.add_argument("model")
    ap.add_argument("--gpu", default="rtx5090")
    ap.add_argument("--config", default=None, help="optional incumbent ScheduleConfig JSON file")
    args = ap.parse_args(rest)
    incumbent = None
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            incumbent = json.load(f)
    out = harness.propose(args.model, args.gpu, incumbent=incumbent)
    print(json.dumps(out, indent=2))
    return 0


def _eval(rest: list[str]) -> int:
    import argparse
    import harness
    ap = argparse.ArgumentParser(prog="amk eval")
    ap.add_argument("model")
    ap.add_argument("--gpu", default="rtx5090")
    ap.add_argument("--config", default=None, help="ScheduleConfig JSON file (default config if omitted)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args(rest)
    config = None
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = json.load(f)
    with _StdoutToStderr():
        verdict = harness.evaluate(args.model, args.gpu, config, device=args.device)
    print(json.dumps(verdict, indent=2))
    # exit non-zero on a rejected/incorrect verdict so CI/agents can gate on it
    return 0 if (verdict.get("valid") and verdict.get("correct")) else 1


def _loop(rest: list[str]) -> int:
    import argparse
    import harness
    ap = argparse.ArgumentParser(prog="amk loop")
    ap.add_argument("model")
    ap.add_argument("--gpu", default="rtx5090")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default=os.path.join("workspace", "results.tsv"))
    args = ap.parse_args(rest)
    with _StdoutToStderr():
        out = harness.loop(args.model, args.gpu, budget=args.budget, device=args.device,
                           seed=args.seed, results_path=args.results, verbose=True)
    best = out.get("best_verdict")
    print(json.dumps({
        "best_verdict": best,
        "best_config": out.get("best_config"),
        "n_trials": out.get("n_trials"),
        "n_valid": out.get("n_valid"),
        "n_correct": out.get("n_correct"),
        "results_tsv": out.get("results_tsv"),
    }, indent=2))
    return 0 if (best and best.get("correct")) else 1


def _tune_instruction(rest: list[str]) -> int:
    import argparse
    import loop1
    ap = argparse.ArgumentParser(prog="amk tune-instruction")
    ap.add_argument("op", help="ABI op to tune (e.g. gemv_tile, rmsnorm, silu_mul, add, "
                              "rope, attention_tile, embed)")
    ap.add_argument("--gpu", default="rtx5090")
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--device", action="store_true",
                    help="accepted for symmetry with the Loop-2 harness; Loop 1 always builds + "
                         "times on the local CUDA device (the reference is the oracle).")
    ap.add_argument("--results", default=os.path.join("workspace", "results.tsv"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(rest)
    # Redirect ninja/nvcc build chatter (raw fd 1) to stderr so only the JSON summary hits stdout.
    with _StdoutToStderr():
        out = loop1.tune_instruction(args.op, args.gpu, budget=args.budget, dtype=args.dtype,
                                     results_path=args.results, tag=args.tag, verbose=True)
    print(json.dumps(out, indent=2))
    # exit non-zero if any built variant failed correctness (the keep-only-correct gate)
    return 0 if out.get("all_correct") and out.get("best_variant") is not None else 1


def _autoresearch(rest: list[str]) -> int:
    import argparse
    import autoresearch as _ar
    ap = argparse.ArgumentParser(prog="amk autoresearch")
    ap.add_argument("model", help="'toy' / 'toy-2L' or a HuggingFace model id")
    ap.add_argument("--gpu", default="rtx5090")
    ap.add_argument("--iters", type=int, default=None, help="number of iterations (default 20)")
    ap.add_argument("--minutes", type=float, default=None,
                    help="wall-clock budget instead of/with --iters (sleep-on-it mode)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="cpu => analytic cost-model fitness (fast/deterministic); "
                         "cuda/auto => measured GPU latency (correctness-gated)")
    ap.add_argument("--cold", action="store_true",
                    help="ignore the flywheel prior (pure exploration; grows the corpus)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epsilon", type=float, default=_ar.DEFAULT_EPSILON)
    ap.add_argument("--corpus", default=_ar.DEFAULT_CORPUS)
    ap.add_argument("--results", default=_ar.DEFAULT_RESULTS)
    ap.add_argument("--state", default=None)
    ap.add_argument("--overnight", action="store_true",
                    help="run-it-and-sleep mode: never stop on a plateau; basin-hop to fresh "
                         "regions while keeping the global best; write workspace/amk_overnight_report"
                         ".{json,md}. Use with a long --minutes (e.g. --minutes 480 for ~8h).")
    ap.add_argument("--restart-after", type=int, default=6,
                    help="overnight: basin-hop after this many consecutive non-improvements")
    args = ap.parse_args(rest)
    # Redirect any JIT/build chatter (raw fd 1) to stderr; the human-readable trajectory + final
    # report are printed by autoresearch() to the real stdout AFTER the context exits.
    out = _ar.autoresearch(
        args.model, args.gpu, iters=args.iters, minutes=args.minutes, device=args.device,
        cold=args.cold, seed=args.seed, epsilon=args.epsilon, corpus_path=args.corpus,
        results_path=args.results, state_path=args.state,
        overnight=args.overnight, restart_after=args.restart_after, verbose=True)
    print(json.dumps({
        "model": out["model"], "gpu": out["gpu"], "device": out["device"], "cold": out["cold"],
        "iters_run": out["iters_run"], "baseline_us": out["baseline_us"],
        "best_us": out["best_us"], "best_pct_roofline": out["best_pct_roofline"],
        "best_kind": out["best_kind"], "speedup_vs_baseline": out["speedup_vs_baseline"],
        "iters_to_best": out["iters_to_best"], "n_kept": out["n_kept"],
        "n_correct": out["n_correct"], "n_rejected": out["n_rejected"],
        "n_crash": out["n_crash"], "warm_seeds": out["warm_seeds"],
        "ranker_trained": out["ranker_trained"], "state_path": out["state_path"],
    }, indent=2))
    return 0 if out["best_us"] is not None else 1


def _generate(rest: list[str]) -> int:
    import argparse
    import harness
    ap = argparse.ArgumentParser(prog="amk generate")
    ap.add_argument("model", help="'toy' / 'toy-2L' or a HuggingFace model id")
    ap.add_argument("--gpu", default="rtx5090")
    ap.add_argument("--prompt-ids", required=True,
                    help="comma-separated seed token ids, e.g. '1,2,3'")
    ap.add_argument("--max-tokens", type=int, default=32, help="number of NEW tokens to generate")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--verify", action="store_true",
                    help="also eager-greedy-decode and report the token divergence index")
    args = ap.parse_args(rest)
    prompt_ids = [int(x) for x in args.prompt_ids.split(",") if x.strip() != ""]
    if not prompt_ids:
        raise SystemExit("--prompt-ids must contain at least one integer token id")
    with _StdoutToStderr():
        out = harness.generate(args.model, args.gpu, prompt_ids, args.max_tokens,
                               device=args.device, verify=args.verify)
    print(json.dumps({
        "tokens": out["tokens"],
        "generated": out["generated"],
        "per_step_latency_us": out["per_step_latency_us"],
        "divergence_index": out["divergence_index"],
        "max_tokens": out["max_tokens"],
        "matches_eager": out["matches_eager"],
        "device": out["device"],
        "backend": out["backend"],
        "model": out["model"],
        "gpu": out["gpu"],
    }, indent=2))
    # exit non-zero only when verification ran AND found a divergence (CI/agent gate)
    if out["divergence_index"] >= 0 and out["divergence_index"] != out["max_tokens"]:
        return 1
    return 0


def _optimize(rest: list[str]) -> int:
    """THE UNIFIED AGENT HARNESS. One command, any HF model: import -> generate a correctness-gated
    megakernel -> let the agent (Forge if installed, else AMK's builtin search) optimize it -> report.
    Everything else (compile/loop/eval/generate/autoresearch) is a component of this one flow."""
    import argparse
    import autoresearch as _ar
    ap = argparse.ArgumentParser(
        prog="amk optimize",
        description="Unified agent harness: import any HF Llama-family model, generate a correct "
                    "megakernel, agent-optimize it (correctness-gated), and report - one command.")
    ap.add_argument("model", help="'toy' / 'toy-2L' or any HuggingFace Llama-family model id")
    ap.add_argument("--gpu", default="rtx5090")
    ap.add_argument("--iters", type=int, default=None, help="optimize iterations (default 20)")
    ap.add_argument("--minutes", type=float, default=None, help="wall-clock budget (sleep-on-it mode)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="cuda/auto => measured GPU latency (correctness-gated); cpu => fast analytic")
    ap.add_argument("--cold", action="store_true", help="ignore the flywheel prior (pure exploration)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(rest)

    # THE AGENT: the private Forge proposer drives candidate generation when installed (the moat);
    # otherwise AMK's builtin search. Either way every candidate flows through the SAME public gate
    # (lower -> validate -> reference-oracle -> measured keep/revert), so the result is always correct.
    proposer, agent_name = None, "builtin search"
    try:
        from amk_forge import ForgeProposer
        proposer, agent_name = ForgeProposer(), "Forge agent"
    except Exception:
        pass

    print(f"AMK optimize: {args.model} on {args.gpu}  (agent: {agent_name})", file=sys.stderr)
    out = _ar.autoresearch(
        args.model, args.gpu, iters=args.iters, minutes=args.minutes, device=args.device,
        cold=args.cold, seed=args.seed, proposer=proposer, verbose=True)
    dev = out["device"]
    measured = (dev == "cuda")   # analytic (cpu) searches the cost model; it does not MEASURE correctness
    found = out["best_us"] is not None
    correct = (out["n_correct"] > 0) if measured else None
    ok = found and (correct is not False)
    print(json.dumps({
        "model": out["model"], "gpu": out["gpu"], "device": dev, "agent": agent_name,
        "generated": "megakernel" if found else "FAILED (no valid config found)",
        "baseline_us": out["baseline_us"], "optimized_us": out["best_us"],
        "speedup_vs_baseline": out["speedup_vs_baseline"],
        "pct_of_roofline": out["best_pct_roofline"],
        "correct": correct,
        "correctness": ("measured bit-exact vs eager HF (oracle-gated)" if measured else
                        "cost-model search over validated configs; numerical correctness gated on GPU build"),
        "iters_run": out["iters_run"], "iters_to_best": out["iters_to_best"],
        "n_kept": out["n_kept"], "n_correct": out["n_correct"], "n_rejected": out["n_rejected"],
        "vs_vllm": "run `modal run modal_app.py::bench_suite --gpu <X>` for the AMK-vs-vLLM scoreboard",
    }, indent=2))
    return 0 if ok else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "optimize":
        return _optimize(rest)
    if cmd == "compile":
        import compile as _c
        _c.main(rest)
        return 0
    if cmd == "doctor":
        return _doctor()
    if cmd == "verify":
        return _verify()
    if cmd == "corpus":
        return _corpus()
    if cmd == "propose":
        return _propose(rest)
    if cmd == "eval":
        return _eval(rest)
    if cmd == "loop":
        return _loop(rest)
    if cmd in ("tune-instruction", "tune_instruction"):
        return _tune_instruction(rest)
    if cmd == "autoresearch":
        return _autoresearch(rest)
    if cmd == "generate":
        return _generate(rest)
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    print(f"unknown command {cmd!r}. Try: optimize | compile | doctor | verify | corpus | "
          f"propose | eval | loop | tune-instruction | autoresearch | generate")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

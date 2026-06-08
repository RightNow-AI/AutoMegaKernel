# AGENTS.md, AutoMegaKernel for native coding agents (Codex / Claude Code)

This is the Codex/native-agent guide to driving **AutoMegaKernel (AMK)**, the Codex analogue of
`CLAUDE.md`. It tells an agent exactly what AMK is, the one edit surface it owns, the canonical
tool/CLI names, the non-negotiable honesty rules, and a copy-paste optimization loop. Everything
here maps 1:1 to **existing, verified** code; nothing is faked.

For the full human/agent contract read [`HARNESS.md`](HARNESS.md). For what AMK is and what is
measured-real today, read [`README.md`](README.md).

---

## What AMK is

AMK compiles a HuggingFace Llama-family model into **ONE persistent CUDA megakernel**, the whole
forward pass fused into a single cooperative kernel launch, and self-improves the schedule over
its own baseline, correctness-gated, unattended. It is the sibling of AutoKernel (which tunes one
*kernel*); AMK tunes the whole-model *megakernel* via a new search axis: **the schedule**.

The agent's job is to **search the schedule**, not to write kernels. The frozen VM deterministically
lowers a `ScheduleConfig` into a runnable megakernel and *proves it deadlock/race-free before any
launch*. An unsafe config is a clean `REJECTED`, never a hung GPU.

## The edit surface, ScheduleConfig + kernel_knobs ONLY

You edit **one structured object**: a `ScheduleConfig` (a JSON dict of typed knobs), optionally with
a `kernel_knobs` sub-object. You NEVER touch raw kernel code, `vm/`, `Task.sm`, or the frozen ABI.
The schema is [`schemas/schedule_config.schema.json`]; the live, machine-readable knob list comes
from `amk_propose(...)["search_space"]`. The knobs:

| knob | type | choices | meaning |
|---|---|---|---|
| `tiling.gemv.N_tile` | int | 64,128,256,512 | GEMV output-column tile width (the one tiling knob lowered today) |
| `tiling.attention.kv_block` | int | 64,128,256 | KV window block (recorded; not yet lowered) |
| `fusion_grouping` | list[list[str]] | `[]`, `[["gate","up"]]`, ... | op groups to co-reside (a safe hint) |
| `sm_assignment` | str/dict | `round_robin`/`load_balance`/map | SM placement policy |
| `pipelining_depth` | int | 0–4 typ. | instructions ahead to prefetch weights (hides the HBM bubble) |
| `page_allocation` | str | `graph_color`/`linear`/`none` | activation page reuse policy |
| `threads_per_block` | int | 128,256,512 | persistent VM block size (occupancy-proven) |
| `smem_bytes_per_block` | int | 0,16384,49152 | dynamic SMEM opt-in (over-cap = clean reject) |

`kernel_knobs` (e.g. instruction `-D` macros the autoresearch/loop drivers explore) is part of the
search space the loop mutates; pass it inside the config dict to `amk_eval` as a `"kernel_knobs"` object.

## HARD HONESTY RULES (these are enforced in code, not just stated)

- **Correctness FIRST.** A latency is NEVER reported without a correctness PASS vs the CPU
  ReferenceVM. Keep a candidate only if it is correct **AND ≥1% faster** than the incumbent.
- **validate-before-launch.** An unsafe `ScheduleConfig` is a clean `REJECTED` (deadlock/race-free
  proof), never a hung GPU.
- **The edit surface is `ScheduleConfig` + `kernel_knobs` ONLY**, never raw kernel code, never
  `vm/`, never the frozen ABI.
- **Measured-gpu latency is drift-robust;** physically-impossible sub-roofline latencies are withheld.
- **All speedups are vs AMK's OWN baseline**, NOT a claim of beating cuBLAS/vLLM (AMK is currently
  within ~13% of cuBLAS at batch-1, behind it, and says so).

## Canonical tool surface (MCP), use these EXACT names

The MCP server module is [`amk_mcp.py`](amk_mcp.py) (repo root). It exposes:

- `amk_doctor()` → torch/CUDA availability, device name, registered GpuTargets.
- `amk_propose(model, gpu="rtx5090")` → incumbent `ScheduleConfig` + editable `search_space`
  (includes the `kernel_knobs` sub-surface).
- `amk_eval(model, gpu, config, device="auto")` → structured verdict
  `{valid, correct, latency_us, latency_kind, pct_of_roofline, bound_us, schedule_id, ...}`.
  `config` is a JSON object (a `ScheduleConfig`, optionally with a `"kernel_knobs"` object).
- `amk_loop(model, gpu, budget=8, device="auto")` → keep/revert loop result
  `{best_verdict, best_config, rows, ...}`.
- `amk_autoresearch(model, gpu, minutes=None, iters=None, device="auto", overnight=False, cold=False)`
  → unattended campaign result.
- `amk_orchestrate_status()` / `amk_orchestrate_next()` / `amk_orchestrate_report()` → campaign
  state-machine snapshots (structured dicts).
- `amk_orchestrate_record(status, latency_us=None, pct_roofline=None, kind=null, config=None, description="")`
  → record one experiment outcome (`status` ∈ kept|revert|failed|crash|timeout|rejected).

## CLI verbs that already exist (an agent may shell out instead)

```
amk propose|eval|loop|autoresearch|compile|generate|doctor      # the `amk` console command
python amk_orchestrate.py status|next|record|report             # the campaign state machine
```
Every `amk <cmd>` is also `uv run python amk_cli.py <cmd> ...`. The `eval`/`tune-instruction`/etc.
verbs print **only JSON on stdout** (build chatter → stderr), safe to pipe into `jq`/`json.loads`.
`eval` exit code is `0` iff valid+correct (gate your loop on it).

## Codex MCP server config (`~/.codex/config.toml`)

Add this table so Codex launches the AMK MCP server (the `agent` extra pulls in the `mcp` SDK):

```toml
[mcp_servers.automegakernel]
command = "uv"
args = ["run", "--extra", "agent", "python", "amk_mcp.py"]
```

(Claude Code reads the equivalent project config from [`.mcp.json`](.mcp.json) at the repo root.)
The `mcp` SDK is optional; install it once with `uv sync --extra agent`. The tool LOGIC in
`amk_mcp.py` imports ONLY existing AMK modules, so the package stays importable without it.

## The copy-paste loop a Codex agent runs

### Hand-driven: propose → eval → keep/revert (one knob at a time)

```bash
# 1) read the edit surface (the "program.md" step)
uv run python amk_cli.py propose toy --gpu rtx5090 > surface.json

# 2) write a candidate config (edit ONE knob), evaluate it (JSON-only on stdout)
cat > cfg.json <<'JSON'
{ "tiling": {"gemv": {"N_tile": 128}, "attention": {"kv_block": 128}},
  "fusion_grouping": [], "sm_assignment": "load_balance",
  "pipelining_depth": 3, "page_allocation": "graph_color",
  "threads_per_block": 256, "smem_bytes_per_block": 0 }
JSON
uv run python amk_cli.py eval toy --gpu rtx5090 --config cfg.json   # exit 0 => valid+correct

# 3) KEEP the candidate only if it is correct AND >=1% faster than the incumbent; else REVERT.
#    Repeat with the next single-knob edit.

# 4) or let the built-in keep/revert loop do all of the above:
uv run python amk_cli.py loop toy --gpu rtx5090 --budget 16
```

Equivalent via MCP tools: `amk_propose` → `amk_eval` (loop, keeping on correctness-then-≥1%) →
`amk_loop` to automate. Programmatic Python is in HARNESS.md §6.

### Unattended: the autoresearch driver (run it and sleep)

```bash
# fast, deterministic cost-model pass (no GPU)
uv run python amk_cli.py autoresearch toy --gpu rtx5090 --iters 20 --device cpu

# sleep on it: an ~8h real-GPU campaign that never stops on a plateau (basin-hops, preserves best)
uv run python amk_cli.py autoresearch small --gpu rtx5090 --minutes 480 --device cuda --overnight
# wake up:  workspace/amk_overnight_report.md   (best config + speedup vs AMK's OWN baseline)
```

Via MCP: `amk_autoresearch(model, gpu, minutes=..., overnight=true)`. Drive the campaign brain with
`amk_orchestrate_status` / `amk_orchestrate_next` / `amk_orchestrate_record` / `amk_orchestrate_report`
(or `python amk_orchestrate.py status|next|record|report`). Both the headless driver and an agent
talking to the orchestrator write the **same** `results.tsv` + flywheel corpus, so AMK gets smarter
every run regardless of who drove it.

## Targets & models

`<gpu>` is a registered `GpuTarget`: `rtx5090`, `b200`, `h100`, `a100`. `<model>` is `toy` / `toy-2L`
(fully supported) or a HuggingFace id (best-effort via `schedule.graph.from_hf`). The harness uses
the **local** GPU/CPU only, it never touches Modal or any cloud GPU.

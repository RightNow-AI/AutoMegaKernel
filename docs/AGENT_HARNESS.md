# AMK as a Native Coding-Agent Harness

AutoMegaKernel (AMK) is not just a CLI an agent can shell out to, it is a **native coding-agent
harness**. The same verified substrate documented in [`HARNESS.md`](../HARNESS.md) (`propose → eval
→ loop → autoresearch`, correctness ALWAYS first, then a strict ≥1% latency win) is exposed
*natively* to coding agents through every modern agent surface: an **MCP server**, **Claude Code**
skills / slash commands / subagents / workflows (ultracode) / goals, and **Codex** AGENTS.md + MCP.

Nothing here changes the substrate's behavior. Every integration is a thin, faithful adapter over
the **same** functions in `harness.py`, `autoresearch.py`, and `amk_orchestrate.py`. The honesty
rules below are enforced *in those modules*, not re-implemented per surface, an agent driving AMK
through MCP, a slash command, or a raw CLI gets the identical, drift-robust, correctness-gated
verdict.

> **Read first:** [`HARNESS.md`](../HARNESS.md) is the full human/agent contract for the two
> optimization loops and the `ScheduleConfig` + `kernel_knobs` edit surface. This document is the
> *agent-native packaging* of that contract. The two never disagree; if they ever do, `HARNESS.md`
> and the code win.

---

## The canonical tool surface (single source of truth)

Every integration in this document uses these EXACT names. Do not invent variants.

### MCP tools (server module: [`amk_mcp.py`](../amk_mcp.py) at the repo root)

| MCP tool | What it returns | Backed by |
|---|---|---|
| `amk_doctor()` | JSON: torch/cuda availability, device name, registered `GpuTarget`s | env probe |
| `amk_propose(model, gpu="rtx5090")` | incumbent `ScheduleConfig` + editable `search_space` (incl. the `kernel_knobs` sub-surface) | `harness.propose` / `harness.search_space` |
| `amk_eval(model, gpu, config, device="auto")` | structured verdict `{valid, correct, latency_us, latency_kind, pct_of_roofline, bound_us, schedule_id, ...}`. `config` is a JSON object (a `ScheduleConfig`, optionally with a `"kernel_knobs"` object) | `harness.evaluate` |
| `amk_loop(model, gpu, budget=8, device="auto")` | keep/revert loop result `{best_verdict, best_config, rows, ...}` | `harness.loop` |
| `amk_autoresearch(model, gpu, minutes=null, iters=null, device="auto", overnight=false, cold=false)` | unattended campaign result | `autoresearch.autoresearch` |
| `amk_orchestrate_status()` / `amk_orchestrate_next()` / `amk_orchestrate_report()` | campaign state machine (structured dicts) | `amk_orchestrate.*` |
| `amk_orchestrate_record(status, latency_us=null, pct_roofline=null, kind=null, config=null, description="")` | record one experiment outcome | `amk_orchestrate.record` |

These map 1:1 to the existing verified Python signatures (read the modules for the authoritative
arglists):

```text
harness.py
  propose(model_id, gpu, *, incumbent=None)
  evaluate(model_id, gpu, config_dict, device="auto")
  loop(model_id, gpu, budget=8, device="auto", ...)
  search_space(target)                 # target = schedule.ir.TARGETS[gpu]
autoresearch.py
  autoresearch(model, gpu, *, iters=None, minutes=None, device="auto",
               cold=False, overnight=False, restart_after=6, state_path=None, verbose=True)
amk_orchestrate.py
  load_state(path) / get_or_create_state(model, gpu, path)
  record(state, *, latency_us, status, config=None, correctness="PASS",
         pct_of_roofline=None, latency_kind=None, schedule_id="", description="")
  cmd_status / cmd_next / cmd_report(state)   # these PRINT, MCP builds structured dicts from state
```

### CLI verbs (an agent may also shell out)

```text
amk propose|eval|loop|autoresearch|compile|generate|doctor|tune-instruction
python amk_orchestrate.py status|next|record|report
```

Every `amk <cmd>` is equivalently `uv run python amk_cli.py <cmd>`. The full human/agent contract is
[`HARNESS.md`](../HARNESS.md).

---

## 1. MCP server, `amk_mcp.py`

The MCP server exposes the canonical tool surface to any MCP-speaking agent (Claude Code, Codex,
and other MCP clients) over stdio.

### Design: importable without the `mcp` SDK

The `mcp` Python SDK is an **optional** dependency. So all tool LOGIC lives in plain module-level
functions in `amk_mcp.py` (`tool_doctor()`, `tool_propose(...)`, `tool_eval(...)`,
`tool_loop(...)`, `tool_autoresearch(...)`, `tool_orchestrate_status/next/report/record(...)`) that
import **only** the existing AMK modules, there is no `import mcp` at module top level. The
FastMCP / official-SDK server is wired inside `main()`, which **lazy-imports** `mcp`. Consequences:

- The tools are **importable and unit-testable WITHOUT** the `mcp` package
  (`from amk_mcp import tool_eval` works in the bare `uv sync` env).
- The server is **runnable** the moment `mcp` is installed (`python amk_mcp.py`).

### Enable it

```bash
# uv (recommended)
uv sync --extra agent          # installs the optional `mcp` SDK alongside AMK

# or plain pip
pip install "automegakernel[agent]"
```

### Run it

```bash
uv run python amk_mcp.py       # serves the canonical tools over stdio (MCP)
```

### Register it with Claude Code, `.mcp.json`

The repo ships a working `.mcp.json` at the repo root, Claude Code auto-discovers it. It looks
exactly like this (server name `automegakernel`, args include `--extra agent` so the `mcp` SDK
is installed on first run; no `cwd` needed when launched from the repo root):

```jsonc
{
  "mcpServers": {
    "automegakernel": {
      "command": "uv",
      "args": ["run", "--extra", "agent", "python", "amk_mcp.py"]
    }
  }
}
```

Then in Claude Code the tools appear as `amk_doctor`, `amk_propose`, `amk_eval`, `amk_loop`,
`amk_autoresearch`, `amk_orchestrate_status`, `amk_orchestrate_next`, `amk_orchestrate_report`,
`amk_orchestrate_record`.

> **Note for fresh cloners:** `.claude/settings.local.json` is gitignored, so the MCP server is
> not auto-enabled on a clean clone. Enable it either via **Settings > MCP** in the Claude Code
> UI, or by creating `.claude/settings.local.json` at the repo root with the contents
> `{"enabledMcpjsonServers":["automegakernel"]}`.

### Register it with Codex, `~/.codex/config.toml`

```toml
[mcp_servers.automegakernel]
command = "uv"
args = ["run", "--extra", "agent", "python", "amk_mcp.py"]
cwd = "/abs/path/to/AutoMegaKernel"
```

### Copy-paste MCP session (the agent loop)

```text
amk_doctor()
  -> { "torch": true, "cuda": true, "device": "NVIDIA GeForce RTX 5090 ...",
       "targets": ["rtx5090", "b200", "h100", "a100"] }

amk_propose("toy", "rtx5090")
  -> { "schedule_config": {...}, "schedule_id": "sch_...",
       "search_space": { "tiling.gemv.N_tile": {...}, "pipelining_depth": {...},
                         "kernel_knobs.cpasync": {...}, ... } }

amk_eval("toy", "rtx5090",
         { "pipelining_depth": 3, "kernel_knobs": { "cpasync": 1, "cpa_stages": 4 } })
  -> { "valid": true, "correct": true, "latency_us": 1228.03,
       "latency_kind": "measured-gpu", "pct_of_roofline": ..., "schedule_id": "sch_..." }

amk_loop("toy", "rtx5090", budget=16)
  -> { "best_verdict": { "schedule_id": "sch_...", "latency_us": ..., "correct": true },
       "best_config": {...}, "rows": [ ... ] }
```

The verdict schema and its honesty contract are documented field-by-field in
[`HARNESS.md` §3](../HARNESS.md). A latency is **never** present without `correct=true`; an unsafe
config comes back as a clean `valid=false` with a `rejected_reason`.

---

## 2. Claude Code

AMK ships a full set of Claude Code agent-mode artifacts under `.claude/`. Each wraps the canonical
tools/CLI, none re-implements the substrate.

### The skill, `megakernel-optimization`

A Claude Code **skill** that teaches the agent the AMK methodology: read the edit surface with
`amk_propose`, propose ONE knob change, `amk_eval`, keep on correctness-then-≥1%-latency, and stop
on the move-on criteria. The skill embeds the HARD HONESTY RULES (below) so the agent never reports
a latency without a correctness PASS and never edits outside `ScheduleConfig` + `kernel_knobs`.

```
.claude/skills/megakernel-optimization/SKILL.md
```

Invoke it conversationally ("optimize the toy megakernel on rtx5090") or the agent triggers it
autonomously when a megakernel-optimization task is detected.

### Slash commands

| Command | What it does | Wraps |
|---|---|---|
| `/amk-optimize` | Interactive keep/revert tuning of a `(model, gpu)`'s `ScheduleConfig` + `kernel_knobs` | `amk_propose` → `amk_eval` (or `amk_loop`) |
| `/amk-autoresearch` | Launch an unattended campaign (budget in minutes/iters; `--overnight` for the run-and-sleep mode) | `amk_autoresearch` |
| `/amk-compile` | Compile a model to a verified megakernel + report | `amk compile` |

```
.claude/commands/amk-optimize.md
.claude/commands/amk-autoresearch.md
.claude/commands/amk-compile.md
```

Example:

```text
/amk-optimize toy --gpu rtx5090 --budget 16
/amk-autoresearch small --gpu rtx5090 --minutes 480 --overnight
/amk-compile toy --gpu rtx5090 --regime single-stream
```

### The subagent, `amk-megakernel-optimizer`

A dedicated Claude Code **subagent** that runs the full keep/revert loop in its own context window
(so a long tuning session does not crowd the main thread). It is the right tool for "go tune this
for N trials and come back with the best correct config." It uses `amk_propose` / `amk_eval` /
`amk_loop` and reports the best **measured, correct** verdict plus the kept config.

```
.claude/agents/amk-megakernel-optimizer.md
```

### The ultracode `/workflow`

A deterministic, multi-step **workflow** (ultracode) that drives an autoresearch campaign to a stop
criterion: propose → eval → keep/revert → record → check `amk_orchestrate_next` → repeat, then emit
the campaign report. It is the scripted version of "run the methodology to convergence."

```
.claude/workflows/megakernel-autoresearch.js
```

Run it with `/workflow megakernel-autoresearch` (ultracode picks up the JS workflow definition).

### The `/goal`

A standing **goal** that an agent can adopt for a session: "drive the megakernel toward the HBM
roofline, honestly." It encodes the success metric (a correct, measured, ≥1%-better-than-incumbent
schedule, reported as a speedup vs AMK's OWN baseline) and the stop conditions
(`MOVE_ON_CRITERIA`).

```
.claude/goals/optimize-megakernel.md
```

Adopt it with `/goal optimize-megakernel`.

---

## 3. Codex

### `AGENTS.md`

`AGENTS.md` at the repo root is the Codex-native brief: the canonical tool/CLI names, the two
loops, the `ScheduleConfig` + `kernel_knobs` edit surface, and the HARD HONESTY RULES verbatim. A
Codex agent reads `AGENTS.md` on entry and drives AMK through the CLI verbs and/or the MCP tools.

### MCP

Codex consumes the **same** `amk_mcp.py` server. Register it in `~/.codex/config.toml` as shown in
§1. The tool names and verdict schema are identical across Claude Code and Codex, there is exactly
one tool surface.

### Copy-paste (Codex, shelling out)

```bash
uv run python amk_cli.py propose toy --gpu rtx5090 > surface.json
# edit ONE knob into cfg.json, then:
uv run python amk_cli.py eval toy --gpu rtx5090 --config cfg.json   # prints ONLY JSON; exit 0 = valid+correct
uv run python amk_cli.py loop toy --gpu rtx5090 --budget 16
```

---

## 4. The HARD HONESTY RULES (verbatim) + the edit surface

Every doc, skill, slash command, subagent, workflow, and goal in this harness states and enforces
these. They are implemented in `harness.py` / `autoresearch.py` / `eval/`, not re-asserted per
surface.

- **Correctness FIRST:** a latency is NEVER reported without a correctness PASS vs the CPU
  ReferenceVM. Keep a candidate only if it is correct AND ≥1% faster than the incumbent.
- **validate-before-launch:** an unsafe `ScheduleConfig` is a clean REJECTED (deadlock/race-free
  proof), never a hung GPU.
- **The edit surface is `ScheduleConfig` + `kernel_knobs` ONLY**, never raw kernel code, never
  `vm/` or the frozen ABI.
- **Measured-gpu latency is drift-robust;** physically-impossible sub-roofline latencies are
  withheld.
- **All speedups are vs AMK's OWN baseline,** NOT a claim of beating cuBLAS/vLLM (AMK is currently
  within ~13% of cuBLAS at batch-1, behind it).

### The edit surface, exactly

An agent edits a single JSON object, a `ScheduleConfig`, optionally carrying a reserved
`"kernel_knobs"` sub-object. Nothing else.

| Surface | Knobs (see [`HARNESS.md` §2](../HARNESS.md) + `schemas/schedule_config.schema.json`) |
|---|---|
| `ScheduleConfig` | `tiling.gemv.N_tile`, `tiling.attention.kv_block`, `fusion_grouping`, `sm_assignment`, `pipelining_depth`, `page_allocation`, `threads_per_block`, `smem_bytes_per_block` |
| `kernel_knobs` (reserved key; GEMV build knobs, move MEASURED latency only) | `cols_per_warp`, `cpasync`, `cpa_stages`, `cpa_cols` |

A config **without** `kernel_knobs` lowers byte-identically to the production VM build (it is the
exact incumbent). The frozen VM lowers the config deterministically; `validate()` gates the launch.

---

## 5. Which mode when

| Mode | Surface | Use when | Driving |
|---|---|---|---|
| **Interactive** | `/amk-optimize`, MCP `amk_propose`/`amk_eval`, or the CLI | You are iterating by hand and want to see each verdict; small budgets; learning the surface | Human-in-the-loop, one knob at a time |
| **Subagent** | `amk-megakernel-optimizer` | "Go tune this and come back with the best correct config" without crowding the main context | Claude Code subagent runs `amk_loop` in its own window |
| **Workflow / ultracode** | `/workflow megakernel-autoresearch` | You want a deterministic, scripted propose→eval→record loop to a stop criterion, reproducibly | `.claude/workflows/megakernel-autoresearch.js` |
| **Overnight autoresearch** | `/amk-autoresearch ... --overnight` or `amk_autoresearch(..., overnight=true)` | "Run it and sleep", hours of correctness-gated search, basin-hopping, resumable, wake-up report | `autoresearch.autoresearch` headless driver |
| **/goal** | `/goal optimize-megakernel` | A standing objective for a whole session: push toward the roofline honestly, with the move-on criteria as stop conditions | The agent self-directs against the goal's success metric |

All five drive the **same** keep/revert methodology and write the **same** `results.tsv` + flywheel
corpus, so the data flywheel learns from every run regardless of which mode drove it.

---

## See also

- [`HARNESS.md`](../HARNESS.md), the full two-loop integration contract (edit surface, verdict
  schema, keep/revert rules, safety model, autoresearch).
- [`README.md`](../README.md), what is real today (all numbers measured locally).
- [`docs/IR_SPEC.md`](IR_SPEC.md), the standard megakernel IR the schedule edits target.
- `schemas/schedule_config.schema.json`, the machine-readable edit-surface schema.
</content>
</invoke>

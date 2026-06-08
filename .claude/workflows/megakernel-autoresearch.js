// AMK megakernel autoresearch workflow (ultracode / Workflow tool).
// Fan out N agents that each propose ONE distinct ScheduleConfig+kernel_knobs candidate, eval each
// (correctness-gated) via the canonical CLI, adversarially verify the best, then keep + record it.
//
// Honesty rules every agent must obey (see HARNESS.md):
//  - Correctness FIRST: no latency without a correctness PASS vs the CPU ReferenceVM.
//  - Keep only correct AND >=1% faster than the incumbent (measured, drift-robust).
//  - Edit surface is ScheduleConfig + kernel_knobs ONLY, never kernel code, never vm/, never the ABI.
//  - Speedups are vs AMK's OWN baseline, NOT a claim of beating cuBLAS/vLLM.
//  - validate-before-launch: an unsafe config is a clean REJECTED, never a hung GPU.
//
// Runtime constraint: NO clock / random APIs, vary candidates by item index only.

export const meta = {
  name: "megakernel-autoresearch",
  description:
    "Correctness-gated propose->eval->verify->keep AMK megakernel search: N one-knob candidates, " +
    "adversarial re-check of the best, recorded to the orchestrator. Honest speedup vs AMK's own baseline.",
  phases: ["Propose", "Evaluate", "Verify"],
};

// ---- args (with safe defaults) ----------------------------------------------------------------
const model = args.model || "toy";
const gpu = args.gpu || "rtx5090";
const N = Math.max(1, Number(args.candidates || 4));

// Deterministic one-knob variation menu (indexed, never random). Each entry edits exactly ONE knob
// away from the default so keep/revert stays a smooth hill-climb. Index wraps if N exceeds the menu.
const KNOB_VARIATIONS = [
  'set tiling.gemv.N_tile to 64',
  'set tiling.gemv.N_tile to 128',
  'set pipelining_depth to 3',
  'set threads_per_block to 512',
  'set kernel_knobs.cols_per_warp to 2',
  'set kernel_knobs.cpa_stages to 3',
  'set page_allocation to "graph_color"',
  'set sm_assignment to "load_balance"',
];

log(`AMK autoresearch: model=${model} gpu=${gpu} candidates=${N}`);

// ============================================================================================
phase("Propose");
// Fan out N agents; each writes ONE distinct candidate config (cfg_<i>.json). Variation by index.
const proposeThunks = [];
for (let i = 0; i < N; i++) {
  const variation = KNOB_VARIATIONS[i % KNOB_VARIATIONS.length];
  proposeThunks.push(() =>
    agent(
      `You are an AMK schedule optimizer. Read the live edit surface with\n` +
        `  uv run python amk_cli.py propose ${model} --gpu ${gpu}\n` +
        `Take the incumbent ScheduleConfig and make EXACTLY ONE change: ${variation}.\n` +
        `kernel_knobs go under the reserved "kernel_knobs" key inside the config JSON.\n` +
        `Write the resulting config to cfg_${i}.json (valid JSON, no comments).\n` +
        `Use ONLY the canonical amk CLI. Edit surface is ScheduleConfig + kernel_knobs ONLY, ` +
        `never kernel code, never vm/, never the frozen ABI. Output the path cfg_${i}.json.`,
      { label: `propose-${i}`, phase: "Propose" }
    )
  );
}
parallel(proposeThunks);

// ============================================================================================
phase("Evaluate");
// Eval each candidate via the correctness-gated CLI on the real GPU. eval prints ONLY JSON and
// exits 0 iff valid && correct, the agent must report latency ONLY on a correctness PASS.
const evalThunks = [];
for (let i = 0; i < N; i++) {
  evalThunks.push(() =>
    agent(
      `Evaluate AMK candidate cfg_${i}.json (correctness-gated, real GPU):\n` +
        `  uv run python amk_cli.py eval ${model} --gpu ${gpu} --config cfg_${i}.json --device cuda\n` +
        `The command prints ONLY the JSON verdict and exits 0 iff valid && correct.\n` +
        `Honesty: report latency_us ONLY when correct=true; an invalid config is a clean REJECTED ` +
        `(rejected_reason) with NO latency. Return the verdict's valid, correct, latency_us, ` +
        `latency_kind, pct_of_roofline, and schedule_id for cfg_${i}.json.`,
      { label: `eval-${i}`, phase: "Evaluate" }
    )
  );
}
parallel(evalThunks);

// ============================================================================================
phase("Verify");
// Adversarially re-check the single best candidate, then keep + record it to the orchestrator.
agent(
  `From the Evaluate verdicts, pick the BEST candidate = the cfg_<i>.json that is valid AND correct ` +
    `AND has the lowest measured latency_us (latency_kind must be "measured-gpu"). If none is both ` +
    `valid and correct, STOP and report that no candidate qualified (do NOT invent a number).\n\n` +
    `ADVERSARIAL re-check of the winner:\n` +
    `  1. Re-run: uv run python amk_cli.py eval ${model} --gpu ${gpu} --config <best>.json --device cuda\n` +
    `     Confirm correct=true again (correctness is authoritative vs the CPU ReferenceVM) and that ` +
    `     latency reproduces.\n` +
    `  2. Sanity the latency is NOT sub-roofline: it must be >= bound_us (pct_of_roofline >= ~100%). ` +
    `     A physically-impossible sub-roofline latency is an ARTIFACT, withhold it, do not keep it.\n` +
    `  3. Confirm it is >=1% faster than the incumbent (the proposed default from amk propose). If ` +
    `     not faster, it is recorded as "revert" (correct but not kept), not kept.\n\n` +
    `If the winner passes all three, KEEP it and record it:\n` +
    `  uv run python amk_orchestrate.py record kept --latency-us <us> --pct-roofline <pct> ` +
    `--kind measured-gpu --config <best>.json --description "autoresearch workflow: kept"\n` +
    `Then run: uv run python amk_orchestrate.py status\n\n` +
    `Report the measured, drift-robust speedup over AMK's OWN baseline (with latency_kind). This is ` +
    `NOT a claim of beating cuBLAS/vLLM. Use ONLY the canonical amk CLI; obey every honesty rule.`,
  { label: "verify-best", phase: "Verify" }
);

log("Workflow complete: best correct candidate verified + recorded (or none qualified).");

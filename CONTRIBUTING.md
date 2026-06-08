# Contributing to AutoMegaKernel

Thanks for your interest in AMK. This guide covers how to set up, test, and propose changes.

## Development setup

AMK uses [uv](https://docs.astral.sh/uv/). A bare sync provisions the full dev environment
(pytest, ruff, the HF importer, the CUDA JIT helper, and the reporting deps):

```bash
uv sync                  # installs the editable package + the `amk` console command
```

The correctness-bearing core (the IR, the deadlock/race validator, the CPU reference VM,
lowering, HF import, eval, and search) is GPU-free. A CUDA GPU + nvcc is only needed to build
and measure the actual megakernel.

## Running tests

```bash
uv run pytest            # full suite
uv run ruff check .      # lint
```

The CUDA megakernel/instruction tests (`tests/test_cuda_*.py`) **auto-skip** when no GPU is
available (see `conftest.py`), so the suite is green on a laptop, in CI, and on a GPU box alike.
On a GPU box every test runs.

## The hard rule: correctness gates everything

AMK never reports a performance number without a paired correctness PASS. This is enforced in
code, not just in review (`eval/bench.py` physically refuses to time a kernel that did not match
the oracle). See [`HARNESS.md`](HARNESS.md) for the full honesty contract:

- No latency without a correctness PASS.
- No fake measurements, `latency_kind` is `measured-gpu` only for a real CUDA-event-timed run
  whose output matched eager; otherwise it is the clearly-labelled analytic `predicted`.
- No latency for a rejected/invalid schedule.

If you contribute a kernel or schedule change, the correctness verdict comes first; a latency
claim that is not gated on a PASS will not be accepted.

## Proposing changes

- Open a pull request against the default branch.
- Keep changes focused. For tuning work, follow the two AutoKernel-style loops documented in
  [`HARNESS.md`](HARNESS.md): Loop 1 edits one ABI micro-kernel; Loop 2 edits one
  `ScheduleConfig` object.
- Make sure `uv run pytest` and `uv run ruff check .` are green before requesting review.

## Frozen contracts

Two surfaces are **frozen** and changes to them require strong justification plus keeping their
regression tests green:

- **The schedule IR validator** (deadlock- and race-freedom; `schedule/ir.py`). Its guarantees
  are covered by [`tests/test_validator_races.py`](tests/test_validator_races.py).
- **The instruction ABI** (`vm/abi.h`). Its cross-language sync is covered by
  [`tests/test_abi_sync.py`](tests/test_abi_sync.py).

A change that touches either contract must explain why and must keep both test files passing.

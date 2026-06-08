# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in AutoMegaKernel, please report it privately by email
to **security@rightnowai.co**. Do not open a public issue for security-sensitive reports.

Please include enough detail to reproduce the issue. We will acknowledge your report and work
with you on a fix and coordinated disclosure.

## Code generation and execution model

AMK **generates and JIT-compiles CUDA code** (via `torch.utils.cpp_extension` + nvcc) and runs
it on your GPU. As with any code-generation/compilation tool, you should only run models and
configurations you trust. Importing an untrusted model graph or schedule means compiling and
executing code derived from it on your machine.

## Safety by construction

AMK's schedule validator **statically rejects unsafe schedules before any GPU launch**. Every
schedule is lowered into a task-DAG and proven deadlock-free and race-free (acyclic DAG,
satisfiable waits, happens-before provenance for every read) and checked for launch-config
feasibility against the target. A schedule that fails any of these is returned as a clean
`REJECTED` verdict and is never launched, an unsafe schedule is a rejection, not a hung GPU.

This is a safety guarantee about schedule execution, not a sandbox for untrusted model inputs;
the trust guidance above still applies.

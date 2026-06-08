"""
AMK, HOST LOADER for the SINGLE-LAUNCH K-TOKEN PERSISTENT DECODE kernel (the novel path)
========================================================================================

``PersistentDecodeVM`` runs an ENTIRE K-token greedy decode loop inside ONE cooperative kernel
launch (``vm/scheduler_persist.cu`` :: ``amk_megakernel_persist``), no per-token host relaunch,
no per-token host marshalling/sync. This is the capability that distinguishes AMK from vLLM /
TRT-LLM (which relaunch or per-step-CUDA-graph each decoded token) and even from AMK's own
baseline ``MegakernelVM`` (which launches once PER token).

How it works, end to end:

  1. Reuse the EXACT per-step decode program the baseline runs (``schedule.lower.lower`` at the
     generation base position) and the SAME device-table builder (we instantiate a stock
     ``MegakernelVM`` and call its ``_run_full_build`` once, so buffers / instructions / queues /
     scratch / counters / KV caches are laid out byte-identically to the baseline).
  2. PREFILL the prompt with the baseline per-token path so the KV cache holds positions
     ``0 .. n_prompt-2`` exactly as ``generate.py`` does, then
  3. Fire ONE ``persist_launch`` that loops K steps in-kernel: each step sets the position +
     input-token IO cells, runs the per-step DAG (with a register-local position patch for
     KV_APPEND.pos / ATTENTION_TILE.kv_len), in-kernel argmaxes the logits to the next token,
     zeroes the counters, and feeds the token forward. The host reads back the K token ids.

The K tokens this single launch produces are token-for-token identical to the baseline
per-token-relaunch path (``MegakernelVM`` driven by ``generate.py``), because the per-step math,
the position advance, and the greedy argmax all match the reference exactly, proven in
``tests/test_persist_decode.py``.

WDDM/TDR: one launch runs K passes back-to-back, so K must stay under the ~2s local watchdog
(K=8..16 on the laptop). On a no-TDR datacenter GPU K can be large, where the single-launch win
is biggest.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from schedule.ir import BufferKind, MegakernelProgram
from schedule.lower import POS_NAME, TOKEN_NAME  # noqa: E402
from vm.loader import MegakernelVM, _build_extension, _HERE  # noqa: E402

_LOGITS_NAME = "logits"

# Persistent-kernel extension cache (distinct from MegakernelVM's _EXT_CACHE).
_PERSIST_EXT_CACHE: dict[str, Any] = {}


def _build_persist_extension():
    """JIT-build (cached) the persistent-decode extension. Reuses the SAME nvcc flags / MSVC
    discovery / per-arch gencode that vm.loader._build_extension uses (by piggy-backing on it to
    ensure the toolchain is primed), then compiles scheduler_persist.cu + sync.cu + pages.cu into a
    distinct module. The drift guard asserts the POD byte layout matches the numpy packer."""
    cap = torch.cuda.get_device_capability()
    cc = f"{cap[0]}{cap[1]}"
    if cc in _PERSIST_EXT_CACHE:
        return _PERSIST_EXT_CACHE[cc]

    # Prime the toolchain exactly as the baseline does (MSVC path, TORCH_CUDA_ARCH_LIST, gencode).
    # Building the baseline extension first also guarantees the shared headers compile cleanly.
    _build_extension()

    arch = f"{cap[0]}.{cap[1]}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    if os.name == "nt":
        from vm.loader import _ensure_msvc_on_path
        _ensure_msvc_on_path()
    from torch.utils.cpp_extension import load

    sources = [
        os.path.join(_HERE, "scheduler_persist.cu"),
        os.path.join(_HERE, "vm_ext_persist.cpp"),
        os.path.join(_HERE, "sync.cu"),
        os.path.join(_HERE, "pages.cu"),
    ]
    nvcc_flags = [
        f"-gencode=arch=compute_{cc},code=sm_{cc}",
        f"-gencode=arch=compute_{cc},code=compute_{cc}",
        "--extended-lambda",
        "-std=c++17",
    ]
    if os.name == "nt":
        nvcc_flags += ["-Xcompiler", "/Zc:preprocessor",
                       "-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"]
    cflags = (["/std:c++17", "/Zc:preprocessor"] if os.name == "nt" else ["-std=c++17"])
    ext = load(
        name=f"amk_persist_ext_sm{cc}",
        sources=sources,
        extra_include_paths=[_HERE],
        extra_cuda_cflags=nvcc_flags,
        extra_cflags=cflags,
        verbose=True,
    )
    # drift guard: persistent kernel uses the SAME PODs; assert byte layout matches the packer.
    from vm.loader import _assert_layout
    _assert_layout(ext.persist_struct_sizes())
    _PERSIST_EXT_CACHE[cc] = ext
    return ext


class PersistentDecodeVM:
    """Single-launch K-token greedy decoder. One ``MegakernelProgram`` (the per-step decode DAG)
    drives K in-kernel decode steps per ``decode()`` call.

    Construct with the SAME (program, weights) a baseline ``MegakernelVM`` would use for one decode
    step at the generation base position. ``decode(first_token, base_pos, K, kv=...)`` runs K steps
    in ONE cooperative launch and returns the K sampled token ids.
    """

    def __init__(self, program: MegakernelProgram, weights: dict[str, torch.Tensor],
                 device: str = "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("PersistentDecodeVM requires a CUDA device")
        self.prog = program
        self.weights = weights
        self.device = torch.device(device)
        # The stock baseline VM builds + owns the persistent device tables (buffers/instructions/
        # queues/scratch/counters/KV). We reuse them verbatim so the single-launch loop runs the
        # SAME program bytes the baseline would.
        #
        # GEMV path: the persistent kernel (scheduler_persist.cu) is built WITHOUT -DAMK_GEMV_CPASYNC,
        # so it compiles the proven register/coalesced warp-per-column GEMV. We build base_vm with
        # the matching cpasync=0 so (a) the shared device tables' dynamic-smem sizing matches what the
        # persistent kernel expects (no cp.async ring smem), and (b) the prefill / baseline-table
        # build uses the SAME stable GEMV the persistent loop will execute, apples-to-apples.
        self.base_vm = MegakernelVM(program, weights, device=device, knobs={"cpasync": 0})
        self.ext = _build_persist_extension()

        # Resolve the buffer ids of the IO cells the in-kernel loop drives.
        self._pos_bid = self._find_buffer(POS_NAME, BufferKind.IO_INPUT)
        self._token_bid = self._find_buffer(TOKEN_NAME, BufferKind.IO_INPUT)
        self._logits_bid = self._find_buffer(_LOGITS_NAME, BufferKind.IO_OUTPUT)
        self.last_status: dict | None = None

    def _find_buffer(self, name: str, kind: BufferKind) -> int:
        for b in self.prog.buffers:
            if b.name == name and b.kind == kind:
                return b.id
        raise KeyError(f"PersistentDecodeVM: required buffer '{name}' ({kind.name}) not in program")

    def decode(self, first_token: int, base_pos: int, n_tokens: int,
               kv: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
        """Greedily decode ``n_tokens`` tokens in ONE cooperative kernel launch.

        Args:
          first_token: token id fed at step 0 (the last prompt token).
          base_pos:    absolute position of step 0 (== prompt length so far, i.e. n_prompt-1).
          n_tokens:    K, the number of tokens to generate in this single launch.
          kv:          optional prefilled KV cache (positions 0..base_pos-1). If omitted, the cache
                       is whatever the most recent base_vm.run() left in HBM (the persistent tables).

        Returns {"tokens": [K ints], "status": <launch status dict>}.
        """
        if n_tokens < 1:
            raise ValueError("n_tokens (K) must be >= 1")

        # Seed the persistent device tables + KV cache. The FIRST decode() builds + uploads ALL
        # tables (buffers/instructions/queues/scratch/counters/KV) via one baseline run() at base_pos.
        # Subsequent decode()s only RESET state cheaply (no rebuild, no extra forward pass): copy the
        # prefill KV back into the persistent KV tensors + zero counters. This RESET matters for
        # correctness across repeated decode()s, each must start from the clean prefill KV (positions
        # 0..base_pos-1), uncontaminated by the previous decode's K in-kernel appends, and keeps the
        # measured single-launch cost the kernel itself, not a redundant baseline pass.
        ins = {
            TOKEN_NAME: torch.tensor([first_token], dtype=torch.int32, device=self.device),
            POS_NAME: torch.tensor([base_pos], dtype=torch.int32, device=self.device),
            "reshape_id0": torch.tensor([0], dtype=torch.int32, device=self.device),
        }
        first_build = getattr(self.base_vm, "_built_fp", None) is None
        if first_build:
            self.base_vm.run(ins, kv=kv or {})
        else:
            # cheap reset: restore prefill KV values into the persistent KV device tensors.
            if kv:
                for name, t in self.base_vm._kv_tensors.items():
                    if name in kv:
                        t.copy_(kv[name])

        la = self.base_vm._launch_args
        # _launch_args layout (see vm/loader.py _run_full_build step 9):
        #   0 d_buffers.ptr, 1 nb, 2 counters.ptr, 3 nc, 4 d_instr.ptr, 5 ni,
        #   6 d_qflat.ptr, 7 d_qoff.ptr, 8 d_qlen.ptr, 9 grid_dim,
        #   10 arena.ptr, 11 arena_bytes, 12 d_abort.ptr,
        #   13 grid_dim, 14 threads_per_block, 15 dyn_smem_bytes
        d_buffers_ptr = la[0]
        nb = la[1]
        counters_ptr = la[2]
        nc = la[3]
        d_instr_ptr = la[4]
        ni = la[5]
        d_qflat_ptr = la[6]
        d_qoff_ptr = la[7]
        d_qlen_ptr = la[8]
        num_sms = self.base_vm.num_sms
        arena_ptr = la[10]
        arena_bytes = la[11]
        d_abort_ptr = la[12]
        grid_dim = la[13]
        threads_per_block = la[14]
        dyn_smem_bytes = la[15]

        # Device pointers of the IO cells the in-kernel loop writes/reads.
        pos_cell_ptr = self.base_vm._buf_tensor[self._pos_bid].data_ptr()
        token_cell_ptr = self.base_vm._buf_tensor[self._token_bid].data_ptr()

        # Output buffer for the K sampled token ids.
        generated = torch.full((n_tokens,), -1, dtype=torch.int32, device=self.device)

        # Occupancy gate for the PERSISTENT kernel (it may have a different register frame than the
        # baseline). Cap grid_dim to the persistent kernel's co-residency limit.
        max_grid = int(self.ext.persist_max_coresident_blocks(threads_per_block, dyn_smem_bytes))
        if max_grid <= 0:
            raise RuntimeError("AMK persist: kernel reports zero occupancy at this launch config")
        pgrid = min(grid_dim, max_grid)
        # The work was assigned across [0, grid_dim) by the baseline; if the persistent kernel has
        # lower occupancy we must keep the SAME assignment reachable. Rebuild queues capped to pgrid
        # only if needed (rare). For the toy/small decode the frame is comparable, so pgrid==grid.
        if pgrid < grid_dim:
            # Re-assign the baseline VM to the tighter grid and rebuild tables so no queue is
            # stranded beyond the launchable grid.
            self.base_vm.grid_dim = pgrid
            self.base_vm._assign_sms()
            self.base_vm._built_fp = None
            self.base_vm.run(ins, kv=kv or {})
            la = self.base_vm._launch_args
            (d_buffers_ptr, nb, counters_ptr, nc, d_instr_ptr, ni,
             d_qflat_ptr, d_qoff_ptr, d_qlen_ptr, _g9,
             arena_ptr, arena_bytes, d_abort_ptr,
             grid_dim, threads_per_block, dyn_smem_bytes) = la
            pos_cell_ptr = self.base_vm._buf_tensor[self._pos_bid].data_ptr()
            token_cell_ptr = self.base_vm._buf_tensor[self._token_bid].data_ptr()

        # Zero counters + abort flag before the launch (one launch == K passes; the kernel re-zeroes
        # between steps, but step 0 needs a clean slate, the entry barrier confirms visibility).
        self.base_vm._counters.zero_()
        self.base_vm._abort.zero_()
        torch.cuda.synchronize()

        status = self.ext.persist_launch(
            d_buffers_ptr, nb,
            counters_ptr, nc,
            d_instr_ptr, ni,
            d_qflat_ptr, d_qoff_ptr, d_qlen_ptr,
            num_sms,
            arena_ptr, arena_bytes,
            d_abort_ptr,
            pgrid, threads_per_block, dyn_smem_bytes,
            int(n_tokens), int(base_pos),
            pos_cell_ptr, token_cell_ptr,
            d_buffers_ptr, int(self._logits_bid),
            generated.data_ptr(), int(first_token),
        )
        self.last_status = status
        if status["status"] == "TIMEOUT":
            raise TimeoutError("AMK persistent launch hit WDDM TDR: " + status["error"])
        if status["status"] != "OK":
            raise RuntimeError("AMK persistent launch failed: " + status["error"])
        torch.cuda.synchronize()

        tokens = generated.tolist()
        if any(t < 0 for t in tokens):
            raise RuntimeError(f"AMK persist: kernel did not fill all {n_tokens} tokens: {tokens}")
        return {"tokens": tokens, "status": status}


__all__ = ["PersistentDecodeVM"]

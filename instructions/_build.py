"""
AMK Layer-1, CUDA build helper (Windows / sm_120 aware).
=========================================================

Centralises everything fragile about JIT-compiling the ABI-conformant micro-kernels in
``instructions/cuda/*.cu`` with ``torch.utils.cpp_extension.load`` on this machine:

  * **MSVC discovery.** ``torch``'s ninja build shells out to ``where cl`` and fails on a clean
    Windows box where Visual Studio's ``cl.exe`` is not on ``PATH``. We locate ``vcvars64.bat``
    (BuildTools 2022 here), source it once, and merge the resulting INCLUDE/LIB/PATH into this
    process so the compiler is found. No-op on Linux.
  * **sm_120 codegen.** We force ``TORCH_CUDA_ARCH_LIST="12.0"`` and pass
    ``-gencode=arch=compute_120,code=sm_120`` so the kernels are native Blackwell-laptop SASS
    (with PTX kept for forward-compat). These are the exact flags the task pins.
  * **Build caching.** Each ``.cu`` is compiled once per process and memoised by name, so the
    verifier / generation loop can request a kernel module repeatedly without re-running nvcc.

The micro-kernels never include this file; it only orchestrates their compilation. The kernels
themselves are plain CUDA that conform to ``vm/abi.h`` (a ``__device__`` core + a thin torch
wrapper), so the same ``.cu`` is reusable inside the megakernel VM later.
"""
from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

CUDA_DIR = Path(__file__).resolve().parent / "cuda"
_REPO_ROOT = Path(__file__).resolve().parent.parent

# RETARGETING SURFACE: derive the arch from the live device, never hardcode (sm_120 RTX 5090,
# sm_90 H100, sm_80 A100, sm_100 B200 all fall out of this). PTX kept for forward compatibility.
def _device_arch() -> tuple[str, str]:
    """(arch_str, cc) from the live CUDA device. Falls back to sm_120 (local dev) if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            maj, minr = torch.cuda.get_device_capability()
            return f"{maj}.{minr}", f"{maj}{minr}"
    except Exception:
        pass
    return "12.0", "120"


def _gencode(cc: str) -> list[str]:
    return [f"-gencode=arch=compute_{cc},code=sm_{cc}",
            f"-gencode=arch=compute_{cc},code=compute_{cc}"]

# Candidate vcvars locations (BuildTools / Community / Pro / Enterprise, 2022 & 2019).
_VCVARS_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
]

_MSVC_READY = False


def _cl_on_path() -> bool:
    return any((Path(p) / "cl.exe").exists()
               for p in os.environ.get("PATH", "").split(os.pathsep) if p)


def ensure_msvc_env() -> None:
    """Make MSVC ``cl.exe`` discoverable for torch's ninja build on Windows.

    Sources ``vcvars64.bat`` in a child ``cmd`` and merges its environment into this process.
    Idempotent and a no-op on non-Windows / when ``cl`` is already on ``PATH``."""
    global _MSVC_READY
    if _MSVC_READY or sys.platform != "win32" or _cl_on_path():
        _MSVC_READY = True
        return
    vcvars = next((c for c in _VCVARS_CANDIDATES if os.path.exists(c)), None)
    if vcvars is None:
        # Leave it to torch; it may still find an env-configured compiler. Surface a hint.
        print("[amk._build] WARNING: no vcvars64.bat found; MSVC may be missing. "
              "Open a 'x64 Native Tools' prompt or install VS BuildTools.", file=sys.stderr)
        return
    # Nested quoting: outer pair for `cmd /c`, inner pair for the spaced vcvars path.
    cmd = f'cmd /c ""{vcvars}" x64 && set"'
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[amk._build] WARNING: vcvars64 failed (rc={res.returncode}):\n{res.stderr[-400:]}",
              file=sys.stderr)
        return
    for line in res.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            # Windows env is case-insensitive; normalise the search-path key to 'PATH'.
            os.environ["PATH" if k.upper() == "PATH" else k] = v
    _MSVC_READY = True


def cuda_cflags(extra: list[str] | None = None) -> list[str]:
    """Standard nvcc flags for a micro-kernel build, gencode matched to the live device."""
    _, cc = _device_arch()
    flags = _gencode(cc) + [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "--expt-relaxed-constexpr",
    ]
    if sys.platform == "win32":
        # torch's compiled_autograd.h trips MSVC's *traditional* preprocessor ("'std' ambiguous").
        # Force the standard-conforming preprocessor (the warning nvcc itself prints) and silence
        # the CCCL traditional-preprocessor notice.
        flags += ["-Xcompiler", "/Zc:preprocessor",
                  "-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"]
    if extra:
        flags += extra
    return flags


def _variant_tag(extra_cuda_cflags: tuple) -> str:
    """Stable short tag for a set of -D variant flags, so distinct variants get distinct extension
    names + build dirs (torch can't reload one extension name with different code in a process)."""
    if not extra_cuda_cflags:
        return ""
    import hashlib
    h = hashlib.sha1("|".join(extra_cuda_cflags).encode()).hexdigest()[:8]
    return "_" + h


@lru_cache(maxsize=None)
def load_kernel(name: str, *, verbose: bool = False, extra_cuda_cflags: tuple = ()):
    """JIT-build and return the torch extension module for ``instructions/cuda/<name>.cu``.

    Memoised per process by (name, flags). Sets up MSVC + sm_120 flags. Variant flag sets get a
    distinct extension name + build directory so the generation loop can compare rebuilt variants
    in one process. Raises a clear error if the source is missing so the verifier can report it."""
    arch, cc = _device_arch()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)
    ensure_msvc_env()
    src = CUDA_DIR / f"{name}.cu"
    if not src.exists():
        raise FileNotFoundError(f"CUDA source not found: {src}")
    # Import here so a torch-less environment can still import this module's helpers.
    from torch.utils.cpp_extension import load
    tag = f"_sm{cc}" + _variant_tag(extra_cuda_cflags)  # per-arch build dir (no stale SASS reuse)
    build_dir = _REPO_ROOT / ".amk_build" / f"{name}{tag}"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=f"amk_{name}{tag}",
        sources=[str(src)],
        extra_include_paths=[str(_REPO_ROOT / "vm")],  # amk_abi.h available to kernels
        extra_cuda_cflags=cuda_cflags(list(extra_cuda_cflags)),
        extra_cflags=["/O2"] if sys.platform == "win32" else ["-O3"],
        build_directory=str(build_dir),
        verbose=verbose,
    )

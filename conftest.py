"""
Pytest configuration: make the suite runnable ANYWHERE.

The correctness-bearing core (IR, validator, CPU reference VM, lowering, HF import, eval, search,
compile) is GPU-free and runs on any machine / in CI. The CUDA tests (test_cuda_*.py) require a
GPU + nvcc; they are auto-skipped when CUDA is unavailable rather than failing, so `pytest`
is green on a laptop, a CI runner, and a GPU box alike (on a GPU box they all run).
"""
from __future__ import annotations

import pytest

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False


def pytest_collection_modifyitems(config, items):
    if _HAS_CUDA:
        return
    skip_cuda = pytest.mark.skip(reason="no CUDA device (GPU-only test)")
    for item in items:
        # test_cuda_*.py are the GPU megakernel/instruction tests.
        if "test_cuda" in item.nodeid:
            item.add_marker(skip_cuda)

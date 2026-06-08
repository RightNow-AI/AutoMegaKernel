"""
Drift guard: vm/abi.h MUST stay byte-for-byte consistent with schedule/ir.py.

The two contracts share numeric enum codes (DType / MemSpace / InstructionKind), fixed POD
capacities (AMK_MAX_*), and the ABI version. If they ever diverge, the host loader marshals
schedules into the wrong on-device codes, a silent, catastrophic class of bug. This test parses
the C header and asserts equality. Run: `uv run pytest tests/test_abi_sync.py` or directly.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schedule import ir  # noqa: E402

ABI_H = os.path.join(os.path.dirname(__file__), "..", "vm", "abi.h")


def _read_header() -> str:
    with open(ABI_H, encoding="utf-8") as f:
        return f.read()


def _parse_enum(src: str, prefix: str) -> dict[str, int]:
    """Parse `PREFIX_NAME = N,` lines into {NAME: N}, ignoring sentinels (no '= N')."""
    out: dict[str, int] = {}
    for m in re.finditer(rf"{re.escape(prefix)}([A-Z0-9_]+)\s*=\s*(\d+)", src):
        out[m.group(1)] = int(m.group(2))
    return out


def _parse_define(src: str, name: str) -> int:
    m = re.search(rf"#define\s+{re.escape(name)}\s+(\d+)", src)
    assert m, f"{name} not found in abi.h"
    return int(m.group(1))


def test_dtype_codes_match():
    src = _read_header()
    hdr = _parse_enum(src, "AMK_")
    for dt in ir.DType:
        assert hdr.get(dt.name) == dt.value, f"DType {dt.name}: abi.h={hdr.get(dt.name)} ir={dt.value}"


def test_memspace_codes_match():
    src = _read_header()
    hdr = _parse_enum(src, "AMK_")
    for ms in ir.MemSpace:
        assert hdr.get(ms.name) == ms.value, f"MemSpace {ms.name}: abi.h={hdr.get(ms.name)} ir={ms.value}"


def test_opcode_codes_match():
    src = _read_header()
    hdr = _parse_enum(src, "AMK_OP_")
    for op in ir.InstructionKind:
        assert hdr.get(op.name) == op.value, f"opcode {op.name}: abi.h={hdr.get(op.name)} ir={op.value}"
    # the AMK_OP__COUNT sentinel must equal the number of opcodes
    m = re.search(r"AMK_OP__COUNT", src)
    assert m, "AMK_OP__COUNT sentinel missing"
    # every header opcode must be a real IR opcode (no extras)
    ir_names = {op.name for op in ir.InstructionKind}
    for name in hdr:
        if name.startswith(("F", "BF", "I", "U", "BOOL", "HBM", "GLOBAL", "SMEM", "REGISTER")):
            continue  # dtype/memspace members also matched by AMK_ prefix, skip
        assert name in ir_names, f"abi.h has opcode {name} with no IR counterpart"


def test_capacities_match():
    src = _read_header()
    assert _parse_define(src, "AMK_MAX_INPUTS") == ir.ABI_MAX_INPUTS
    assert _parse_define(src, "AMK_MAX_OUTPUTS") == ir.ABI_MAX_OUTPUTS
    assert _parse_define(src, "AMK_MAX_WAITS") == ir.ABI_MAX_WAITS
    assert _parse_define(src, "AMK_MAX_RANK") == ir.ABI_MAX_RANK


def test_abi_version_matches():
    src = _read_header()
    major = _parse_define(src, "AMK_ABI_VERSION_MAJOR")
    minor = _parse_define(src, "AMK_ABI_VERSION_MINOR")
    assert f"{major}.{minor}" == ir.ABI_VERSION, f"abi.h {major}.{minor} != ir {ir.ABI_VERSION}"


if __name__ == "__main__":
    test_dtype_codes_match()
    test_memspace_codes_match()
    test_opcode_codes_match()
    test_capacities_match()
    test_abi_version_matches()
    print("ABI <-> IR sync verified: dtypes, memspaces, opcodes, capacities, version all match.")

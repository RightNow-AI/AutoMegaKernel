"""Tests for the public FUSED instruction substrate (recipe interpreter + structural validator)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
import torch  # noqa: E402

from instructions.reference import (  # noqa: E402
    REFERENCE, RefCtx, ref_add, ref_fused, ref_rmsnorm, validate_recipe,
)
from schedule.ir import InstructionKind, OP_REGISTRY  # noqa: E402

ADD, RMS, FUSED = int(InstructionKind.ADD), int(InstructionKind.RMSNORM), int(InstructionKind.FUSED)


def test_fused_registered():
    assert int(InstructionKind.FUSED) == 19
    assert OP_REGISTRY[InstructionKind.FUSED].max_inputs == -1
    assert InstructionKind.FUSED in REFERENCE


def test_ref_fused_bit_identical_to_unfused():
    torch.manual_seed(0)
    h = 48
    x, res, w = torch.randn(1, h), torch.randn(1, h), torch.randn(h)
    ctx = RefCtx()
    add_out = torch.empty_like(x)
    ref_add([x, res], [add_out], {}, ctx)
    unfused = torch.empty_like(x)
    ref_rmsnorm([add_out, w], [unfused], {"eps": 1e-5, "hidden": h}, ctx)

    recipe = {"steps": [
        {"op": ADD, "args": [{"in": 0}, {"in": 1}], "out": 0, "out_shape": [1, h], "out_dtype": "float32"},
        {"op": RMS, "args": [{"s": 0}, {"in": 2}], "out": "final", "params": {"eps": 1e-5, "hidden": h}},
    ]}
    fused = torch.empty_like(x)
    ref_fused([x, res, w], [fused], {"recipe": recipe}, ctx)
    assert torch.equal(fused, unfused)


def test_validate_recipe_accepts_wellformed():
    good = {"steps": [{"op": ADD, "args": [{"in": 0}, {"in": 1}], "out": "final"}]}
    assert validate_recipe(good, 2) == []


@pytest.mark.parametrize("recipe", [
    {"steps": [{"op": ADD, "args": [{"in": 5}], "out": "final"}]},          # input index out of range
    {"steps": [{"op": ADD, "args": [{"s": 1}], "out": "final"}]},           # reads unproduced scratch
    {"steps": [{"op": ADD, "args": [{"in": 0}], "out": 0}]},                # never writes 'final'
    {"steps": [{"op": FUSED, "args": [], "out": "final"}]},                 # nested FUSED
    {"steps": [{"op": ADD, "args": [{"in": 0}], "out": 0, "out_dtype": "evil"}, {"op": ADD, "args": [{"s": 0}], "out": "final"}]},  # dtype not in allowlist
    {"steps": []},                                                         # empty
])
def test_validate_recipe_rejects_malformed(recipe):
    assert validate_recipe(recipe, 2), f"expected rejection for {recipe}"


def test_ref_fused_raises_on_malformed():
    ctx = RefCtx()
    bad = {"steps": [{"op": ADD, "args": [{"in": 9}], "out": "final"}]}
    with pytest.raises(ValueError):
        ref_fused([torch.zeros(1, 4)], [torch.zeros(1, 4)], {"recipe": bad}, ctx)


def test_recipe_json_round_trips():
    import json
    recipe = {"steps": [
        {"op": ADD, "args": [{"in": 0}, {"in": 1}], "out": 0, "out_shape": [1, 8], "out_dtype": "float32"},
        {"op": RMS, "args": [{"s": 0}, {"in": 2}], "out": "final", "params": {"eps": 1e-5, "hidden": 8}},
    ]}
    assert json.loads(json.dumps(recipe)) == recipe

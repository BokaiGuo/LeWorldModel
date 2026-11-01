import copy

import pytest

from experiments.spectral_reachability import e3_cell11_saved_receipt_20261031_main as receipt


def sample():
    boundary = {"completed": True, "pending": None,
                "counts": {k: {"attempted": 1, "returned": 1} for k in ("encoder", "projector")}}
    return {
        "binding_chunks_complete": True, "full_saved_strict": False, "scientific_admission": False,
        "base": {"status": "completed20",
                 "saved": {"physical": {"control_steps": 20, "warm_steps": 10, "frames": 32, "stopped": False}},
                 "encoding_ledger": {"boundaries": [dict(boundary, boundary=10), dict(boundary, boundary=15)]},
                 "partial_native": {"terminal_row_verified": True, "unmatched_frame_count": 0}},
    }


def test_compact_contract_and_limits():
    result = receipt.compact(sample())
    assert result["status"] == "completed20"
    assert not result["scientific_admission"]
    assert not result["equal_budget_certified"]


@pytest.mark.parametrize("change", ["partial", "stop", "missing_frame", "boundary"])
def test_reject_incomplete_or_unbound(change):
    value = copy.deepcopy(sample())
    if change == "partial":
        value["base"]["status"] = "partial"
    elif change == "stop":
        value["base"]["saved"]["physical"]["stopped"] = True
    elif change == "missing_frame":
        value["base"]["partial_native"]["unmatched_frame_count"] = 1
    else:
        value["base"]["encoding_ledger"]["boundaries"][1]["completed"] = False
    with pytest.raises(Exception):
        receipt.compact(value)

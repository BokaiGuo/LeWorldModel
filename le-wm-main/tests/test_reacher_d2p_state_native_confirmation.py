"""Admission fixtures for the prospective CUDA confirmation driver."""
import copy

import pytest

from experiments.spectral_reachability.run_reacher_d2p_state_native_confirmation import validate_alignment


def fixture():
    return {"schema": "reacher_d2p_state_native_selector_audit.v1", "status": "completed",
        "gates": dict.fromkeys(("cuda_only", "weights_unchanged", "all_declared_roots", "all17_words_every_root", "all_core_equivalence_gates",
            "all_physical_gates", "prefix_seal_unchanged", "suffix_seal_unchanged", "reference_unchanged", "implementation_unchanged"), True),
        "sealed_protocol": {"root_seeds": list(range(8))}, "rows": [{"seed": s} for s in range(8)]}


def test_complete_alignment_required():
    report = fixture()
    assert validate_alignment(report) == report["sealed_protocol"]


@pytest.mark.parametrize("kind", ["missing_gate", "truthy", "status", "schema", "partial", "duplicate", "wrong_root"])
def test_rejects_partial_or_invalid_alignment(kind):
    report = fixture()
    if kind == "missing_gate":
        del report["gates"]["all_core_equivalence_gates"]
    elif kind == "truthy":
        report["gates"]["cuda_only"] = 1
    elif kind == "status":
        report["status"] = "running"
    elif kind == "schema":
        report["schema"] = "other.v1"
    elif kind == "partial":
        report["rows"].pop()
    elif kind == "duplicate":
        report["rows"][0] = copy.deepcopy(report["rows"][1])
    else:
        report["rows"][0]["seed"] = 99
    with pytest.raises(ValueError):
        validate_alignment(report)

from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import (
    fixed_prefix_gate, validate_reference, suffix_conditions,
)


def reference():
    return {"schema": "reacher_d2p_cem_incumbent_audit.v1", "status": "completed",
            "gates": {"integrity": True}, "sealed_protocol": {"root_seeds": [2, 1]},
            "rows": [{"seed": 1}, {"seed": 2}]}


def test_identity_pairing_accepts_different_reference_order():
    assert [row["seed"] for row in validate_reference(reference())] == [1, 2]


@pytest.mark.parametrize("change", ["duplicate", "missing", "extra", "failed", "empty_gates"])
def test_reference_fails_closed(change):
    report = deepcopy(reference())
    if change == "duplicate":
        report["rows"][1]["seed"] = 1
    elif change == "missing":
        report["rows"].pop()
    elif change == "extra":
        report["rows"].append({"seed": 3})
    elif change == "failed":
        report["gates"]["integrity"] = False
    else:
        report["gates"] = {}
    with pytest.raises(ValueError):
        validate_reference(report)


def test_only_cem_prefixes_have_stored_tail_control():
    assert suffix_conditions("legacy_cem") == ("shared_dls", "stored_cem_tail", "observed_cem")
    assert suffix_conditions("global_best_cem") == suffix_conditions("legacy_cem")
    assert suffix_conditions("dispatch") == ("shared_dls", "observed_cem")
    assert suffix_conditions("same_bank_greedy") == suffix_conditions("dispatch")
    with pytest.raises(ValueError):
        suffix_conditions("unknown")


def test_prefix_gate_checks_steps_values_and_length():
    prefix = np.arange(10, dtype=np.float32).reshape(5, 2)
    actions = np.concatenate((prefix, np.zeros((10, 2), np.float32)))
    assert fixed_prefix_gate(actions, prefix)
    changed = actions.copy()
    changed[4, 1] += 1
    assert not fixed_prefix_gate(changed, prefix)
    assert not fixed_prefix_gate(actions[:-1], prefix)
    assert not fixed_prefix_gate(actions, prefix[:-1])

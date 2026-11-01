"""CEM provenance schema regressions; no model invocation."""
import json

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_goal_joint_limits import load_cem_choices
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest


def choice(seed=1):
    return {"seed": seed, "cem_params": [], "cem_receipt": {}, "root_rgb_sha256": "r", "goal_rgb_sha256": "g"}


def test_fresh_choice_is_read_directly():
    report = {"schema": "reacher_d2p_state_native_confirmation.v1", "sealed_protocol": {"root_seeds": [1]}}
    assert load_cem_choices(report, {"rows": [choice()]}) == {1: choice()}


def test_known_audit_follows_hash_bound_parent(tmp_path):
    path = tmp_path / "parent.json"
    seal = path.with_suffix(".selection.json")
    seal.write_text(json.dumps({"rows": [choice()]}))
    path.write_text(json.dumps({"status": "completed", "gates": {"exact": True}, "selection_sha256": digest(seal)}))
    report = {"schema": "reacher_d2p_state_native_selector_audit.v1", "sealed_protocol": {
        "root_seeds": [1], "reference_path": str(path), "reference_sha256": digest(path)}}
    assert load_cem_choices(report, {"rows": [{"seed": 1}]}) == {1: choice()}
    seal.write_text(json.dumps({"rows": [choice(2)]}))
    with pytest.raises(ValueError):
        load_cem_choices(report, {"rows": [{"seed": 1}]})


@pytest.mark.parametrize("bad", ["missing", "duplicate", "wrong_root"])
def test_incomplete_or_duplicate_cem_choices_fail_closed(bad):
    report = {"schema": "reacher_d2p_state_native_confirmation.v1", "sealed_protocol": {"root_seeds": [1]}}
    rows = [{"seed": 1}] if bad == "missing" else [choice(), choice()] if bad == "duplicate" else [choice(2)]
    with pytest.raises(ValueError):
        load_cem_choices(report, {"rows": rows})

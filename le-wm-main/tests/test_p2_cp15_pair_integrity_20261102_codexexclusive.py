from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import p2_cp15_pair_integrity_20261102_codexexclusive as audit


def test_real_saved_pair_and_root_aliases():
    result = audit.audit_saved()
    assert result["slots"] == 4
    assert result["distinct_known_roots"] == 2
    assert result["distinct_paired_contexts"] == 3
    assert result["duplicate_bases"] == [[2, 3]]
    assert result["physical_plan_suffix_matches_winner"] is True
    assert result["raw_steps_field_matches_saved_commands"] is False
    assert result["P1_admission"] is False and result["P2_admission"] is False
    assert result["equal_total_compute"] is False


@pytest.fixture
def inputs():
    joined = json.loads(audit.reuse.PHYSICAL_JOIN.read_bytes())
    branches = [json.loads((audit.NEW_RUN / f"branch_{i}.json").read_bytes()) for i in range(8)]
    return joined, branches


@pytest.mark.parametrize("damage", ["suffix", "context", "feedback", "goal", "prefix", "budget"])
def test_detects_fair_pair_binding_drift(inputs, damage):
    joined, branches = deepcopy(inputs)
    row = joined["policy_slots"][0]["row"]
    if damage == "suffix":
        raw = row["plan"]["raw_float32_hex"]
        row["plan"]["raw_float32_hex"] = raw[:-2] + ("00" if raw[-2:] != "00" else "ff")
    elif damage == "context":
        row["plan"]["new_model_plan"]["input_context"]["latent_history"][0][0] += 1
    elif damage == "feedback":
        row["feedback_guard"]["compile_input_context_sha256"] = "0" * 64
    elif damage == "goal":
        row["goal_pixel_sha256"] = "0" * 64
    elif damage == "prefix":
        row["feedback_guard"]["prefix_raw15_bits"] = "0" * 240
    else:
        row["own_returned"] = 24
    with pytest.raises(ValueError):
        audit.inspect_pairs(joined, branches)

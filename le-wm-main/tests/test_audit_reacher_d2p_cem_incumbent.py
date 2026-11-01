from copy import deepcopy

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_cem_incumbent import legacy_replay_gates


def method():
    return {"first_receipt": {"trace": [1., .5]}, "action_witness_raw": [[1., 0.]],
        "observed_first_chunk": {"candidate_goal_cost": .2}, "physical": {"final_qpos_distance": .3},
        "execution_identity": {"initial": "same"}}


def test_all_legacy_replay_fields_must_be_exact():
    assert all(legacy_replay_gates(method(), method()).values())


@pytest.mark.parametrize("field", ["first_receipt", "action_witness_raw", "observed_first_chunk", "physical", "execution_identity"])
def test_replay_rejects_each_kind_of_drift(field):
    original, changed = method(), deepcopy(method())
    if field == "first_receipt":
        changed[field]["trace"][0] += 1e-14
    elif field == "action_witness_raw":
        changed[field][0][0] += 1e-14
    elif field == "observed_first_chunk":
        changed[field]["candidate_goal_cost"] += 1e-14
    elif field == "physical":
        changed[field]["final_qpos_distance"] += 1e-14
    else:
        changed[field]["initial"] = "different"
    assert not all(legacy_replay_gates(changed, original).values())

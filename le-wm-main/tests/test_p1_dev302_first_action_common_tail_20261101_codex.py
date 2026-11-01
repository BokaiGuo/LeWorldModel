import numpy as np

from experiments.spectral_reachability import p1_dev302_first_action_common_tail_20261101_codex as subject


def test_holdout_contract_freezes_same_intervention_before_physics():
    value = subject.build()
    rows = {row["role"]: np.asarray(row["actions"], np.float32) for row in value["roles"]}
    assert np.array_equal(rows["d2p_first"][1:], rows["zero_first"][1:])
    assert np.array_equal(rows["d2p_first"][1:], rows["gn_first"][1:])
    assert value["rule_frozen_from_dev301_before_dev302_physics"] is True
    assert value["dev302_physical_outcomes_used_for_selection"] is False
    assert value["training_membership"] == "unknown"
    assert value["independent_confirmation"] is False
    assert value["P1_admission"] is False

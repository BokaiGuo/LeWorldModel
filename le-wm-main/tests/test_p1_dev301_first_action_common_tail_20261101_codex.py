import numpy as np

from experiments.spectral_reachability import p1_dev301_first_action_common_tail_20261101_codex as subject


def test_contract_has_one_action_intervention_and_exact_common_tail():
    value = subject.build()
    rows = {row["role"]: np.asarray(row["actions"], np.float32) for row in value["roles"]}
    assert list(rows) == list(subject.ROLES)
    assert np.array_equal(rows["d2p_first"][1:], rows["zero_first"][1:])
    assert np.array_equal(rows["d2p_first"][1:], rows["gn_first"][1:])
    assert not np.array_equal(rows["d2p_first"][0], rows["zero_first"][0])
    assert not np.array_equal(rows["d2p_first"][0], rows["gn_first"][0])
    assert value["physical_outcomes_used_to_design_diagnostic"] is True
    assert value["physical_outcomes_used_to_choose_within_run"] is False
    assert value["independent_confirmation"] is False
    assert value["P1_admission"] is False

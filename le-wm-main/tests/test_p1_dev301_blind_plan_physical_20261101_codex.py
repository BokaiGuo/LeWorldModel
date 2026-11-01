from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev301_blind_plan_physical_20261101_codex as subject


def test_build_binds_three_fixed_arms_and_saved_sources():
    value = subject.build()
    assert [row["arm"] for row in value["arms"]] == list(subject.ARMS)
    assert value["development_only"] is True
    assert value["equal_budget_claim"] is False
    assert value["independent_confirmation"] is False
    assert value["P1_admission"] is False
    assert value["source"]["training_membership"] == "unknown"
    assert value["source"]["historical_coverage"] == "unknown"
    assert value["planner"]["model_terminal_cost"]["line_search_GN"] < value["planner"]["model_terminal_cost"]["full10_conditional_D2P"]
    assert np.array_equal(np.asarray(value["arms"][2]["actions"], np.float32), np.zeros((10, 2), np.float32))


@pytest.mark.parametrize("defect", ["shape", "nan", "range", "precision"])
def test_action_array_rejects_noncontract_actions(defect):
    value = np.zeros((10, 2), np.float32)
    if defect == "shape":
        value = value[:9]
    elif defect == "nan":
        value[0, 0] = np.nan
    elif defect == "range":
        value[0, 0] = 1.1
    else:
        value = value.astype(np.float64)
        value[0, 0] = float(np.float64(np.float32(0.1)) + np.float64(1e-12))
    with pytest.raises(Exception):
        subject.action_array(value)


def test_contract_does_not_promote_physical_or_scientific_claims():
    value = deepcopy(subject.build())
    assert value["execution"]["physical_outcomes_used_for_selection"] is False
    assert value["execution"]["new_neural_calls"] == 0
    assert value["planner"]["equal_budget_claim"] is False

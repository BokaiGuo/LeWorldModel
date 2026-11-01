"""Saved-array arithmetic fixtures, not CUDA or physical efficacy evidence."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability import geometry
from experiments.spectral_reachability.audit_reacher_d2p_mechanism_metric_contract import compare_role


def cells():
    result = {}
    for i, (j, own_residual, stage, terminal) in enumerate(((1.,20.,1.,1.), (1.,20.,.5,.5), (.5,.1,1.1,.1))):
        common, own = geometry([[j]],[1.]), geometry([[j]],[own_residual])
        result[str(i)] = {"stage":stage,"terminal":terminal,"initial_stage":1.2,
            "fits":[{"scale":scale,"own_objective_mean":own["regularized_objective"],
                "common_gain_fraction":common["gain_fraction"]} for scale in (.1,.25)]}
    return result


def roles():
    return {"zero":0,"greedy":1,"native_candidate":2}


def test_own_value_and_common_gain_are_different_estimands():
    result = compare_role(cells(),roles(),"native_candidate")
    assert result["checks"]["own_state_value_better_at_both_scales"] is True
    assert result["checks"]["common_gain_better_at_both_scales"] is False
    assert result["stage_plus_own_reach_conjunction"] is True


def test_zero_prefix_and_initial_regression_are_not_interchangeable():
    result = compare_role(cells(),roles(),"native_candidate")
    assert result["checks"]["same_space_stage_regression_vs_zero"] is True
    assert result["initial_state_stage_regression"] is False
    data = cells()
    data["0"]["stage"] = 1.2
    for value in data.values(): value["initial_stage"] = 1.
    result = compare_role(data,roles(),"native_candidate")
    assert result["checks"]["same_space_stage_regression_vs_zero"] is False
    assert result["initial_state_stage_regression"] is True


def test_missing_initial_rgb_anchor_remains_unknown_not_passed():
    data = cells()
    for value in data.values(): value["initial_stage"] = None
    result = compare_role(data,roles(),"native_candidate")
    assert result["initial_state_stage_regression"] is None
    assert result["zero_prefix_stage_minus_initial"] is None


def test_greedy_fallback_never_has_strict_own_benefit_over_itself():
    mapping = {**roles(),"native_candidate":1}
    result = compare_role(cells(),mapping,"native_candidate")
    assert result["checks"]["own_state_value_better_at_both_scales"] is False
    assert result["stage_plus_own_reach_conjunction"] is False


@pytest.mark.parametrize("defect",["missing_scale","wrong_scale","nan","negative_stage"])
def test_invalid_or_incomplete_comparisons_rejected(defect):
    data = copy.deepcopy(cells())
    if defect == "missing_scale": data["2"]["fits"].pop()
    elif defect == "wrong_scale": data["2"]["fits"][0]["scale"] = .05
    elif defect == "nan": data["2"]["fits"][0]["own_objective_mean"] = np.nan
    else: data["2"]["stage"] = -1.
    with pytest.raises(ValueError): compare_role(data,roles(),"native_candidate")

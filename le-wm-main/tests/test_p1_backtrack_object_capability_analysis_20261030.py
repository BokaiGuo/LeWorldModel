import math

import pytest

from experiments.spectral_reachability.p1_backtrack_object_capability_analysis_20261030 import measure, paired


def test_separate_physical_units_and_angle_wrap():
    anchor = dict(index=0, root_slot=0, condition="off", arm="a", role="incumbent",
                  root_state=[0, 0, 0, 0, 0, 0, 0], goal=[0, 0, 1, 0, 1, 0, 0])
    base = [0, 0, 10, 20, math.pi - 0.01, 0, 0]
    endpoints = [base] + [base.copy() for _ in range(20)]
    endpoints[1] = [0, 0, 11, 20, -math.pi + 0.09, 0, 0]
    endpoints[2] = [0, 0, 9, 20, math.pi - 0.11, 0, 0]
    result = measure(anchor, endpoints)
    assert result["position_jacobian"][0] == [4.0, 0.0]
    assert result["angle_jacobian"][0] == pytest.approx(0.4)
    assert result["position_singular_values"] == pytest.approx([4.0, 0.0])
    assert result["position_goal_response"][0] == pytest.approx(4.)
    assert result["angle_goal_response"][0] == pytest.approx(0.4)
    assert result["zero_relative_position_error_gains"][0] == 0.
    assert result["fixed_bank_best_position_gain"] == pytest.approx(
        max(result["zero_relative_position_error_gains"]))


def test_missing_probes_not_imputed():
    anchor = dict(index=0, root_slot=0, condition="off", arm="a", role="incumbent",
                  root_state=[0.] * 7, goal=[0.] * 7)
    endpoints = [[0.] * 7 for _ in range(21)]
    endpoints[1] = None
    result = measure(anchor, endpoints)
    assert result["status"] == "incomplete"
    assert result["position_jacobian"][0] is None
    assert result["position_singular_values"] is None
    assert result["fixed_bank_best_position_gain"] is None
    endpoints[0] = None
    assert measure(anchor, endpoints)["status"] == "zero_endpoint_missing"


def test_pairs_keep_order_and_missing_values():
    rows = [dict(anchor=i, root_slot=i // 8, condition=("off", "on")[(i // 4) % 2],
                 arm=("a", "b")[(i // 2) % 2], role=("incumbent", "model_selected")[i % 2],
                 fixed_bank_best_position_gain=float(i), fixed_bank_best_angle_gain=None,
                 zero_relative_position_error_gains=[float(i), None],
                 zero_relative_angle_error_gains=[0.]) for i in range(16)]
    pair = paired(rows)[0]
    assert pair["selected_minus_incumbent_fixed_bank_best_position_gain"] == 1.
    assert pair["selected_minus_incumbent_fixed_bank_best_angle_gain"] is None
    assert pair["selected_minus_incumbent_zero_relative_position_error_gains"] == [1., None]
    rows[1]["role"] = "incumbent"
    with pytest.raises(ValueError, match="identity"):
        paired(rows)

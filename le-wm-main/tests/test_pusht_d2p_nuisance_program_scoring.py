"""Analytic array fixtures, not PushT efficacy or mechanism evidence."""
import copy
import json

import numpy as np
import pytest

from experiments.spectral_reachability.pusht_d2p_goal_nuisance_metric import compile_metric
from experiments.spectral_reachability.pusht_d2p_nuisance_program_scoring import (
    compile_candidates, score_terminal_predictions,
)


def bank():
    # Coordinate zero is nuisance; task coordinate one remains observable.
    return dict(goal_vectors=[[0., 0.], [4., 0.]],
                candidate_ids=["zero", "reconfigure"],
                prefix_params=np.array([[[0., 0.]], [[.5, 0.]]]),
                prefix_vectors=[[100., 1.], [-100., 2.]],
                suffix_nominal_params=np.zeros((2, 1, 2)),
                suffix_nominal_terminal_vectors=[[50., 3.], [-50., 1.]],
                suffix_jacobians=[[[9., 0.], [1., 0.]], [[2., 0.], [4., 0.]]],
                zero_candidate_id="zero", damping=1., beta=10.)


def test_affine_closed_form_same_metric_and_normalization():
    inputs = bank()
    untouched = copy.deepcopy(inputs)
    result = compile_candidates(**inputs)
    zero, other = result["rows"]
    assert zero["stage_cost"] == pytest.approx(.5)
    assert zero["own_reach_value"] == pytest.approx(9 / 4)
    assert other["stage_cost"] == pytest.approx(2.)
    assert other["own_reach_value"] == pytest.approx(1 / 34)
    assert result["selected_candidate_id"] == "reconfigure"
    assert result["greedy_candidate_id"] == "zero"
    assert result["zero_candidate_id"] == "zero"
    for row in result["rows"]:
        assert row["own_reach_value"] == pytest.approx(row["linear_terminal_cost"] + row["ridge_penalty"])
        assert row["objective"] == pytest.approx(row["stage_cost"] + 10 * row["own_reach_value"])
    np.testing.assert_allclose(other["suffix_delta_params"], [-4 / 17, 0.])
    np.testing.assert_allclose(other["program_params"], [[.5, 0.], [-4 / 17, 0.]])
    predictions = []
    for i, row in enumerate(result["rows"]):
        predictions.append(np.asarray(inputs["suffix_nominal_terminal_vectors"])[i]
                           + np.asarray(inputs["suffix_jacobians"])[i] @ row["suffix_delta_params"])
    terminal = score_terminal_predictions(goal_vectors=inputs["goal_vectors"],
                                         terminal_vectors=predictions,
                                         projection=result["metric"]["projection"])
    np.testing.assert_allclose(terminal["predicted_terminal_costs"],
                               [r["linear_terminal_cost"] for r in result["rows"]])
    assert terminal["metric_sha256"] == result["metric"]["sha256"]
    for key in inputs:
        np.testing.assert_equal(inputs[key], untouched[key])
    json.dumps(result, allow_nan=False)


def test_own_state_residual_not_shared_root_or_prefix_residual():
    inputs = bank()
    result = compile_candidates(**inputs)
    np.testing.assert_allclose(result["rows"][1]["own_projected_residual"], [0., -1.])
    # Deliberately shared residual changes V by 9x: this is NOT the compiler.
    wrong_shared = 3**2 / (2 * 17)
    assert result["rows"][1]["own_reach_value"] != pytest.approx(wrong_shared)
    assert result["rows"][1]["own_reach_value"] != pytest.approx(2**2 / (2 * 17))
    inputs["suffix_nominal_terminal_vectors"][0][1] = 300.
    assert compile_candidates(**inputs)["rows"][1] == result["rows"][1]


def test_matches_augmented_ridge_system_with_nonzero_nominal():
    inputs = bank()
    inputs["suffix_nominal_params"][:] = .25
    result = compile_candidates(**inputs)
    q, _ = compile_metric(inputs["goal_vectors"])
    for i, row in enumerate(result["rows"]):
        k = q @ np.asarray(inputs["suffix_jacobians"])[i]
        r = q @ (np.asarray(inputs["goal_vectors"])[0] - inputs["suffix_nominal_terminal_vectors"][i])
        solution = np.linalg.lstsq(np.vstack([k, np.eye(2)]), np.r_[r, [0., 0.]], rcond=None)[0]
        np.testing.assert_allclose(row["suffix_delta_params"], solution)
        np.testing.assert_allclose(np.asarray(row["suffix_params"]).ravel(), .25 + solution)


def test_zero_response_remains_valid_and_no_fabricated_pass():
    inputs = bank()
    inputs["suffix_jacobians"] = np.zeros((2, 2, 2))
    inputs["beta"] = 0.
    result = compile_candidates(**inputs)
    assert result["selected_candidate_id"] == result["greedy_candidate_id"] == "zero"
    assert result["rows"][0]["own_reach_value"] == pytest.approx(4.5)
    assert result["P1_admission"] is False
    assert result["mechanism_pass"] is None
    assert result["physical_labels_used"] is False
    assert result["model_inference_executed"] is False
    assert "gates" not in result
    assert "predicted_terminal_cost" not in result["rows"][0]


def test_metric_mismatch_rejected_at_every_role_and_terminal():
    inputs = bank()
    q, _ = compile_metric(inputs["goal_vectors"])
    bindings = {role: q for role in ("stage", "reach", "terminal")}
    compile_candidates(**inputs, metric_by_role=bindings)
    for role in bindings:
        with pytest.raises(ValueError, match="inconsistent"):
            compile_candidates(**inputs, metric_by_role={**bindings, role: np.eye(2)})
    with pytest.raises(ValueError, match="bindings"):
        compile_candidates(**inputs, metric_by_role={"stage": q})
    with pytest.raises(ValueError, match="inconsistent terminal"):
        score_terminal_predictions(goal_vectors=inputs["goal_vectors"],
                                   terminal_vectors=[[0., 1.]], projection=np.eye(2))


def test_goal_variant_and_nuisance_changes_do_not_change_scores():
    inputs = bank()
    before = compile_candidates(**inputs)
    inputs["goal_vectors"] = [[4., 0.], [0., 0.]]
    inputs["prefix_vectors"][0][0] = 9999.
    inputs["suffix_nominal_terminal_vectors"][1][0] = -9999.
    after = compile_candidates(**inputs)
    for old, new in zip(before["rows"], after["rows"], strict=True):
        for key in ("stage_cost", "own_reach_value", "objective", "suffix_params"):
            np.testing.assert_equal(old[key], new[key])


@pytest.mark.parametrize("key,value", [
    ("damping", 0.), ("damping", float("nan")), ("beta", -1.),
    ("beta", float("inf")), ("candidate_ids", ["zero", "zero"]),
    ("zero_candidate_id", "missing"), ("prefix_vectors", [[0., 1.]]),
    ("suffix_jacobians", np.zeros((2, 2, 3))),
    ("suffix_nominal_terminal_vectors", [[0., 0.], [0., float("nan")]]),
    ("goal_vectors", [[0., 0.]]),
])
def test_bad_contract_rejected(key, value):
    inputs = bank()
    inputs[key] = value
    with pytest.raises(ValueError):
        compile_candidates(**inputs)


def test_false_zero_control_rejected_and_stable_ties():
    inputs = bank()
    inputs["prefix_params"][0, 0, 0] = 1.
    with pytest.raises(ValueError, match="zero control"):
        compile_candidates(**inputs)
    inputs = bank()
    inputs["prefix_vectors"][1] = inputs["prefix_vectors"][0]
    inputs["beta"] = 0.
    assert compile_candidates(**inputs)["selected_candidate_id"] == "zero"


def test_no_outcome_input_surface():
    with pytest.raises(TypeError):
        compile_candidates(**bank(), physical_outcomes=[1, 0])


def test_general_affine_multi_step_bank_against_independent_ridge():
    rng = np.random.default_rng(81)
    inputs = bank()
    goals = np.array([[1., 2., 3., 4., 5.], [2., 3., 3., 4., 5.]])
    inputs.update(goal_vectors=goals, prefix_vectors=rng.normal(size=(2, 5)),
                  suffix_nominal_params=rng.normal(size=(2, 3, 2)),
                  suffix_nominal_terminal_vectors=rng.normal(size=(2, 5)),
                  suffix_jacobians=rng.normal(size=(2, 5, 6)), damping=.07)
    result = compile_candidates(**inputs)
    q, _ = compile_metric(goals)
    for i, row in enumerate(result["rows"]):
        k = q @ inputs["suffix_jacobians"][i]
        r = q @ (goals[0] - inputs["suffix_nominal_terminal_vectors"][i])
        solution = np.linalg.lstsq(np.vstack((k, np.sqrt(.07) * np.eye(6))),
                                  np.r_[r, np.zeros(6)], rcond=None)[0]
        np.testing.assert_allclose(row["suffix_delta_params"], solution, atol=1e-12)
        assert row["own_reach_value"] == pytest.approx(
            (np.linalg.norm(r - k @ solution)**2 + .07 * solution @ solution) / 5)
        assert len(row["program_params"]) == 4

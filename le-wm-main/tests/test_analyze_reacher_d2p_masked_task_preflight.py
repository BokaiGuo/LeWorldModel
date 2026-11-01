"""Synthetic contract checks only; never CUDA or efficacy evidence."""
import copy
import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import analyze_reacher_d2p_masked_task_preflight as analysis
from experiments.spectral_reachability.reacher_d2p_feasible_goal import feasible_goal_delta


def endpoint(raw):
    raw = np.asarray(raw, np.float32)
    value = raw.sum(axis=0)
    observed = np.tile(value, 96)
    boundaries = [{"root_rgb_sha256": "a" * 64,
        "action_sha256": hashlib.sha256(np.ascontiguousarray(raw[i:i+5]).tobytes()).hexdigest()} for i in range(0, len(raw), 5)]
    return {"raw_actions": raw.tolist(), "boundaries": boundaries, "observed_vector": observed.tolist(), "qpos": value.tolist()}


def horizon_fixture(horizon):
    matrix = np.tile(np.eye(2), (96, 5*horizon))
    tasks = {}
    for task in analysis.TASKS:
        indices = analysis.columns(task, horizon)
        jacobian = matrix[:, indices]
        direction = np.ones(len(indices))
        point = {"direction": direction.tolist(), "positive_vector": (.001*jacobian@direction).tolist(),
            "negative_vector": (-.001*jacobian@direction).tolist(), "relative_error": 0.}
        tasks[task] = {"active_columns": indices, "model_jacobian": jacobian.tolist(), "full_column_max_error": 0.,
            "fd_directions": [copy.deepcopy(point) for _ in range(4)], "fd_median_relative_error": 0.}
    predictions, observations = [], []
    for scale in analysis.SCALES:
        for coordinate in range(10*horizon):
            for sign in (-1, 1):
                raw = np.zeros((5*horizon, 2), np.float32)
                raw.ravel()[coordinate] = scale*sign
                point = endpoint(raw)
                key = {"scale": scale, "coordinate": coordinate, "sign": sign}
                predictions.append({**key, "raw_actions": point["raw_actions"], "predicted_vector": point["observed_vector"]})
                observations.append({**key, **point})
    return ({"horizon": horizon, "tasks": tasks, "center_predicted_vector": [0.]*192, "probes": predictions},
        {"horizon": horizon, "center": endpoint(np.zeros((5*horizon, 2))), "endpoints": observations})


@pytest.fixture
def fixture():
    forecasts, actuals = zip(*(horizon_fixture(h) for h in analysis.HORIZONS))
    roots = [{"seed": i, "source_episode": i+300, "source_start_row": i*200} for i in range(8)]
    _, admission = feasible_goal_delta([0., 0.], [.65, -.65], [False, True], [[0., 0.], [-2.79, 2.79]])
    predicted = [{**r, "goal_vector": [.1]*192, "goal_admission": copy.deepcopy(admission), "horizons": copy.deepcopy(forecasts)} for r in roots]
    actual = [{**r, "horizons": copy.deepcopy(actuals)} for r in roots]
    protocol = {"root_seeds": list(range(8)), "source_roots": roots, "excluded_source_episodes": list(range(192)),
        "tasks": {k: list(v) for k, v in analysis.TASKS.items()}, "horizons": [1, 2], "probe_scales": [.1, .25],
        "fd_epsilon": .001, "fd_directions_per_task": 4, "fd_median_relative_error_max": .1,
        "target_delta": [.65, -.65], "goal_contract": "joint-feasible-v1", "damping": .01,
        "inactive_raw_torques_literal_zero": True, "no_post_training": True,
        "all_predictions_sealed_before_any_future_execution": True, "scope": "synthetic contract fixture only"}
    report = {"schema": "reacher_d2p_masked_task_preflight.v1", "status": "completed", "sealed_protocol": protocol,
        "gates": {k: True for k in analysis.REQUIRED_GATES}, "rows": actual, "physical_futures": 976, "actual_model_predict_samples": 2272}
    return report, {"rows": predicted}


def test_complete_fixture_retains_all_tasks_and_roots(fixture):
    result = analysis.summarize(*fixture)
    assert result["physical_futures"] == 976 and result["P1_admission"] is False
    assert len(result["rows"]) == 8
    for r in result["rows"]:
        for h in r["horizons"]:
            assert set(h["tasks"]) == set(analysis.TASKS)
            for task in analysis.TASKS:
                for scale in h["tasks"][task]["scales"]:
                    assert scale["actual_admissible_probe_count"] == 1+2*len(analysis.columns(task, h["horizon"]))


def test_probe_root_array_order_does_not_pair_metrics(fixture):
    report, predictions = fixture
    report["rows"].reverse()
    predictions["rows"] = predictions["rows"][3:] + predictions["rows"][:3]
    result = analysis.summarize(report, predictions)
    assert [r["seed"] for r in result["rows"]] == list(range(8))


@pytest.mark.parametrize("mutation", ["gate_missing", "gate_false", "gate_integer", "partial_roots", "duplicate_root",
    "source", "overlap", "task", "scale", "prediction_seal_flag", "center_action", "probe_action", "packet_hash",
    "missing_probe", "duplicate_probe", "projection", "columns", "derivative", "fd", "accounting", "horizon"])
def test_fail_closed_on_changed_contracts(fixture, mutation):
    report, predictions = fixture
    p = predictions["rows"][0]["horizons"][0]
    m = report["rows"][0]["horizons"][0]
    if mutation == "gate_missing": report["gates"].pop("cuda_only")
    elif mutation == "gate_false": report["gates"]["cuda_only"] = False
    elif mutation == "gate_integer": report["gates"]["cuda_only"] = 1
    elif mutation == "partial_roots": report["rows"].pop()
    elif mutation == "duplicate_root": predictions["rows"][1]["seed"] = 0
    elif mutation == "source": predictions["rows"][0]["source_start_row"] += 1
    elif mutation == "overlap": report["sealed_protocol"]["source_roots"][0]["source_episode"] = 0
    elif mutation == "task": report["sealed_protocol"]["tasks"]["wrist_only"] = [0, 1]
    elif mutation == "scale": report["sealed_protocol"]["probe_scales"] = [.1, .3]
    elif mutation == "prediction_seal_flag": report["sealed_protocol"]["all_predictions_sealed_before_any_future_execution"] = False
    elif mutation == "center_action": m["center"]["raw_actions"][0][1] = .01
    elif mutation == "probe_action": p["probes"][0]["raw_actions"][0][1] = .01
    elif mutation == "packet_hash": m["endpoints"][0]["boundaries"][0]["action_sha256"] = "b"*64
    elif mutation == "missing_probe": m["endpoints"].pop()
    elif mutation == "duplicate_probe": m["endpoints"][-1] = copy.deepcopy(m["endpoints"][0])
    elif mutation == "projection": m["endpoints"][0]["observed_vector"] = [0., 0.]
    elif mutation == "columns": p["tasks"]["wrist_only"]["active_columns"][0] = 0
    elif mutation == "derivative": p["tasks"]["wrist_only"]["model_jacobian"][0][0] += .1
    elif mutation == "fd": p["tasks"]["wrist_only"]["fd_directions"][0]["positive_vector"][0] += 1
    elif mutation == "accounting": report["actual_model_predict_samples"] += 1
    else: m["horizon"] = 2
    with pytest.raises(ValueError):
        analysis.summarize(report, predictions)


def test_limited_wrist_residual_is_not_wrapped_as_periodic():
    actual = [-2.7, -2.7]
    target = [2.7, 2.7]
    value = analysis.joint_residual(target, actual, [False, True])
    np.testing.assert_allclose(value, [5.4-2*np.pi, 5.4])


def test_independent_bounded_fit_matches_interior_and_boundary():
    small = analysis.linear_geometry(np.eye(2), np.array([.01, .02]), .1)
    np.testing.assert_allclose(small["bounded_regularized_gain"], .0005/1.01, atol=1e-12)
    saturated = analysis.linear_geometry(np.eye(1), np.array([3.]), .25)
    np.testing.assert_allclose(saturated["bounded_regularized_gain"], 9-(2.75**2+.01*.25**2), atol=1e-12)
    assert saturated["physically_executed_fit"] is False


def test_true_model_and_joint_matrices_are_distinct_data_sources():
    forecast, measured = horizon_fixture(1)
    measured["endpoints"][0]["observed_vector"] = (2*np.asarray(measured["endpoints"][0]["observed_vector"])).tolist()
    _, matrices, _ = analysis.probe_matrices(forecast, measured, [False, True])
    assert not np.array_equal(matrices[.1]["model_fd"], matrices[.1]["physical_rgb_fd"])
    assert matrices[.1]["physical_qpos_fd"].shape == (2, 10)

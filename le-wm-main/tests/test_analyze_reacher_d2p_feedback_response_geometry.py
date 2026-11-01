import copy

import numpy as np
import pytest

from experiments.spectral_reachability import analyze_reacher_d2p_feedback_response_geometry as analysis
from tests.test_reacher_d2p_feedback_response_geometry import endpoints


def probe(scale=.1):
    e = endpoints(scale)
    # Identity-like fixture is a data-contract test, never efficacy evidence.
    j = np.ones((192, 20))
    q = np.stack((np.ones(20), np.zeros(20)))
    return {"scale": scale, "endpoints": e, "model_fd": j.tolist(), "physical_rgb_fd": (2 * j).tolist(), "physical_qpos_fd": q.tolist()}


def test_independent_full_probe_reconstruction():
    result = analysis.check_probe(probe())
    assert result["physical_rgb_fd"].shape == (192, 20)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "matrix", "action", "projection", "scale"])
def test_probe_errors_fail_closed(mutation):
    value = copy.deepcopy(probe())
    if mutation == "missing":
        value["endpoints"].pop()
    elif mutation == "duplicate":
        value["endpoints"][-1]["coordinate"] = 0
    elif mutation == "matrix":
        value["physical_rgb_fd"][0][0] += .1
    elif mutation == "action":
        value["endpoints"][0]["plus"]["raw_actions"][0][1] = .1
    elif mutation == "projection":
        value["endpoints"][0]["minus"]["observed"] = [0.] * 2
    else:
        value["scale"] = .2
    with pytest.raises(ValueError):
        analysis.check_probe(value)


def test_independent_box_gain_matches_analytic_interior():
    response, residual = np.eye(2), np.array([.1, .2])
    np.testing.assert_allclose(analysis.independent_gain(response, residual), np.sum(residual ** 2) / 1.01, atol=1e-12)


def test_independent_box_gain_respects_boundary():
    np.testing.assert_allclose(analysis.independent_gain(np.eye(1), np.array([3.])), 4.99, atol=1e-12)


def test_physical_boundary_label_required():
    with pytest.raises(ValueError):
        analysis.check_geometry({"physical_labels_used": False}, np.eye(2), np.ones(2))


def test_actual_outcomes_use_all_fixed_futures_not_surrogate():
    first, second = probe(.1), probe(.25)
    center = {"observed": [1.] * 192, "qpos": [1., 0.]}
    result = analysis.actual_probe_metrics({"center": center, "target_qpos": [0., 0.], "probes": [first, second]}, np.zeros(192))
    np.testing.assert_allclose(result["best_measured_rgb_cost"], .04, atol=1e-8)
    np.testing.assert_allclose(result["best_measured_qpos_distance"], .1, atol=1e-8)
    assert result["rgb_cost_improvement_over_center"] > 0


@pytest.mark.parametrize("gates", [{}, {k: True for k in analysis.REQUIRED_GATES - {"cuda_only"}},
    {**{k: True for k in analysis.REQUIRED_GATES}, "buffers_unchanged": False}])
def test_partial_or_false_gates_rejected(gates):
    with pytest.raises(ValueError):
        analysis.summarize({"schema": "reacher_d2p_feedback_response_geometry.v1", "status": "completed", "sealed_protocol": {}, "gates": gates})

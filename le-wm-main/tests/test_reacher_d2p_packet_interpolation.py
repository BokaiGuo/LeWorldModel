import copy

import numpy as np
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_packet_interpolation import ALPHAS, METHODS, interpolate_packet, summarize


def test_interpolation_endpoints_are_bitwise_exact_and_independent():
    source = np.full((5, 2), .12345679, np.float32)
    target = np.full((5, 2), -.9123456, np.float32)
    for alpha, expected in ((0, source), (1, target)):
        actual = interpolate_packet(source, target, alpha)
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, expected)


def test_interpolation_is_in_raw_torque_not_nonlinear_parameter_space():
    actual = interpolate_packet(np.zeros((5, 2)), np.ones((5, 2)), .25)
    np.testing.assert_array_equal(actual, np.full((5, 2), .25, np.float32))


@pytest.mark.parametrize("alpha", [-1, 2, np.nan, np.inf])
def test_invalid_interpolation_cannot_extrapolate(alpha):
    with pytest.raises(ValueError):
        interpolate_packet(np.zeros((5, 2)), np.ones((5, 2)), alpha)


@pytest.mark.parametrize("target", [np.zeros((2, 5)), np.full((5, 2), np.nan)])
def test_bad_packets_rejected(target):
    with pytest.raises(ValueError):
        interpolate_packet(np.zeros((5, 2)), target, .5)


def rows():
    methods = {name: [{"predicted_cost": 0., "observed_cost": 0., "latent_l2_error": 1.} for _ in ALPHAS] for name in METHODS}
    first = {"seed": 2, "model_eligible": True, "methods": methods}
    second = copy.deepcopy(first)
    second["seed"], second["model_eligible"] = 1, False
    for cell in first["methods"]["dispatch"][1:]:
        cell["predicted_cost"], cell["observed_cost"] = 1., -1.
    return [first, second]


def test_summary_keeps_all_roots_but_does_not_count_anchor_ties_as_success():
    result = summarize(rows())
    assert result["0.0"]["branch_count"] == 8
    assert result["0.0"]["strict_order_agreements"] == 0
    assert result["0.0"]["tied_model_or_observed"] == 1
    assert result["1.0"]["strict_order_reversals"] == [2]
    assert result == summarize(list(reversed(rows())))


def test_duplicate_or_empty_roots_cannot_form_completed_summary():
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError):
        summarize([rows()[0], rows()[0]])

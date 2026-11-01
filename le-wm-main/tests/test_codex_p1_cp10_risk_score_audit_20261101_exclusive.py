import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_cp10_risk_score_audit_20261101_exclusive as audit


def test_fixed_perturbations_and_full_cost():
    packet = np.zeros((5, 2), dtype=np.float32)
    variants = audit.variants(packet)
    assert len(variants) == 5
    assert np.array_equal(variants[0], packet)
    assert np.allclose(variants[1], 0.05)
    assert np.allclose(variants[2], -0.05)
    assert np.allclose(variants[3][:, 0], 0.05)
    assert np.allclose(variants[3][:, 1], -0.05)
    assert np.allclose(variants[4], -variants[3])
    calls = []
    result = audit.evaluate(packet, lambda x: calls.append(x.copy()) or float(np.sum(x*x)))
    assert len(calls) == 5
    assert result["robust_score"] == pytest.approx(result["mean_q"] + result["range_q"])


def test_projection_is_uniform_and_invalid_data_fails():
    packet = np.ones((5, 2), dtype=np.float32)
    assert all(np.abs(x).max() <= 1 for x in audit.variants(packet))
    with pytest.raises(ValueError):
        audit.variants(np.ones((4, 2)))
    with pytest.raises(ValueError):
        audit.evaluate(packet, lambda x: float("nan"))


def test_stable_tie_and_candidate_order():
    candidates = [np.zeros((5, 2)), np.ones((5, 2)) * 0.1]
    value = audit.select(candidates, lambda _: 0.0)
    assert value["chosen_index"] == 0
    assert len(value["scores"]) == 2 and value["predictor_calls"] == 10

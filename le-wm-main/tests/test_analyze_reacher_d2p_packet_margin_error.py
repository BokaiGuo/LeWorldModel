"""Analytic identities only; no CPU neural computation."""
import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_packet_margin_error import decompose


@pytest.mark.parametrize("seed", range(8))
def test_exact_identity_on_all_full_coordinates(seed):
    rng = np.random.default_rng(seed)
    vectors = [rng.normal(size=192) for _ in range(5)]
    result = decompose(*vectors)
    assert result["identity_abs_error"] < 1e-12
    assert result["margin_error"] == pytest.approx(result["baseline_bias"] + result["response_linear"] + result["response_quadratic"])
    assert result["oracle_future_neutral_successor_used"] is True


@pytest.mark.parametrize("bad", [[0.] * 191, [float("nan")] * 192, [float("inf")] * 192])
def test_incomplete_or_nonfinite_coordinate_rejected(bad):
    with pytest.raises(ValueError):
        decompose(bad, *[[0.] * 192 for _ in range(4)])


def test_exact_predictions_have_zero_error_terms():
    p, pz, g = np.arange(192) / 192, np.zeros(192), np.ones(192)
    result = decompose(p, pz, p.copy(), pz.copy(), g)
    assert all(result[k] == 0. for k in ("margin_error", "baseline_bias", "response_linear", "response_quadratic"))

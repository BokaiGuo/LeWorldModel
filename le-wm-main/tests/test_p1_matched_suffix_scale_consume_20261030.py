import pytest

from experiments.spectral_reachability.p1_matched_suffix_scale_consume_20261030 import response


def test_fixed_response_and_missing_not_imputed():
    anchor = {"goal": [0., 0., 1., 0., 0., 0., 0.]}
    endpoints = [[0.] * 7 for _ in range(21)]
    endpoints[1] = [0., 0., .5, 0., 0., 0., 0.]
    endpoints[2] = None
    result = response(anchor, endpoints)
    assert result[0]["position_gain_vs_zero"] == 0.
    assert result[1]["position_gain_vs_zero"] == .5
    assert result[2] is None
    endpoints[0] = None
    assert response(anchor, endpoints) == [None] * 21


def test_bad_endpoint_rejected():
    with pytest.raises(ValueError, match="fixed endpoints"):
        response({"goal": [0.] * 7}, [])

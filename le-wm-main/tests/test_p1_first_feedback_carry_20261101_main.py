import pytest
import json
from experiments.spectral_reachability.p1_first_feedback_carry_20261101_main import dimensions, binding


def test_first_feedback_has_carry_and_exhaustion_has_none():
    assert dimensions(5) == (2, True)
    assert dimensions(10) == (1, True)
    assert dimensions(15) == (1, False)
    assert binding.extra_cost(8, 2, has_carry=True)["predict_calls"] == 18
    assert binding.extra_cost(8, 1, has_carry=False)["predict_calls"] == 8


def test_unsupported_boundary_rejected():
    with pytest.raises(ValueError):
        dimensions(6)


def test_protocol_survives_json_without_tuple_drift():
    from experiments.spectral_reachability.p1_first_feedback_carry_20261101_main import horizon_protocol
    value = horizon_protocol()
    assert json.loads(json.dumps(value)) == value

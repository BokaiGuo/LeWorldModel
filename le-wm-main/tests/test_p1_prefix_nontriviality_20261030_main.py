import numpy as np
import pytest

from experiments.spectral_reachability import p1_prefix_nontriviality_20261030_main as audit


def test_signed_zero_is_not_a_reconfiguration():
    actions = np.zeros(10, dtype="<f4")
    actions[0] = -0.0
    result = audit.classify(actions.tobytes().hex())
    assert result["exact_numerical_zero"] is True
    assert result["signed_zero_components"] == 1
    assert result["nonzero_components"] == 0


def test_nonzero_is_not_a_mechanism_claim():
    actions = np.zeros(10, dtype="<f4")
    actions[0] = 0.1
    result = audit.classify(actions.tobytes().hex())
    assert result["exact_numerical_zero"] is False
    assert result["nonzero_components"] == 1


def test_nonfinite_and_out_of_bounds_rejected():
    for v in (float("nan"), 1.1):
        actions = np.zeros(10, dtype="<f4")
        actions[0] = v
        with pytest.raises(ValueError):
            audit.classify(actions.tobytes().hex())

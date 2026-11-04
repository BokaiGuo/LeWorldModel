import numpy as np
import pytest

from experiments.spectral_reachability import pc_response_temporal_warm_cem_v1 as warm


class Meter:
    token_cap = 1920

    def __init__(self):
        self.token_samples = 0

    def predict(self, actions):
        self.token_samples += len(actions) * actions.shape[1] // 5
        if self.token_samples > self.token_cap:
            raise ValueError("budget exceeded")
        scores = ((actions - 0.8) ** 2).mean(axis=(1, 2))
        return np.zeros((len(actions), 2)), scores


@pytest.mark.parametrize("step", [1, 2])
def test_carry_used_as_initialization_and_incumbent(step):
    meter = Meter()
    carry = np.full((15 - 5 * step, 2), 0.8)
    result = warm.decide("smooth_cem", 0, step, np.zeros(2), meter.predict, meter, carry)
    assert result["initialization"] == "carry"
    assert result["selected"]["terminal_score"] == 0
    assert np.array_equal(result["selected"]["chain"], carry)
    assert len(result["generations"]) > 1
    assert meter.token_samples <= meter.token_cap


def test_zero_selected_when_better():
    meter = Meter()
    result = warm.decide("action_cem", 0, 2, np.zeros(2), meter.predict, meter, np.full((5, 2), -1))
    assert result["initialization"] == "zero"


def test_invalid_carry_rejected():
    meter = Meter()
    with pytest.raises(ValueError):
        warm.decide("smooth_cem", 0, 1, np.zeros(2), meter.predict, meter, np.full((10, 2), 1.1))

import numpy as np
import pytest

from experiments.spectral_reachability import pc_response_temporal_two_start_cem_v1 as method


class Meter:
    token_cap = 1920

    def __init__(self):
        self.token_samples = 0

    def predict(self, actions):
        self.token_samples += len(actions) * actions.shape[1] // 5
        assert self.token_samples <= self.token_cap
        return np.zeros((len(actions), 2)), ((actions - 0.8) ** 2).mean(axis=(1, 2))


@pytest.mark.parametrize("step", [1, 2])
def test_two_starts_preserve_incumbent_and_split_budget(step):
    meter = Meter()
    carry = np.full((15 - 5 * step, 2), .8)
    result = method.decide("smooth_cem", 0, step, np.zeros(2), meter.predict, meter, carry)
    assert result["selected"]["terminal_score"] == 0
    assert np.array_equal(result["selected"]["chain"], carry)
    assert [entry["name"] for entry in result["restarts"]] == ["zero", "carry"]
    assert abs(result["restarts"][0]["samples"] - result["restarts"][1]["samples"]) <= 1
    assert all(len(entry["generations"]) > 1 for entry in result["restarts"])
    assert meter.token_samples == 1920

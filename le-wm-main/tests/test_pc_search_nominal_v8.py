import numpy as np
import pytest

from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_search_nominal_v8 as method


@pytest.mark.parametrize("dimension", [2, 5])
@pytest.mark.parametrize("step", [0, 1, 2])
def test_v8_budget_and_final_selection(dimension, step):
    horizon = 15 - 5*step
    meter, trace = Meter(1920), []
    def predict(actions):
        a = np.asarray(actions, dtype=np.float32)
        def evaluate(batch):
            position = np.stack((batch.sum(axis=(1, 2)), batch[:, :, 0].sum(axis=1)), axis=1)
            return position, np.linalg.norm(position - [1, .5], axis=1)
        p, s = meter.predict(a, evaluate)
        trace.append((a.copy(), s.copy()))
        return p, s
    carry = None if step == 0 else np.full((horizon, dimension), .1)
    result = method.decide("response", "response", 0, step, [1, .5], predict, meter,
                           np.zeros((2, 5, dimension)), carry)
    assert meter.token_samples <= 1920
    assert len(result["warm"]["generations"]) == 4
    assert result["warm"]["generations"][0]["loc_after"] == result["warm"]["generations"][1]["loc_before"]
    assert result["warm"]["tokens_before_v7"] == (48 + 1 + int(step > 0)) * (3-step)
    assert len(result["induced_nominals"]) == 2
    assert result["final_comparison"]["choice"] == int(np.argmin(result["final_comparison"]["scores"]))
    np.testing.assert_array_equal(np.asarray(result["selected"]["chain"], np.float32),
                                  trace[-1][0][result["final_comparison"]["choice"]])
    assert result["selected"]["terminal_score"] == trace[-1][1][result["final_comparison"]["choice"]]
    assert sum(len(a)*len(a[0])//5 for a, _ in trace) == meter.token_samples


def test_rejects_missing_later_carry():
    with pytest.raises(ValueError, match="missing carry"):
        method.decide("response", "response", 0, 1, [0, 0], None, Meter(1920), np.zeros((2, 5, 2)))

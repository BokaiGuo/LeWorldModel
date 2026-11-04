import numpy as np
import pytest

from experiments import pc_v22_pointmaze_feedback_adapter as adapter
from experiments.pc_equal_resource_dev_20261103_codex import Meter


@pytest.mark.parametrize("arm", ["response", "cem"])
@pytest.mark.parametrize("step", [0, 1, 2])
def test_shared_meter_horizon_and_carry(arm, step):
    meter = Meter(adapter.CAP)
    carry = None if step == 0 else np.full((15-5*step, 2), .1, np.float32)
    batches = []

    def predict(chains):
        def raw(actions):
            batches.append(actions.copy())
            position = actions.sum(axis=1)
            return position, np.linalg.norm(position - [1., -1.], axis=1)
        return meter.predict(chains, raw)

    result = adapter.decide(arm, 0, step, np.array([1., -1.]), predict, meter, carry)
    assert np.asarray(result["selected"]["chain"]).shape == (15-5*step, 2)
    assert result["resource"]["prediction_tokens"] == meter.token_samples <= adapter.CAP
    if step:
        np.testing.assert_array_equal(batches[0][1], carry)


def test_wrong_input_contract_fails_closed():
    meter = Meter(adapter.CAP)
    with pytest.raises(ValueError, match="carry"):
        adapter.decide("response", 0, 1, np.zeros(2), None, meter, None)
    with pytest.raises(ValueError, match="fresh"):
        adapter.decide("response", 0, 0, np.zeros(2), None, Meter(100))

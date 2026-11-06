import numpy as np
import pytest

from experiments import pc_v25_prefix_random_pointmaze_feedback_dev as adapter


@pytest.mark.parametrize("arm", ["response", "cem"])
def test_adapter_feedback_horizons_and_fresh_tail(arm):
    carry = None
    state = np.zeros(2)
    for step in range(3):
        class Meter:
            token_cap = adapter.CAP
            token_samples = 0

        meter = Meter()
        seen = []
        observation = state.copy()

        def predict(actions):
            seen.append(actions.copy())
            meter.token_samples += len(actions) * (actions.shape[1] // 5)
            successor = observation + actions.sum(axis=1)
            return successor, np.linalg.norm(successor - [1., -1.], axis=1)

        result = adapter.decide(arm, 1, step, np.array([1., -1.]), predict, meter, carry)
        chain = np.asarray(result["selected"]["chain"])
        assert chain.shape == (15 - 5 * step, 2)
        assert result["resource"]["prediction_tokens"] == meter.token_samples <= adapter.CAP
        if carry is not None:
            assert any(np.array_equal(candidate, carry) for candidate in seen[0])
        if arm == "response":
            assert result["planner"]["horizon_policy"]["random_tail_enabled"] == (step < 2)
        state += chain[:5].sum(axis=0)
        carry = chain[5:] if step < 2 else None

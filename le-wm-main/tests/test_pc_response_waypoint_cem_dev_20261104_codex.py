import numpy as np
import pytest

from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability.pc_response_waypoint_cem_dev_20261104_codex import plan


@pytest.mark.parametrize("mode", ["response", "random", "geometric"])
def test_direction_and_budget_on_known_linear_response(mode):
    meter = Meter(1920)
    calls = []

    def predict(actions):
        actions = np.asarray(actions, np.float32)
        def model(batch):
            features = batch.sum(axis=1) * np.array([.2, .1])
            return features, np.linalg.norm(features - [1., 0.], axis=1)
        features, scores = meter.predict(actions, model)
        calls.append((actions.copy(), scores.copy()))
        return features, scores

    result = plan(predict, meter, [1., 0.], 44, mode)
    direction = np.asarray(result["response"]["direction"])
    assert np.isclose(np.linalg.norm(direction), 1)
    if mode in ("response", "geometric"):
        np.testing.assert_allclose(direction, [0., 1.], atol=1e-12)
    assert result["response"]["rank"] == 2
    assert meter.token_samples <= 1920
    np.testing.assert_array_equal(np.asarray(result["selected"], np.float32),
                                  calls[-1][0][int(result["abstained"])])
    assert result["selected_score"] == min(float(s.min()) for a, s in calls if a.shape[1] == 15)


def test_zero_rank_fails_closed():
    meter = Meter(1920)

    def predict(actions):
        return meter.predict(np.asarray(actions, np.float32),
                             lambda a: (np.zeros((len(a), 2)), np.ones(len(a))))

    with pytest.raises(ValueError, match="zero response rank"):
        plan(predict, meter, [1., 0.], 44, "response")

import numpy as np
import pytest

from experiments import pc_v25_beam_resource_curve_dev_20261104_codex as experiment


@pytest.mark.parametrize("cap", experiment.CAPS)
@pytest.mark.parametrize("arm", experiment.ARMS)
def test_budgeted_arm_uses_scored_incumbent(cap, arm):
    calls = []

    def predict(actions):
        calls.append(actions.copy())
        features = actions.sum(axis=1)
        return features, np.linalg.norm(features - [1., -1.], axis=1)

    config = experiment.config_for(0, arm, cap)
    result = experiment.plan(predict, np.array([1., -1.]), config, arm)
    tokens = sum(len(batch) * (batch.shape[1] // 5) for batch in calls)
    assert tokens == experiment.token_count(result, arm) <= cap
    selected = np.asarray(result["selected"], np.float32)
    assert selected.shape == (15, 2)
    full = [batch for batch in calls if batch.shape[1] == 15]
    assert any(np.any(np.all(batch == selected, axis=(1, 2))) for batch in full)
    best = min(np.linalg.norm(batch.sum(axis=1) - [1., -1.], axis=1).min() for batch in full)
    assert result["selected_score"] == pytest.approx(best)

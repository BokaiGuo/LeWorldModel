import numpy as np
import pytest

from experiments import pc_successor_rgb_execution_dev_20261103_codex as experiment


class Meter:
    def __init__(self):
        self.token_samples = 0


@pytest.mark.parametrize("arm", experiment.ARMS)
@pytest.mark.parametrize("step", [0, 1, 2])
def test_fixed_terminal_budget_and_real_successor_composition(arm, step):
    meter = Meter()
    horizon = 15 - 5 * step

    def predict(actions):
        actions = np.asarray(actions)
        meter.token_samples += len(actions) * actions.shape[1] // 5
        assert meter.token_samples <= 480
        positions = actions.sum(axis=1)
        return positions, np.linalg.norm(positions - np.array([.3, -.2]), axis=1)

    carry = None if step == 0 else np.zeros((horizon, 2))
    result = experiment.decide(arm, 0, step, np.array([.3, -.2]), predict, meter, carry)
    assert np.asarray(result["selected"]["chain"]).shape == (horizon, 2)
    assert sum(batch["tokens"] for batch in result["prediction_trace"]) == meter.token_samples
    if arm == "successor" and step == 0:
        assert len(result["bank"]) == 4
        assert len(result["successor_generator"]) == 2
        assert all(item["samples"] >= 13 for item in result["programs"])
        assert result["spectral_counts"]["svd"] == 3
    if carry is not None:
        assert result["selected"]["terminal_score"] <= result["carry_score"] + 1e-12

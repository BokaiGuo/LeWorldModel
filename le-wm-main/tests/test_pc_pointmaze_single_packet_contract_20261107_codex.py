import numpy as np
import pytest
import torch

from experiments import pc_pointmaze_same_renderer_three_way_20261107_codex as sealed
from experiments.pc_pointmaze_single_packet_contract_20261107_codex import difference
from experiments.pc_pointmaze_heldout_amplitude_encoding_20261107_codex import comparisons


def test_difference_checks_shape_and_reports_mse():
    result = difference(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]))
    assert result == {"max_abs": 2.0, "mse": 2.5}
    with pytest.raises(ValueError, match="shape mismatch"):
        difference(torch.zeros(1), torch.zeros(2))


def test_sealed_panel_reproduces_prediction_encoding_disagreement():
    with (np.load(sealed.PRED, allow_pickle=False) as pred,
          np.load(sealed.ACTUAL, allow_pickle=False) as actual):
        matches = 0
        for root in range(4):
            for camera in range(2):
                key = f"r{root}_v{camera}"
                predicted = pred[key + "_pred"]
                encoded = actual[key]
                goal = pred[key + "_goal"]
                # Selection disagreement is a symptom, not proof of a code bug.
                report = comparisons(predicted, encoded, goal, np.arange(4.0))
                matches += report["selected"]["predicted"] == report["selected"]["encoded"]
        assert matches == 1

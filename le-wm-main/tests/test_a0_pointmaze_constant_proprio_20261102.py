import pytest
import torch
import numpy as np

from experiments.a0_pointmaze_constant_proprio_20261102 import constant_observation
from experiments.a0_pointmaze_constant_proprio_validate_20261102 import validate


def test_constant_proprio_does_not_depend_on_episode_and_rgb_is_scaled():
    rgb = torch.full((1, 3, 4, 4), 0.5)
    constant = torch.tensor([1.0, 2.0, 3.0, 4.0])
    first = constant_observation(rgb, constant)
    second = constant_observation(rgb * 0, constant)
    assert torch.equal(first["proprio"], second["proprio"])
    assert first["visual"][0, 0, 0, 0, 0] == 127.5
    assert first["proprio"].shape == (1, 1, 4)


def test_invalid_rgb_fails_closed():
    with pytest.raises(ValueError):
        constant_observation(torch.full((1, 3, 2, 2), 1.5), torch.zeros(4))


def test_saved_validation_rejects_wrong_choice():
    source_rows = []
    result_rows = []
    for root in range(4):
        positions = np.zeros((8, 8, 2))
        positions[0, 1] = [1, 0]
        scores = np.full((8, 8), 3.)
        scores[0, 0] = 0.
        source_rows.append({"root": root, "physical_terminal_positions": positions.tolist(),
                            "goal_position_evaluation_only": [0., 0.],
                            "physical_terminal_costs": np.linalg.norm(positions, axis=-1).tolist()})
        result_rows.append({"root": root, "model_scores": scores.tolist(), "model_pair": [0, 0],
                            "physical_regret": 0.})
    result = {"parameter_hash_before": "x", "parameter_hash_after": "x", "rows": result_rows}
    assert len(validate(result, {"rows": source_rows})) == 4
    result["rows"][0]["model_pair"] = [0, 1]
    with pytest.raises(ValueError):
        validate(result, {"rows": source_rows})

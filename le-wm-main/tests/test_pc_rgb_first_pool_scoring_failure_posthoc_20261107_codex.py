import numpy as np
import pytest

from experiments.pc_rgb_first_pool_scoring_failure_posthoc_20261107_codex import evaluate


def test_oracle_and_scores_not_conflated():
    model = np.arange(9).reshape(3, 3)
    encoded = -model
    success = np.zeros((3, 3, 25), dtype=bool)
    success[2, 2, 4] = True
    result = evaluate(model, encoded, success)
    assert result["ever_oracle"] and not result["ever_model"] and result["ever_encoded"]
    assert not result["endpoint_oracle"]


def test_tie_order_and_shape():
    result = evaluate(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3, 25)))
    assert result["model_index"] == result["encoded_index"] == 0
    with pytest.raises(ValueError):
        evaluate(np.zeros((3, 2)), np.zeros((3, 3)), np.zeros((3, 3, 25)))

import numpy as np
import pytest

from experiments.spectral_reachability import pc_recursive_response_v13 as v13
from experiments.spectral_reachability import pc_recursive_response_v15 as v15


def test_portfolio_accounts_for_all_predictions_and_keeps_incumbent():
    batches = []

    def predict(actions):
        actions = np.asarray(actions, np.float32)
        batches.append(actions.copy())
        features = actions.mean(axis=1)
        return features, np.sum((features - .37) ** 2, axis=1)

    config = v13.Config(action_dim=2, segment_length=5, segments=3,
        nominals=(tuple((0., 0.) for _ in range(5)),), token_cap=2409, seed=31)
    result = v15.compile_program(predict, [.37, .37], config)
    assert sum(len(x) * x.shape[1] // 5 for x in batches) == result["resource"]["prediction_tokens"]
    assert result["resource"]["prediction_tokens"] <= 2409
    scores = [predict_score for batch in batches if batch.shape[1] == 15
              for predict_score in np.sum((batch.mean(axis=1) - .37) ** 2, axis=1)]
    assert result["selected_score"] <= min(scores) + 1e-7
    assert len(result["cem_generations"]) > 1


def test_portfolio_rejects_other_contract_before_prediction():
    config = v13.Config(action_dim=2, segment_length=5, segments=3,
        nominals=(tuple((0., 0.) for _ in range(5)),), token_cap=2408)
    with pytest.raises(ValueError, match="2409"):
        v15.compile_program(lambda _: pytest.fail("unexpected prediction"), [0., 0.], config)

import numpy as np

from experiments.spectral_reachability import pc_global_multistart_cem as baseline
from experiments.spectral_reachability import pc_recursive_response_v13 as v13


def test_multistart_is_bounded_and_keeps_best_scored_incumbent():
    batches = []

    def predict(actions):
        raw = np.asarray(actions, np.float32)
        batches.append(raw.copy())
        features = raw.mean(axis=1)
        return features, ((features - .25) ** 2).sum(axis=1)

    config = v13.Config(action_dim=2, segment_length=5, segments=3,
        nominals=(tuple((0., 0.) for _ in range(5)),), token_cap=600, seed=44, global_samples=4)
    result = baseline.plan(predict, config, seed=72, starts=4)
    assert sum(len(batch) * len(batch[0]) // 5 for batch in batches) == result["prediction_tokens"]
    assert result["prediction_tokens"] <= 600
    assert len(result["bank_indices"]) == 4
    assert max(run["samples"] for run in result["starts"]) - min(run["samples"] for run in result["starts"]) <= 1
    assert result["selected_score"] <= min(run["score"] for run in result["starts"]) + 1e-6

import numpy as np

from experiments.spectral_reachability import pc_recursive_response_v13 as v13
from experiments.spectral_reachability import pc_recursive_response_v16 as v16


def test_multinominal_is_seeded_bounded_and_uses_shared_budget():
    batches = []

    def predict(actions):
        batches.append(actions.copy())
        values = actions.mean(axis=1)
        return values, ((values - .2) ** 2).sum(axis=1)

    config = v13.Config(action_dim=2, segment_length=5, segments=3,
        nominals=(tuple((0., 0.) for _ in range(5)),), token_cap=4095, seed=7)
    result = v16.compile_program(predict, [.2, .2], config, nominal_seed=99)
    assert len(result["extra_nominals"]) == 4
    assert result["extra_nominals"] == v16.extra_nominals(5, 2, 99)
    assert np.max(np.abs(result["extra_nominals"])) <= .8
    assert sum(len(b) * b.shape[1] // 5 for b in batches) == result["resource"]["prediction_tokens"]
    assert result["resource"]["prediction_tokens"] <= 4095
    assert len(result["child"]["layers"][0]["responses"]) == 5

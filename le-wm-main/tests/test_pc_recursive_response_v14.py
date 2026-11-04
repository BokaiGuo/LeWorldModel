import numpy as np

from experiments.spectral_reachability import pc_recursive_response_v14 as v14
from experiments.spectral_reachability import pc_recursive_response_v14_child as child


def test_carry_seeds_each_layer_without_losing_original_incumbent():
    batches = []

    def predict(actions):
        batches.append(actions.copy())
        features = actions.mean(axis=1)
        return features, ((features - .9) ** 2).sum(axis=1)

    nominal = tuple((0.,) for _ in range(5))
    config = child.Config(action_dim=1, segment_length=5, segments=3,
                          nominals=(nominal, nominal), token_cap=1920, seed=9)
    carry = np.ones((15, 1), np.float32)
    result = child.compile_program(predict, [.9], config, carry)
    np.testing.assert_array_equal(batches[0][1], carry)
    for layer in result["layers"]:
        candidates = [c for c in layer["candidates"]
                      if c["path"][-1]["nominal"] == 1 and c["path"][-1]["mode"] is None]
        assert candidates
        for candidate in candidates:
            np.testing.assert_allclose(np.asarray(candidate["chain"])[-5:], .95, atol=1e-7)
    assert all(np.max(np.abs(batch)) <= 1 for batch in batches)
    assert sum(len(x) * x.shape[1] // 5 for x in batches) == result["resource"]["prediction_tokens"]
    assert result["resource"]["prediction_tokens"] <= config.token_cap


def test_v14_uses_the_same_unprivileged_global_bank_and_retains_carry():
    batches = []

    def predict(actions):
        batches.append(actions.copy())
        features = actions.mean(axis=1)
        return features, ((features - .37) ** 2).sum(axis=1)

    nominal = tuple((0.,) for _ in range(5))
    config = v14.Config(action_dim=1, segment_length=5, segments=2,
                        nominals=(nominal, nominal), token_cap=1920, seed=27)
    carry = np.full((10, 1), .37, np.float32)
    result = v14.compile_program(predict, [.37], config, carry)
    assert result["global_proposals"]["choice"] == 1
    np.testing.assert_array_equal(batches[0][1], carry)
    np.testing.assert_array_equal(batches[1][1], carry)
    assert result["selected_score"] <= 1e-12
    assert sum(len(x) * x.shape[1] // 5 for x in batches) == result["resource"]["prediction_tokens"]

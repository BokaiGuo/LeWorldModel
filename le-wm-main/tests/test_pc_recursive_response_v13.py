import numpy as np
import pytest

from experiments.spectral_reachability import pc_recursive_response_v13 as v13


@pytest.mark.parametrize("horizon", [5, 10, 15])
@pytest.mark.parametrize("dim", [1, 2, 5])
def test_global_proposals_and_recursive_compiler_share_one_meter(horizon, dim):
    batches = []

    def predict(actions):
        batches.append(actions.copy())
        features = actions.mean(axis=1)
        return features, ((features - .37) ** 2).sum(axis=1)

    config = v13.Config(action_dim=dim, segment_length=5, segments=horizon // 5,
        nominals=(tuple(tuple(0. for _ in range(dim)) for _ in range(5)),),
        token_cap=1920, seed=27, global_samples=64)
    carry = np.full((horizon, dim), .37, np.float32)
    result = v13.compile_program(predict, np.full(dim, .37), config, carry)
    assert batches[0].shape == (66, horizon, dim)
    np.testing.assert_array_equal(batches[0][1], carry)
    assert result["global_proposals"]["choice"] == 1
    np.testing.assert_array_equal(batches[1][1], carry)
    assert result["selected_score"] <= 1e-12
    assert sum(len(x) * x.shape[1] // 5 for x in batches) == result["resource"]["prediction_tokens"]
    assert result["resource"]["prediction_tokens"] <= config.token_cap
    assert all(x.dtype == np.float32 and np.max(np.abs(x)) <= 1 for x in batches)


def test_invalid_budget_and_carry_are_rejected_before_prediction():
    def never(_):
        pytest.fail("unexpected prediction")

    settings = v13.Config(action_dim=1, segment_length=5, segments=1,
                         nominals=(tuple((0.,) for _ in range(5)),), token_cap=10)
    with pytest.raises(ValueError, match="budget"):
        v13.compile_program(never, [0.], settings)
    with pytest.raises(ValueError, match="carry"):
        v13.compile_program(never, [0.], settings, carry=np.zeros((6, 1)))


def test_global_bank_is_seeded_and_nonprivileged():
    banks = []
    config = v13.Config(action_dim=2, segment_length=5, segments=2,
        nominals=(tuple((0., 0.) for _ in range(5)),), token_cap=1200, seed=30)

    def predict(actions):
        banks.append(actions.copy())
        features = actions.mean(axis=1)
        return features, np.sum(features ** 2, axis=1)

    first = v13.compile_program(predict, [0., 0.], config)
    split = len(banks)
    second = v13.compile_program(predict, [0., 0.], config)
    np.testing.assert_array_equal(banks[0], banks[split])
    assert first["global_proposals"]["choice"] == second["global_proposals"]["choice"]
    assert first["resource"]["prediction_tokens"] == second["resource"]["prediction_tokens"]

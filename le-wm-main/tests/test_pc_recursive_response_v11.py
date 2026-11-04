import numpy as np
import pytest

from experiments.spectral_reachability.pc_recursive_response_v11 import Config, compile_program


def config(segment=5, segments=2, **kwargs):
    kwargs.setdefault("token_cap", 600)
    return Config(action_dim=1, segment_length=segment, segments=segments,
                  nominals=(tuple((0.,) for _ in range(segment)),),
                  **kwargs)


def test_packet_ledger_is_independent_of_search_segment():
    for segment, segments in ((5, 2), (10, 1)):
        seen = []

        def predict(actions):
            seen.append(actions.copy())
            feature = actions.sum(axis=1)
            return feature, (feature[:, 0] - .7) ** 2

        result = compile_program(predict, [.7], config(segment, segments))
        assert result["resource"]["prediction_tokens"] == sum(
            len(batch) * batch.shape[1] // 5 for batch in seen
        )
        assert result["resource"]["prediction_calls"] == len(seen)
        assert all(batch.shape[1] % 5 == 0 for batch in seen)
        assert seen[-1].shape[1] == 10


def test_carry_rescored_as_incumbent_and_refinement_seed():
    carry = np.full((10, 1), .31, dtype=np.float32)
    batches = []

    def predict(actions):
        batches.append(actions.copy())
        feature = actions.sum(axis=1)
        return feature, ((actions - carry[:actions.shape[1]]) ** 2).sum(axis=(1, 2))

    result = compile_program(predict, [3.1], config(), carry=carry)
    np.testing.assert_array_equal(batches[0][1], carry)
    assert result["incumbent_source"] == "carry"
    assert result["refinement"]
    np.testing.assert_allclose(result["refinement"][0]["mean_before"], carry)
    np.testing.assert_array_equal(np.asarray(result["selected"], np.float32), carry)


def test_invalid_packet_and_carry_fail_before_predictor():
    def never(_):
        pytest.fail("predictor must not be called")

    with pytest.raises(ValueError):
        compile_program(never, [1.], config(segment=3))
    with pytest.raises(ValueError, match="carry"):
        compile_program(never, [1.], config(), carry=np.zeros((5, 1)))


def test_budget_rejection_never_exceeds_cap():
    spent = 0

    def predict(actions):
        nonlocal spent
        spent += len(actions) * actions.shape[1] // 5
        return np.zeros((len(actions), 1)), np.zeros(len(actions))

    with pytest.raises(RuntimeError, match="budget"):
        compile_program(predict, [1.], config(token_cap=10))
    assert spent <= 10


@pytest.mark.parametrize("horizon", [15, 10, 5])
def test_feedback_horizons_reconcile_actual_calls(horizon):
    spent = 0

    def predict(actions):
        nonlocal spent
        spent += len(actions) * actions.shape[1] // 5
        features = actions.sum(axis=1)
        return features, ((features - .4) ** 2).sum(axis=1)

    nominals = tuple(tuple(tuple(row) for row in np.tile(n, (5, 1)))
                     for n in ([.25, 0.], [.25, .5]))
    settings = Config(action_dim=2, segment_length=5, segments=horizon // 5,
                      nominals=nominals, token_cap=1920)
    carry = None if horizon == 15 else np.full((horizon, 2), .1, np.float32)
    result = compile_program(predict, [.4, .4], settings, carry=carry)
    assert spent == result["resource"]["prediction_tokens"] <= 1920
    assert np.shape(result["selected"]) == (horizon, 2)

import numpy as np
import pytest

from experiments.spectral_reachability.pc_recursive_response_v10 import Config, compile_program


def run_model(config, target):
    seen = []
    target = np.asarray(target)
    weights = np.arange(1, config.action_dim * len(target) + 1, dtype=float).reshape(config.action_dim, -1) / 10

    def predict(actions):
        assert actions.dtype == np.float32
        assert np.abs(actions).max() <= 1
        seen.append(actions.copy())
        features = actions.sum(axis=1) @ weights
        return features, np.linalg.norm(features - target, axis=1)

    result = compile_program(predict, target, config)
    return result, seen


def test_recursive_expansion_and_true_spectral_selection():
    common = dict(action_dim=2, segment_length=3, segments=3,
                  nominals=(((0., 0.),) * 3,), amplitudes=(.2,),
                  token_cap=1000, seed=42)
    one, seen = run_model(Config(**common, max_modes=1), [1., -.2, .4])
    two, _ = run_model(Config(**common, max_modes=2), [1., -.2, .4])
    assert len(one["layers"]) == 3
    assert len(one["layers"][0]["candidates"]) == 3
    assert len(two["layers"][0]["candidates"]) == 5
    assert len(one["layers"][1]["responses"]) == 3
    assert one["resource"]["prediction_tokens"] == sum(len(b) * b.shape[1] // 3 for b in seen)
    assert one["resource"]["prediction_tokens"] <= 1000
    assert one["resource"]["svd_calls"] == sum(len(layer["responses"]) for layer in one["layers"])
    assert all(np.shape(x["matrix"]) == (3, 6) for layer in one["layers"] for x in layer["responses"])
    assert len(one["selected"]) == 9
    np.testing.assert_array_equal(seen[-1][int(one["abstained"])], np.asarray(one["selected"], np.float32))
    assert one["selected_score"] == min(one["paired_scores"])


def test_action_dimension_and_different_horizon_are_configurable():
    config = Config(action_dim=3, segment_length=2, segments=2,
                    nominals=(((0., 0., 0.),) * 2,), token_cap=400)
    result, seen = run_model(config, [.4, .5])
    assert np.asarray(result["selected"]).shape == (4, 3)
    assert any(x.shape[1] == 2 for x in seen)
    assert any(x.shape[1] == 4 for x in seen)


def test_budget_insufficiency_fails_before_exceeding_limit():
    config = Config(action_dim=2, segment_length=3, segments=3,
                    nominals=(((0., 0.),) * 3,), token_cap=10)
    total = 0

    def predict(actions):
        nonlocal total
        total += len(actions) * actions.shape[1] // 3
        return np.zeros((len(actions), 2)), np.zeros(len(actions))

    with pytest.raises(RuntimeError, match="budget"):
        compile_program(predict, [0., 0.], config)
    assert total <= config.token_cap


def test_invalid_nominal_and_bad_predictor_are_rejected():
    config = Config(action_dim=1, segment_length=1, segments=1, nominals=(((1.,),),))
    with pytest.raises(ValueError, match="nominal"):
        compile_program(lambda x: (x, x), [0.], config)
    config = Config(action_dim=1, segment_length=1, segments=1, nominals=(((0.,),),))
    with pytest.raises(ValueError, match="predictor output"):
        compile_program(lambda x: (np.zeros((len(x), 2)), np.zeros(len(x))), [0.], config)


def test_zero_rank_and_identical_pair_fail_closed():
    config = Config(action_dim=1, segment_length=2, segments=2,
                    nominals=(((0.,), (0.,)),), token_cap=100, refine_batch=99)
    result = compile_program(lambda x: (np.zeros((len(x), 1)), np.zeros(len(x))), [1.], config)
    assert all(not r["selected_modes"] for layer in result["layers"] for r in layer["responses"])
    assert result["abstained"]
    assert result["incumbent"] == result["comparison"]

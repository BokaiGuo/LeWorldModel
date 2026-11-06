import numpy as np

from experiments.spectral_reachability import pc_recursive_reachability_v18 as compiler


def test_successor_reachability_is_metered_and_has_remaining_horizon():
    config = compiler.Config(action_dim=2, segment_length=5, segments=3,
                             nominals=(tuple((0., 0.) for _ in range(5)),),
                             token_cap=4095, seed=4)
    calls = []

    def predict(actions):
        calls.append(np.asarray(actions).copy())
        features = actions.sum(axis=1)
        return features, np.linalg.norm(features - [2., -1.], axis=1)

    result = compiler.compile_program(predict, [2., -1.], config)
    assert result["layers"][0]["reachability"]
    assert result["layers"][1]["reachability"]
    assert not result["layers"][2]["reachability"]
    assert all(item["tail_steps"] == 10 for item in result["layers"][0]["reachability"])
    assert all(item["tail_steps"] == 5 for item in result["layers"][1]["reachability"])
    assert result["resource"]["prediction_tokens"] == sum(len(a) * a.shape[1] // 5 for a in calls)
    assert result["resource"]["prediction_tokens"] <= config.token_cap
    selected = np.asarray(result["selected"], np.float32)
    assert any(a.shape[1:] == selected.shape and np.any(np.all(a == selected, axis=(1, 2))) for a in calls)


def test_reachability_respects_packet_budget_on_short_horizons():
    for segments in (1, 2, 3):
        config = compiler.Config(action_dim=2, segment_length=5, segments=segments,
                                 nominals=(tuple((0., 0.) for _ in range(5)),),
                                 token_cap=1920)
        def predict(actions):
            features = actions.sum(axis=1)
            return features, np.linalg.norm(features, axis=1)
        result = compiler.compile_program(predict, [0., 0.], config)
        assert result["resource"]["prediction_tokens"] <= config.token_cap
        assert len(result["selected"]) == 5 * segments

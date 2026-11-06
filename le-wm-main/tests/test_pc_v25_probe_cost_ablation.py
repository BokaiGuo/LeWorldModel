import numpy as np

from experiments.spectral_reachability import pc_global_reachability_v25 as original
from experiments.spectral_reachability import pc_global_reachability_v25_probe_ablation as variant


def test_enabled_matches_original_and_disabled_reallocates():
    nominal = tuple(tuple((0., 0.) for _ in range(5)) for _ in range(2))
    common = dict(action_dim=2, segment_length=5, segments=3, model_packet_length=5,
                  nominals=nominal, token_cap=1920, seed=43, max_modes=2)
    def run(module, config):
        calls = []
        def predict(actions):
            calls.append(actions.copy())
            state = actions.sum(axis=1)
            return state, np.linalg.norm(state - [1., -1.], axis=1)
        result = module.compile_program(predict, np.array([1., -1.]), config,
                                        score_scale=1., score_target=0.)
        return result, calls

    frozen, frozen_calls = run(original, original.Config(**common))
    enabled, enabled_calls = run(variant, variant.Config(**common, successor_probe_enabled=True))
    assert len(frozen_calls) == len(enabled_calls)
    for actual, expected in zip(enabled_calls, frozen_calls, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(enabled["selected"], frozen["selected"])
    assert enabled["selected_score"] == frozen["selected_score"]
    disabled, disabled_calls = run(variant, variant.Config(**common, successor_probe_enabled=False))
    assert not any(layer["reachability"] for layer in disabled["child"]["layers"])
    assert sum(c.shape[0] * (c.shape[1] // 5) for c in disabled_calls) <= 1920
    assert disabled["resource"]["prediction_tokens"] >= 1918
    # Global proposal and first-layer finite differences and candidates are
    # identical; differences thereafter can be caused by selection or budget.
    for actual, expected in zip(disabled_calls[:4], frozen_calls[:4], strict=True):
        np.testing.assert_array_equal(actual, expected)

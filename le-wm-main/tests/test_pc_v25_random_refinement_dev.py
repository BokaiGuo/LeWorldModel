import numpy as np

from experiments.spectral_reachability import pc_global_reachability_v25 as frozen
from experiments.spectral_reachability import pc_v25_random_refinement_dev as variant


def test_disabled_exact_parity_and_enabled_budget_incumbent():
    nominal = tuple(tuple((0., 0.) for _ in range(5)) for _ in range(2))
    config = frozen.Config(action_dim=2, segment_length=5, segments=3, nominals=nominal,
                           token_cap=1920, seed=43, max_modes=2)
    def run(function, **extra):
        calls = []
        def predict(actions):
            calls.append(actions.copy())
            state = actions.sum(axis=1)
            return state, np.linalg.norm(state - [1., -1.], axis=1)
        result = function(predict, np.array([1., -1.]), config, score_scale=1., score_target=0., **extra)
        return result, calls
    original, expected = run(frozen.compile_program)
    disabled, actual = run(variant.compile_program)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(original["selected"], disabled["selected"])
    enabled, changed = run(variant.compile_program, random_refinement=True)
    assert enabled["resource"]["prediction_tokens"] <= config.token_cap
    assert enabled["refinement_seed_policy"]["replaced_seed_indices"]
    assert enabled["child"]["refinement_starts"][1]["source"] == "uniform_full_chain"
    assert len(changed) == len(actual)
    for a, b in zip(changed[:4], actual[:4], strict=True):
        np.testing.assert_array_equal(a, b)
    full = [(actions, np.linalg.norm(actions.sum(axis=1) - [1., -1.], axis=1))
            for actions in changed if actions.shape[1] == 15]
    assert enabled["selected_score"] <= min(float(scores.min()) for _, scores in full) + 1e-12

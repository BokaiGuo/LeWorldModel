import numpy as np

from experiments import pc_v25_fixed_opportunity_witness_calibration_posthoc_20261104_codex as calibration


def test_first_checkpoint_matches_previous_fixed_opportunity_and_curve_is_nested():
    prefix = np.full((1, 5, 2), .1, np.float32)
    calls = []

    def predict(actions):
        calls.append(actions.copy())
        features = actions.sum(axis=1)
        return features, np.linalg.norm(features - [1., -1.], axis=1)

    result = calibration.optimize(predict, prefix, 0)
    first = []
    count = 0
    for batch in calls:
        if count >= 49:
            break
        first.append(batch.copy())
        count += len(batch)
    assert count == 49
    expected = []

    def old_predict(actions):
        expected.append(actions.copy())
        features = actions.sum(axis=1)
        return features, np.linalg.norm(features - [1., -1.], axis=1)

    old = calibration.fixed.optimize(old_predict, prefix, {"greedy": [0], "lookahead": [0]}, 0)
    assert len(first) == len(expected)
    for a, b in zip(first, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    curve = result[0]["curve"]
    assert curve["1x49"] == old["all_bank_best"]
    assert curve["4x121"] <= curve["2x121"] <= curve["1x121"] <= curve["1x49"]
    assert sum(len(batch) for batch in calls) == 4 * 121

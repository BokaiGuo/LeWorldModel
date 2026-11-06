import numpy as np

from experiments.pc_paired_segment_repair_dev_20261105_codex import make_candidates, run_arm, witnesses


def test_candidate_composition_and_bounds():
    parent = np.full((15, 2), .99, np.float32)
    singles, pairs, extras, segments = make_candidates(parent, 20)
    assert singles.shape == (24, 15, 2)
    assert pairs.shape == extras.shape == (12, 15, 2)
    assert np.abs(pairs).max() <= 1
    for i, (a, b) in enumerate(segments):
        for j in range(3):
            expected = singles[2*i] if j == a else singles[2*i+1] if j == b else parent
            np.testing.assert_array_equal(pairs[i, 5*j:5*j+5], expected[5*j:5*j+5])


def test_witness_threshold_and_evaluation_budget():
    parent = np.zeros((15, 2), np.float32)
    singles, _, _, segments = make_candidates(parent, 3)
    flags = witnesses(1., np.full(24, 1.), np.full(12, .98), parent, singles, segments)
    assert all(flags)
    calls = []
    def predict(chains):
        calls.extend(np.asarray(chains).copy())
        feat = np.asarray(chains).sum(axis=1)
        return feat, np.linalg.norm(feat - [1., -1.], axis=1)
    result = run_arm(predict, parent, "pair", 3, 5)
    scores = np.linalg.norm(np.asarray(calls).sum(axis=1) - [1., -1.], axis=1)
    assert len(calls) == 157
    assert np.isclose(result["final_score"], scores.min())

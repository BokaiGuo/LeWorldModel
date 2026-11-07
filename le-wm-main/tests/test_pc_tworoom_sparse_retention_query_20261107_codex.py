import numpy as np

from experiments.pc_tworoom_sparse_retention_query_20261107_codex import ARMS, sparse_search


def test_queries_only_retained_b_and_preserves_candidate_identity():
    prefixes = np.zeros((13, 10, 2), np.float32)
    prefixes[:, :, 0] = np.arange(13)[:, None] / 20
    tails = np.zeros((9, 15, 2), np.float32)
    tails[:, :, 1] = np.arange(9)[:, None] / 20
    calls = []

    def score(indices, actions):
        calls.append(indices)
        for (p, t), action in zip(indices, actions, strict=True):
            np.testing.assert_array_equal(action[:10], prefixes[p])
            np.testing.assert_array_equal(action[10:], tails[t])
        return [float(p + t / 10) for p, t in indices]

    a, arms = sparse_search(prefixes, tails, np.arange(13)[::-1], score)
    assert a.shape == (13, 5)
    assert list(map(len, calls)) == [65, 8, 8, 8]
    for arm, indices in zip(ARMS, calls[1:], strict=True):
        assert {p for p, _ in indices} == set(arms[arm]["retained"])
        assert {t for _, t in indices} == {5, 6, 7, 8}
        assert arms[arm]["selected"][0] in arms[arm]["retained"]

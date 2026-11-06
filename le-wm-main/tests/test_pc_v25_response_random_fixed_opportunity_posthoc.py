import numpy as np

from experiments import pc_v25_response_random_fixed_opportunity_posthoc_20261104_codex as diagnostic


def test_random_bank_is_deterministic_bounded_and_preserves_nominals():
    anchor = np.zeros((5, 2), np.float32)
    candidate = anchor.copy()
    candidate[:, 0] = .25
    result = {"config": {"nominals": [anchor.tolist()]},
              "child": {"layers": [{"candidates": [
                  {"chain": anchor.tolist(), "path": [{"nominal": 0}]},
                  {"chain": candidate.tolist(), "path": [{"nominal": 0}]}]}]}}
    response, random = diagnostic.make_banks(result, 2)
    np.testing.assert_array_equal(random, diagnostic.make_banks(result, 2)[1])
    np.testing.assert_array_equal(response[0], random[0])
    assert np.max(np.abs(random)) <= 1
    np.testing.assert_allclose(np.linalg.norm(response[1]), np.linalg.norm(random[1]), rtol=1e-6)


def test_same_tail_opportunities_and_fixed_budget():
    bank = np.zeros((3, 5, 2), np.float32)
    bank[1, :, 0], bank[2, :, 1] = .2, -.3
    calls = []

    def predict(actions):
        calls.append(actions.copy())
        positions = actions.sum(axis=1)
        return positions, np.linalg.norm(positions - [1., -1.], axis=1)

    selection = diagnostic.score_and_choose(predict, bank, 0)
    for call in calls[1:]:
        np.testing.assert_array_equal(call[:, 5:], diagnostic.shared_tails(0))
    calls.clear()
    scores = diagnostic.optimize(predict, bank, selection["selections"], 0)
    assert sum(len(call) for call in calls) == len(bank) * diagnostic.SAMPLES
    assert all(len(indices) == 2 for indices in selection["selections"].values())
    assert scores["all_bank_best"] == min(scores["scores"])

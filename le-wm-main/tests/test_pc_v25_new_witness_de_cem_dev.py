import numpy as np
import pytest

from experiments import pc_v25_new_witness_de_cem_dev_20261104_codex as experiment
from experiments import pc_v25_new_witness_incumbent_audit_20261104_codex as incumbent_audit


@pytest.mark.parametrize("arm", experiment.ARMS)
def test_equal_budget_prefix_seed_and_incumbent(arm):
    prefixes = np.full((2, 5, 2), .2, np.float32)
    calls = []

    def predict(chains):
        calls.append(chains.copy())
        return chains.sum(axis=1), np.square(chains).sum(axis=(1, 2))

    result = experiment.optimize(predict, prefixes, 0, arm)
    assert sum(len(batch) for batch in calls) == 2 * experiment.RESTARTS * experiment.SAMPLES
    assert all(batch.shape[1:] == (15, 2) for batch in calls)
    assert all(np.max(np.abs(batch)) <= 1 for batch in calls)
    first_batch = calls[0]
    np.testing.assert_array_equal(first_batch[0, 5:], experiment.initial(0, 0, 0))
    np.testing.assert_array_equal(first_batch[:, :5], np.broadcast_to(prefixes[0], first_batch[:, :5].shape))
    for record in result:
        for run in record["runs"]:
            assert run["samples"] == experiment.SAMPLES
            actions = np.asarray(run["actions"])
            assert run["score"] == pytest.approx(np.square(actions).sum(), abs=1e-5)
    assert result[0]["best_score"] == min(np.square(batch).sum(axis=(1, 2)).min()
                                          for batch in calls[:len(calls) // 2])


def test_saved_batch_audit_detects_nonbest_return():
    prefix = np.zeros((5, 2), np.float32)
    chains = np.ones((experiment.SAMPLES, 15, 2), np.float32)
    chains[:, :5] = prefix
    chains[0, 5:] = 0
    scores = np.square(chains).sum(axis=(1, 2))
    calls = iter([(chains, np.zeros((len(chains), 2)), scores)])
    with pytest.raises(ValueError, match="not a scored best"):
        incumbent_audit.audit_restart(calls, prefix,
                                     {"actions": chains[1].tolist(), "score": float(scores[1])})

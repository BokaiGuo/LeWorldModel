import numpy as np

from experiments.pc_paired_new_task_costed_dev_20261105_codex import cem_arm, parent_bank, tasks
from experiments.pc_action_effect_ranked_end_to_end_dev_20261104_codex import tasks as old_tasks


def test_panel_and_parent_bank_contract():
    assert len(tasks()) == 16
    assert all(t not in old_tasks() for t in tasks())
    a = parent_bank(2)
    np.testing.assert_array_equal(a, parent_bank(2))
    assert a.shape == (64, 15, 2) and np.abs(a).max() <= 1
    assert np.count_nonzero(a[0]) == 0


def test_cem_budget_and_incumbent():
    calls = []
    def predict(chains):
        calls.extend(np.asarray(chains).copy())
        features = np.asarray(chains).sum(axis=1)
        return features, np.linalg.norm(features - [1., -1.], axis=1)
    result = cem_arm(predict, np.zeros((15, 2), np.float32), 31)
    scores = np.linalg.norm(np.asarray(calls).sum(axis=1) - [1., -1.], axis=1)
    assert len(calls) == 157
    assert np.isclose(result["final_score"], scores.min())

import numpy as np

from experiments.pc_cheap_probe_ablation_dev_20261105_codex import probe_arm
from experiments.pc_cheap_direction_edit_dev_20261105_codex import cheap_arm


def test_probe_identity_and_budget():
    logs = []
    def predict(actions):
        logs.append(np.asarray(actions).copy())
        features = np.asarray(actions).sum(axis=1)
        return features, np.linalg.norm(features - np.array([1., -1.]), axis=1)
    parent = np.full((15, 2), .98, np.float32)
    a = probe_arm(predict, parent, 123, True)
    probe_log = [x.copy() for x in logs]
    logs.clear()
    cheap_arm(predict, parent, [1., -1.], 123, True)
    np.testing.assert_array_equal(probe_log[1], logs[1])
    assert sum(map(len, probe_log)) == 157
    assert a["refine_samples"] == 150

import numpy as np

from experiments import pc_v25_scale_feasibility_posthoc_20261104_codex as witness


def test_fixed_two_paths_are_legal_and_replayed():
    task = witness.source.tasks()[0]
    records = witness.calculate(task)
    assert [record["height"] for record in records] == [-1.1, 1.1]
    assert any(record.get("success", False) for record in records)
    for record in records:
        if record["eligible"]:
            actions = np.asarray(record["actions"])
            assert actions.shape == (15, 2)
            assert np.max(np.abs(actions)) <= 1
            assert record["success"] == (record["distance"] < .05)

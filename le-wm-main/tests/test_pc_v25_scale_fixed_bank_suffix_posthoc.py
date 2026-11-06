import numpy as np

from experiments import pc_v25_scale_fixed_bank_suffix_posthoc_20261104_codex as experiment


def test_shared_tail_bank_and_equal_retention_capacity():
    np.testing.assert_array_equal(experiment.tail_bank(1), experiment.tail_bank(1))
    assert not np.array_equal(experiment.tail_bank(1), experiment.tail_bank(2))
    assert experiment.tail_bank(1).shape == (25, 10, 2)
    layer = {"candidates": [{"score": float(i)} for i in range(6)],
             "retention_alternatives": {name: [0, 1, 2, 3]
                                        for name in ("greedy", "diverse", "response", "random")}}
    selections = experiment.selection(layer, [6, 5, 4, 3, 2, 1])
    assert selections["lookahead"] == [0, 1, 5, 4]
    assert len({len(value) for value in selections.values()}) == 1

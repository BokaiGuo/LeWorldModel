import numpy as np

from experiments.decision_benchmark.run_uncertainty_gated_selection import (
    _root_rank_instability,
)


def test_rank_instability_is_zero_for_consensus_seeds():
    task = np.asarray([
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [1.1, 0.0]],
        [[0.0, 0.0], [0.9, 0.0]],
    ])
    result = _root_rank_instability(
        task,
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        ["r0", "r0"],
    )
    assert result["r0"] == 0.0


def test_rank_instability_detects_seed_selection_disagreement():
    task = np.asarray([
        [[0.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [0.0, 0.0]],
    ])
    result = _root_rank_instability(
        task,
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        ["r0", "r0"],
    )
    assert result["r0"] == 0.5

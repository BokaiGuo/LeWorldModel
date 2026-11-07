import json
from pathlib import Path

import numpy as np
import pytest

from experiments.pc_pusht_shared_pool_real_endpoint_20261107_codex import summarize

ROOT = Path(__file__).resolve().parents[1]


def test_success_ranking_includes_ties_and_all_candidates():
    metrics = summarize(np.array([2., 1., 2., 3.]), np.array([True, False, False, False]),
                        np.array([True, False, False, False]))
    assert metrics["final_best_rank"] == 2
    assert metrics["final_pair_correct"] == 1
    assert metrics["final_pair_ties"] == 1
    assert metrics["final_pair_total"] == 3
    assert not metrics["winner_final"]


@pytest.mark.parametrize("task,expected_success", [(0, 13), (1, 1)])
def test_saved_layer_metrics_are_reproducible(task, expected_success):
    path = ROOT / f"results/pc_osfal_pusht_success_pool_20261107_codex_task{task}.json"
    if not path.exists():
        pytest.skip("optional GPU experiment artifact is not available")
    data = json.loads(path.read_text())
    ever = np.asarray(data["labels_ever"], dtype=bool)
    final = np.asarray(data["labels_final"], dtype=bool)
    assert len(data["hashes"]) == len(ever) == len(final) == 301
    assert final.sum() == expected_success
    assert len(data["layers"]) == 13
    for layer in data["layers"].values():
        costs = np.asarray(layer["costs"])
        assert costs.shape == (301,)
        assert np.isfinite(costs).all()
        assert summarize(costs, ever, final) == layer["metrics"]

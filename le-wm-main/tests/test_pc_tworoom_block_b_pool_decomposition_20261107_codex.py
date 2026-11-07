import numpy as np
import pytest

from experiments.pc_tworoom_block_b_pool_decomposition_20261107_codex import analyze_task


def test_rank_recall_and_physical_regret_include_prefix_success():
    costs = np.arange(64, dtype=float)
    outcomes = [{"final_distance": float(63 - i), "ever": i in (1, 3),
                 "prefix_success": i == 1} for i in range(64)]
    result = analyze_task(costs, outcomes, {"x": 3, "y": 1})
    assert result["model_choice"] == 0
    assert result["pool_success_count"] == 2
    assert result["pool_tail_success_count"] == 1
    assert result["best_success_model_rank"] == 2
    assert result["best_tail_success_model_rank"] == 4
    assert result["methods"]["x"]["distance_regret_vs_model"] == -3
    assert result["methods"]["y"]["prefix_success"]


def test_rejects_missing_or_nonfinite():
    with pytest.raises(ValueError):
        analyze_task([0], [], {})
    outcomes = [{"final_distance": 0, "ever": False, "prefix_success": False}] * 64
    with pytest.raises(ValueError):
        analyze_task([float("nan")] * 64, outcomes, {})

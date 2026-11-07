import pytest

from experiments.pc_lineage_pool_bound_20261108_codex import success_bound


def test_saturated_pool_bound_is_pointwise_not_only_equal_counts():
    result = success_bound([[.01, .1], [.2, .3]], [.01, .2])
    assert result["success_bound_attained_pointwise"]
    assert result["missed_success_runs"] == 0


def test_missed_success_is_detected_without_changing_threshold():
    result = success_bound([[.01, .1], [.05, .3]], [.1, .05])
    assert result["possible_success_runs"] == 1
    assert result["selected_success_runs"] == 0
    assert result["missed_success_runs"] == 1


def test_unwitnessed_selection_is_rejected():
    with pytest.raises(ValueError, match="witnessed"):
        success_bound([[.1, .2]], [.15])

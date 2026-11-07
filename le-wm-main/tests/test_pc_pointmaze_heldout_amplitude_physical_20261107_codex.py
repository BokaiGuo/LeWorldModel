import pytest

from experiments.pc_pointmaze_heldout_amplitude_physical_20261107_codex import choices


def test_choices_use_only_supplied_four_scores_and_stable_ties():
    row = {"heldout_model_costs": [2, 1, 1, 3], "predictions": {
        "parent": [1, 1, 1, 1], "probe_min": [2, 2, 2, 2],
        "linear_ad": [3, 2, 1, 0], "quadratic": [0, 1, 2, 3]}}
    assert choices(row) == {"model": 1, "parent": 0, "probe_min": 0, "linear_ad": 3, "quadratic": 0}
    row["predictions"]["quadratic"][0] = float("nan")
    with pytest.raises(ValueError):
        choices(row)

import json

import pytest

from experiments.pc_jepa_privileged_bank_success_rank_posthoc_20261105_codex import rank_group


def test_ties_and_selection_are_explicit():
    result = rank_group([0, 1, 2], [1., 1., 2.], [False, True, True])
    assert result["selected_index"] == 0
    assert result["best_success_rank"] == {"strictly_better_count": 0, "tied_count": 2}
    assert result["successes_in_top_1_3_5"] == [0, 2, 2]
    json.dumps(result, allow_nan=False)


def test_no_success_and_nonfinite():
    assert rank_group([0], [1.], [False])["best_success_rank"] is None
    with pytest.raises(ValueError):
        rank_group([0], [float("nan")], [True])

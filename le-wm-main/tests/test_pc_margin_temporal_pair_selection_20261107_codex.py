import numpy as np
import pytest

from experiments.pc_margin_temporal_pair_selection_20261107_codex import scores_to_choices


def test_prefix_mask_and_strict_tie():
    scores = np.full((8, 9, 5), 10.)
    scores[:, :, :2] = 1.
    scores[:, 1, 2] = 2.
    scores[:, 1, 4] = 11.
    choices = scores_to_choices(scores)
    assert choices["post_prefix"][:, 0].all()
    assert not choices["all_time"][:, 0].any()
    assert not choices["endpoint"][:, 0].any()


def test_reject_nonfinite_and_wrong_clock():
    with pytest.raises(ValueError):
        scores_to_choices(np.zeros((8, 9, 4)))
    with pytest.raises(ValueError):
        scores_to_choices(np.full((8, 9, 5), np.nan))

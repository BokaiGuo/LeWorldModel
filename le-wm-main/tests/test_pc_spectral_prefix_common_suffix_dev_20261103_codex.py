import pytest

from experiments.pc_spectral_prefix_common_suffix_dev_20261103_codex import select


def test_eligible_candidate_does_not_qualify_different_selected_chain():
    first = [1, .9, 1.2, .8, 1.3, 1.3, 1.3, 1.3]
    terminal = [2, 1.9, 1.8, 1.7, 4, 4, 4, 4]
    result = select(first, terminal, 1)
    assert result["eligible_indices"] == [2]
    assert result["selected_index"] == 3
    assert result["abstained_reconfiguration"]


def test_selected_eligible_and_ties_do_not_pass():
    first = [1, .9, 1.2, 1.3, 1.3, 1.3, 1.3, 1.3]
    assert select(first, [2, 1.9, 1.8, 4, 4, 4, 4, 4], 1)["selected_is_qualified"]
    assert select(first, [2, 1.9, 1.9, 4, 4, 4, 4, 4], 1)["abstained_reconfiguration"]
    with pytest.raises(ValueError):
        select(first, [float("nan")] * 8, 1)

import pytest

from experiments.spectral_reachability import codex_p1_cem_budget_preflight_20261101_exclusive as m


def test_sample_allocation_reserves_winner_replay():
    assert m.allocate(174, 3, 4, 4) == {"samples": 14, "forward": 171, "unused": 3}
    assert m.allocate(174, 2, 4, 4) == {"samples": 21, "forward": 170, "unused": 4}
    assert m.allocate(174, 1, 4, 4) == {"samples": 43, "forward": 173, "unused": 1}


@pytest.mark.parametrize("args", [(14, 3, 4, 4), (174, 0, 4, 4), (True, 3, 4, 4),
                                  (174, 3, 4, 0)])
def test_invalid_or_unaffordable_fails_closed(args):
    with pytest.raises(ValueError):
        m.allocate(*args)


def test_saved_prepares_compare_without_inventing_usage():
    for root in (0, 1):
        result = m.inspect(root)
        assert result["identity_source_prepare_sha_verified"]
        assert result["feedback_horizons_match"]
        assert result["existing_cem_forward"] == [195, 130, 65, 65]
        assert result["structured_declared_forward"] == 174
        assert result["structured_declared_backward"] == 240
        assert result["proposal_forward"] == [171, 170, 173, 173]
        assert not result["proposal_implemented"]
        assert not result["actual_cost_verified"]
        assert not result["ready_to_run_comparison"]

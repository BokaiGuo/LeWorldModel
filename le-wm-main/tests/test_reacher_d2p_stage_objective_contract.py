"""Numeric bound fixtures, never neural or efficacy evidence."""
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_stage_objective_contract import stage_bound


def test_perfect_suffix_still_cannot_win_when_stage_exceeds_feasible_zero():
    r = stage_bound([1.]+[2.]*16, .192)
    assert r["zero_total_score_feasible_upper_bound"] == pytest.approx(1.001)
    assert r["ruled_out_even_with_perfect_suffix"] == list(range(1,17))
    assert r["all_non_greedy_ruled_out"]


def test_bound_does_not_claim_impossible_when_a_candidate_could_win():
    r = stage_bound([1.,1.0005]+[2.]*15, .192)
    assert not r["all_non_greedy_ruled_out"]
    assert 1 not in r["ruled_out_even_with_perfect_suffix"]


def test_zero_index_wins_exact_tie_and_empty_candidates_are_separate():
    assert stage_bound([1.]+[2.]*16,192.)["all_non_greedy_ruled_out"]
    r = stage_bound([1.]*17,0.)
    assert r["no_non_greedy_candidates"] and not r["all_non_greedy_ruled_out"]


@pytest.mark.parametrize("stages,cost", [([1.]*16,1.), ([float("nan")]*17,1.), ([1.]*17,-1.), ([-1.]*17,1.)])
def test_rejects_invalid_bounds(stages,cost):
    with pytest.raises(ValueError): stage_bound(stages,cost)

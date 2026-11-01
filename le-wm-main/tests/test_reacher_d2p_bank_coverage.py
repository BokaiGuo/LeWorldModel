import copy

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_bank_coverage import classify


def bank():
    return [{"index":i,"physical_non_greedy":False,"physical_terminal_rgb_better":False,
        "physical_terminal_joint_better":False,"model_non_greedy":False,"model_terminal_better":False} for i in range(17)]


def test_separates_coverage_qualification_and_selection_failures():
    b=bank()
    assert classify(b,1,1,[])["diagnostic_category"]=="no_measured_physical_candidate_in_bank"
    b[4].update({k:True for k in b[4] if k!="index"})
    r=classify(b,1,1,[])
    assert r["physical_candidates"]==r["model_recognizes_physical_candidates"]==[4]
    assert r["diagnostic_category"]=="qualification_did_not_admit_any_physical_candidate"
    assert classify(b,1,1,[4])["diagnostic_category"]=="selector_did_not_choose_available_qualified_physical_candidate"
    r=classify(b,4,4,[4])
    assert r["common_selected_physical_candidate"] and r["P1_admission"] is False


def test_endpoint_win_without_non_greedy_or_other_metric_does_not_qualify():
    b=bank()
    b[3].update(physical_terminal_rgb_better=True,physical_terminal_joint_better=True)
    assert not classify(b,3,3,[3])["physical_candidates"]
    b[3].update(physical_non_greedy=True,physical_terminal_rgb_better=False)
    assert not classify(b,3,3,[3])["physical_candidates"]


def test_rejects_missing_roots_or_changed_indices():
    b=copy.deepcopy(bank())
    b.pop()
    with pytest.raises(ValueError): classify(b,1,1,[])
    with pytest.raises(ValueError): classify(bank(),17,1,[])


def test_missing_historical_proposal_is_not_backfilled_as_a_sealed_decision():
    r=classify(bank(),1,None,[])
    assert r["proposal_selected_physical_candidate"] is None

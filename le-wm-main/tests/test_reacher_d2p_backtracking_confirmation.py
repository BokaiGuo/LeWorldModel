from copy import deepcopy

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_backtracking_confirmation import summarize


def _method(final, prefix):
    return {"physical": {"final_qpos_distance": final, "after_first_chunk_qpos_distance": prefix},
            "observed_first_chunk": {"candidate_goal_cost": prefix}}


def _report():
    return {"status": "completed", "gates": {"integrity": True}, "sealed_protocol": {"root_seeds": [1, 2]},
        "rows": [
            {"seed": 1, "source_episode": 11, "reconfiguration": _method(0.1, 0.8),
             "zero_prefix": _method(0.6, 0.7), "same_bank_greedy": _method(0.3, 0.2),
             "model_selection": {"non_greedy_witness_found": True, "future_reference_score_improvement": 0.2}},
            {"seed": 2, "source_episode": 12, "reconfiguration": None,
             "zero_prefix": _method(0.8, 0.7), "same_bank_greedy": _method(0.1, 0.2),
             "model_selection": {"non_greedy_witness_found": False}}]}


def test_summary_keeps_abstentions_and_pairs_identity_not_array_position():
    report = _report()
    expected = summarize(report)
    report["rows"].reverse()
    assert summarize(report) == expected
    assert expected["all_root_count"] == 2
    assert expected["eligible_root_count"] == 1
    assert expected["all_root_neutral_fallback_diagnostic"]["mean_final_minus_greedy"] == pytest.approx(0.25)
    assert expected["eligible_final_better_than_both_controls"] == 1


def test_summary_rejects_missing_duplicate_or_posthoc_eligible_roots():
    report = _report()
    missing = deepcopy(report)
    missing["rows"].pop()
    with pytest.raises(ValueError, match="root identity"):
        summarize(missing)
    duplicate = deepcopy(report)
    duplicate["rows"].append(duplicate["rows"][0])
    with pytest.raises(ValueError, match="root identity"):
        summarize(duplicate)
    report["rows"][1]["model_selection"]["non_greedy_witness_found"] = True
    with pytest.raises(ValueError, match="eligibility changed"):
        summarize(report)

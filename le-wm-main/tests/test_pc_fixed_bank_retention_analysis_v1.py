import copy

import pytest

from experiments.pc_fixed_bank_retention_analysis_v1 import analyze


def fixture():
    scores = [3.0, 2.0, 1.0]
    choices = {"response": [0, 2], "greedy": [0, 1], "diverse": [0, 1],
               "random": [1, 2], "lookahead": [0, 1]}
    selection = {"layer": {"candidates": [{}, {}, {}]}, "selections": choices}
    rules = {}
    for name, indices in choices.items():
        winner = min(indices, key=lambda i: (scores[i], i))
        rules[name] = {"indices": indices, "winner": winner, "best_score": scores[winner]}
    result = {"status": "completed", "diagnostic": {
        "candidates": [{"index": i, "best_score": s, "runs": [{"score": s}]}
                       for i, s in enumerate(scores)],
        "rules": rules, "samples_per_candidate": 20}}
    return selection, result


def test_response_added_removed_and_quality():
    selection, result = fixture()
    summary = analyze(selection, result)
    comparison = summary["response_comparisons"]["greedy"]
    assert comparison["response_added"] == [{"index": 2, "suffix_score": 1.0}]
    assert comparison["response_removed"] == [{"index": 1, "suffix_score": 2.0}]
    assert comparison["response_minus_baseline"] == -1.0
    assert summary["rules"]["greedy"]["gap_to_bank_best"] == 1.0


def test_rejects_tampered_result():
    selection, result = fixture()
    changed = copy.deepcopy(result)
    changed["diagnostic"]["rules"]["response"]["best_score"] = 0.0
    with pytest.raises(ValueError, match="disagrees"):
        analyze(selection, changed)


def test_rejects_unequal_capacity():
    selection, result = fixture()
    selection["selections"]["response"] = [2]
    result["diagnostic"]["rules"]["response"]["indices"] = [2]
    with pytest.raises(ValueError, match="capacity"):
        analyze(selection, result)

import copy

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_action_word_result import audit
from experiments.spectral_reachability import reacher_d2p_refined_dispatch as selector


def fixture():
    rows = [{"proposal_index": i, "immediate_goal_cost": 1. if i == 0 else (.2 if i == 1 else 1.2),
             "refined_goal_cost": .1 if i == 2 else .5} for i in range(17)]
    decision = selector.select_refined_bank(rows)
    cell = {"action_witness_raw": [[0., 0.]] * 30, "observed_prefix_rgb_cost": 1.,
        "physical": {"final_qpos_distance": .1}, "prefix_boundaries": [{"raw_steps_executed": step,
            "history_rgb_sha256": ["a", "b", "c"]} for step in (5, 10, 15, 20)],
        "execution_identity": {"reset_rgb_sha256": "root", "goal_rgb_sha256": "goal"}}
    bank = {"word_" + str(i): copy.deepcopy(cell) for i in range(17)}
    physical = {"seed": 10, "source_episode": 1, "source_start_row": 20, "decision": decision, "word_results": bank,
        "methods": {"dispatch": copy.deepcopy(bank["word_2"]), "same_bank_greedy": copy.deepcopy(bank["word_1"]), "zero": copy.deepcopy(bank["word_0"])}}
    sealed = {"seed": 10, "source_episode": 1, "source_start_row": 20, "root_rgb_sha256": "root", "goal_rgb_sha256": "goal",
        "receipt": {"proposal_rows": rows, "dispatch": decision, "word_support": {"labels": [{"signs": [1, 1], "amplitude": .25} for i in range(17)]}}}
    return {"status": "completed", "gates": {"checked": True}, "sealed_protocol": {"root_seeds": [10]}, "rows": [physical]}, {"rows": [sealed]}


def test_all_words_and_order_pairs_retained():
    report, selection = fixture()
    result = audit(report, selection)
    assert result["aggregate"]["nonzero_word_cells"] == 16
    assert result["aggregate"]["order_pair_count"] == 8


def test_root_identity_not_array_position():
    report, selection = fixture()
    second, sealed = copy.deepcopy(report["rows"][0]), copy.deepcopy(selection["rows"][0])
    second["seed"] = sealed["seed"] = 11
    report["rows"].insert(0, second)
    selection["rows"].append(sealed)
    report["sealed_protocol"]["root_seeds"].append(11)
    assert [row["seed"] for row in audit(report, selection)["rows"]] == [10, 11]


@pytest.mark.parametrize("mutation", ["duplicate", "missing_word", "bad_alias", "partial_action", "collapsed_history", "root", "source", "decision"])
def test_corrupted_evidence_rejected(mutation):
    report, selection = fixture()
    row = report["rows"][0]
    if mutation == "duplicate":
        report["rows"].append(copy.deepcopy(row))
    elif mutation == "missing_word":
        del row["word_results"]["word_16"]
    elif mutation == "bad_alias":
        row["methods"]["dispatch"]["physical"]["final_qpos_distance"] = .2
    elif mutation == "partial_action":
        row["word_results"]["word_16"]["action_witness_raw"] = [[0., 0.]] * 15
    elif mutation == "collapsed_history":
        row["word_results"]["word_16"]["prefix_boundaries"] = row["word_results"]["word_16"]["prefix_boundaries"][-1:]
    elif mutation == "root":
        row["word_results"]["word_16"]["execution_identity"]["reset_rgb_sha256"] = "other"
    elif mutation == "source":
        row["source_start_row"] += 1
    else:
        row["decision"] = dict(row["decision"], selected_program_index=1)
    with pytest.raises(ValueError):
        audit(report, selection)

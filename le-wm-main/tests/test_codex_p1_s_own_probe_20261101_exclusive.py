import copy

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_s_own_probe_20261101_exclusive as probe


def candidate_rows():
    return [
        {"kind": "optimized_candidates", "payload": {"raw": np.zeros((8, 3, 5, 2)).tolist()}},
        {"kind": "final_selection", "payload": {"index": 0, "scores": list(range(8))}},
        {"kind": "first_feedback_selection", "payload": {"selected_source": "carry"}},
    ]


def test_full_h3_selection_and_shape():
    plan, index, scores = probe.choose(candidate_rows())
    assert index == 0 and len(plan) == 15 and scores[0] == 0


@pytest.mark.parametrize("change", ["tie", "wrong_index", "nan", "bad_carry", "out_of_range"])
def test_fail_closed_selection(change):
    rows = copy.deepcopy(candidate_rows())
    if change == "tie":
        rows[1]["payload"]["scores"][1] = 0
    elif change == "wrong_index":
        rows[1]["payload"]["index"] = 1
    elif change == "nan":
        rows[1]["payload"]["scores"][0] = float("nan")
    elif change == "bad_carry":
        rows[2]["payload"]["selected_source"] = "candidate"
    else:
        rows[0]["payload"]["raw"][0][0][0][0] = 2
    with pytest.raises(ValueError):
        probe.choose(rows)


def test_known_sources_and_rebound_identity():
    manifest, images = probe.build()
    assert len(manifest["slot"]["commands"]) == 20
    assert len(images) == 3
    assert manifest["original_selection"]["actual_carry_index"] == 8
    assert manifest["original_selection"]["precarry_h3_index"] == 0
    assert manifest["prospectively_sealed"] is False
    assert manifest["neural_calls_this_probe"] == 0

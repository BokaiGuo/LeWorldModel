import copy
import json

import pytest

from experiments.spectral_reachability import p1_response_integrated_evidence_20261031 as integration


def inputs():
    prepared = json.loads(integration.binding.NEW.read_text())
    feedback = json.loads(integration.FEEDBACK.read_text())
    rows = []
    for anchor in prepared["anchors"]:
        for probe in range(42):
            rows.append({**{k: anchor[k] for k in ("logical_pair", "role", "root_slot", "condition", "arm")},
                         "anchor": anchor["index"], "probe_index": probe})
    panel = {"rows": rows, "fixed_prefixes": 16, "fixed_branches": 588,
             "P1_admission": False, "candidate_comparisons": []}
    return panel, feedback


def test_separate_feedback_and_common_suffix_denominators():
    panel, feedback = inputs()
    result = integration.integrate(panel, feedback)
    assert len(result["common_suffix_rows"]) == 588
    assert len(result["feedback_by_anchor"]) == 14
    missing = [row["anchor"] for row in result["feedback_by_anchor"] if row["feedback"] is None]
    assert missing == [3]
    assert not result["equal_budget_certified"]


def test_feedback_role_or_probe_identity_mismatch_rejected():
    panel, feedback = inputs()
    changed = copy.deepcopy(feedback)
    changed["rows"][0]["role"] = "model_selected"
    with pytest.raises(ValueError, match="feedback task identity"):
        integration.integrate(panel, changed)
    panel["rows"][0]["probe_index"] = 100
    with pytest.raises(ValueError, match="panel roster"):
        integration.integrate(panel, feedback)

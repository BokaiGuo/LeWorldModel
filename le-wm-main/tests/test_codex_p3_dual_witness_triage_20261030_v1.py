"""Synthetic strict-witness tests; no model or physical execution."""
from copy import deepcopy

import pytest

from experiments.spectral_reachability import codex_p3_dual_witness_triage_20261030_v1 as m


def panel(h, qualifying=()):
    rows = [dict(index=i, status="complete", immediate=2 if i in qualifying else 1,
                 reach=0 if i in qualifying else 1, eligible=i in qualifying) for i in range(8)]
    return dict(bank=dict(horizon=h), control=dict(index="zero", status="complete", immediate=1, reach=1),
                rows=rows, eligible_indices=list(qualifying),
                diagnostic_reach_winner=min(qualifying) if qualifying else None,
                status="witness_found" if qualifying else "no_witness")


def test_both_strict_inequalities_and_selection():
    result = m.audit_panel(panel(3, (2, 6)), 3)
    assert result["eligible_indices"] == [2, 6] and result["selection_index"] == 2
    assert result["rows"][0]["immediate_gt_zero"] is False
    assert result["rows"][0]["reach_lt_zero"] is False


@pytest.mark.parametrize("damage", ["tie_immediate", "tie_reach", "wrong_eligible", "missing",
                                   "nonfinite", "horizon", "diagnostic", "partial", "reorder"])
def test_strict_failure_and_tamper(damage):
    p = deepcopy(panel(3, (2,)))
    if damage == "tie_immediate":
        p["rows"][2]["immediate"] = 1
    elif damage == "tie_reach":
        p["rows"][2]["reach"] = 1
    elif damage == "wrong_eligible":
        p["eligible_indices"] = []
    elif damage == "missing":
        p["rows"].pop()
    elif damage == "nonfinite":
        p["rows"][2]["reach"] = float("nan")
    elif damage == "horizon":
        p["bank"]["horizon"] = 1
    elif damage == "diagnostic":
        p["diagnostic_reach_winner"] = None
    elif damage == "partial":
        p["rows"][4]["status"] = "attempted"
    else:
        p["rows"].reverse()
    with pytest.raises((ValueError, KeyError)):
        m.audit_panel(p, 3)


def test_future_contract_never_authorizes():
    contract = m.future_contract()
    assert not contract["planning_or_physical_authorized"] and not contract["scientific_pass"]
    assert contract["original_contract"] == m.dual.contract()
    assert "no packet" in contract["no_witness"]


def test_failure_classification_is_not_dispatch(monkeypatch):
    monkeypatch.setattr(m.dual.core, "audit_ledger", lambda receipt: None)
    decision = dict(schema=m.dual.NAME + ".decision", contract=m.dual.contract(),
                    branch="reconfiguration", decision=dict(continue_expanding=True,
                    program_type="reconfiguration", extension_type="reconfiguration",
                    candidates=8, horizon=3), status="failed",
                    error="ValueError: R group has no witness; both panels required",
                    selected_actions=None, selected_horizon=None, stage=2,
                    panels=[panel(1, (2,)), panel(3)], resource_receipt={})
    result = m.classify(decision)
    assert result["classification"] == "strict_no_witness"
    assert result["original_run_status"] == "failed" and not result["dispatch_authorized"]
    decision["panels"][1] = panel(3, (1,))
    with pytest.raises(ValueError, match="absent witness"):
        m.classify(decision)

"""Read-only reconciliation of two saved E1 ledger projections."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"
ANCESTOR = "codex_e1_ancestor_integrated_ledger_20261030_v1_LOCAL01.json"
BRIDGE = "codex_e1_hi_unallocated_event_bridge_20261030_v1_LOCAL01.json"
HASHES = {
    ANCESTOR: "0bd6601848d43ed8c5e0763f98e7858a7d6909347046b4a277497eadc0a5988b",
    BRIDGE: "7673ac35fcf7103e8231b5b6e0ea1578da05528ebb18fce587119a5a8879261c",
}
ARMS = ("arbitrary_latent_waypoint", "existing_learned_prior", "dynamics_compiled_witness")


def load(name):
    data = (RESULTS / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == HASHES[name]
    return json.loads(data)


def crosscheck(ancestor, bridge):
    for obj in (ancestor, bridge):
        assert obj["status"] == "blocked_incomplete"
        assert all(obj[k] is False for k in ("ranking_authorized", "equal_budget_certified", "scientific_admission"))
        assert len(obj["slots"]) == 6
    assert all(bridge["sources"].get(k) == v for k, v in ancestor["sources"].items())
    assert set(ancestor["unresolved"]) <= set(bridge["unresolved"])
    for index, (before, after) in enumerate(zip(ancestor["slots"], bridge["slots"])):
        assert before["slot"] == after["slot"] == index
        assert before["root"] == after["root"] == index // 3
        assert before["ledger_input"]["arm"] == ARMS[index % 3]
        assert before["ledger_input"] == after["ledger_input"]
        assert before["ledger_preflight"] == after["ledger_preflight"]
        assert after["ledger_preflight"]["structural_status"] == "incomplete"
        assert "proposal:receipt_missing" in after["ledger_preflight"]["blockers"]
        assert "inherited:receipt_missing" in after["ledger_preflight"]["blockers"]
        assert before.get("partial_inherited_source") == after.get("partial_inherited_source")
        assert ("partial_inherited_source" in after) == (index % 3 == 2)
        assert ("unallocated_hi_source_row" in after) == (index % 3 == 1)
        if index % 3 == 1:
            hi = after["unallocated_hi_source_row"]
            assert hi["root"] == index // 3 and hi["condition"] == "off"
            assert hi["allocation"] == "row_execution_only_not_target_causal_cost"
            assert all(hi[k] == "unknown" for k in ("cross_stage_overlap", "event_coverage", "failed_attempts", "shared_costs"))
            assert hi["axes"]["high_predictor_calls"]["observed_returned_events"] == 14
            assert hi["axes"]["high_input_rows"]["observed_returned_input_rows"] == 98
        if index % 3 == 2:
            inherited = after["ledger_input"]["phases"]["inherited"]
            assert inherited["scope"] == "unknown" and inherited["receipt_sha256"] is None
    for root in range(2):
        slots = bridge["slots"][root * 3 : root * 3 + 3]
        assert len({s["ledger_input"]["task_sha256"] for s in slots}) == 1
        assert len({s["ledger_input"]["target_sha256"] for s in slots}) == 3


def test_saved_projection_crosscheck():
    crosscheck(load(ANCESTOR), load(BRIDGE))


@pytest.mark.parametrize("mutation", ["ledger", "identity", "hi_allocation", "gate", "target"])
def test_mutations_rejected(mutation):
    ancestor, bridge = load(ANCESTOR), copy.deepcopy(load(BRIDGE))
    if mutation == "ledger":
        bridge["slots"][0]["ledger_input"]["phases"]["proposal"]["scope"] = "per_arm"
    elif mutation == "identity":
        bridge["slots"][1]["root"] = 1
    elif mutation == "hi_allocation":
        bridge["slots"][1]["unallocated_hi_source_row"]["allocation"] = "per_arm"
    elif mutation == "gate":
        bridge["ranking_authorized"] = True
    else:
        bridge["slots"][1]["ledger_input"]["target_sha256"] = bridge["slots"][0]["ledger_input"]["target_sha256"]
    with pytest.raises(AssertionError):
        crosscheck(ancestor, bridge)

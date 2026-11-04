import copy
import json

import pytest

from experiments.spectral_reachability import p3_paired_rank_physical_audit_20261102_codexexclusive as audit


@pytest.fixture(scope="module")
def inputs():
    checked = json.loads(audit.selected.CONSUME.read_text())
    plans = json.loads((audit.ROOT / "results/p3_venv_winner_physical_prepare_20261102_codexexclusive_01.json").read_text())["plans"]
    report = json.loads((audit.PHYSICAL_RUN / "report.json").read_text())
    with (audit.RUN / "events.jsonl").open() as f:
        events = [json.loads(line)["event"] for line in f
                  if '"kind":"candidate_complete"' in line or '"kind": "candidate_complete"' in line]
    return checked, plans, report, events


def test_observed_pair_and_full_ordered_input(inputs):
    paired, roster = audit.analyze(*inputs)
    assert len(roster) == 30
    assert [(p["root_slot"], p["same_step_actions"]) for p in paired] == [(0, 0), (1, 10)]
    assert [(p["arms"]["signal"]["rank_in_union_15"],
             p["arms"]["reverse"]["rank_in_union_15"]) for p in paired] == [(1, 3), (1, 1)]


@pytest.mark.parametrize("fault", ["repeat_cost", "repeat_action", "missing", "winner", "physical_selection"])
def test_reject_inconsistent_pair(inputs, fault):
    checked, plans, report, events = copy.deepcopy(inputs)
    if fault == "repeat_cost":
        next(e for e in events if e["scope"][1:] == ["reverse", "search", 0])["candidate"]["cost"] += 1
    elif fault == "repeat_action":
        next(e for e in events if e["scope"][1:] == ["reverse", "search", 0])["candidate"]["commands"][0][0] = 0.0
    elif fault == "missing":
        events.pop()
    elif fault == "winner":
        plans[0]["candidate_index"] = 0
    else:
        report["rows"][0]["selection_digest"] = "wrong"
    with pytest.raises(ValueError):
        audit.analyze(checked, plans, report, events)

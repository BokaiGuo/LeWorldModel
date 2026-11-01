import copy
import json

import pytest

from experiments.spectral_reachability import p1_packet_objective_physical_20261031 as m


def test_saved_packet_roster_and_score_tamper():
    plan = json.loads(m.PLAN.read_bytes())
    original = json.loads(m.prior.OLD.read_bytes())
    blind = json.loads(m.prior.planner.bundle.OUT.read_bytes())
    roster = m.assemble(plan, original, blind)
    assert len(roster) == 24
    assert all(row["raw10_sha256"] == item["selected_action_sha256"]
               for row, item in zip(roster, plan["rows"], strict=True))
    changed = copy.deepcopy(plan)
    changed["rows"][0]["candidate_scores"][0] += 0.01
    with pytest.raises(ValueError):
        m.assemble(changed, original, blind)
    changed = copy.deepcopy(plan)
    changed["rows"][0]["selected_actions"][0][0][0] += 0.01
    with pytest.raises(ValueError):
        m.assemble(changed, original, blind)

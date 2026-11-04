import copy

import pytest

from experiments.spectral_reachability import p3_preoutcome_physical_reuse_audit_20261102_codexexclusive as audit


def test_known_root_action_reuse_detected():
    result = audit.build()
    assert result["status"] == "all_new_winners_reuse_existing_physical_inputs"
    assert [len(row["old_matches"]) for row in result["mapped_slots"]] == [1, 2, 1, 2]
    assert result["new_physical_observations"] == 0
    assert not result["independent_outcome"]


def test_new_action_does_not_inherit_old_record(monkeypatch):
    plan, data = audit.new.verify(audit.NEW_PREP, audit.NEW_SHA)
    plan = copy.deepcopy(plan)
    plan["plans"][0]["raw_float32_sha256"] = "0" * 64
    monkeypatch.setattr(audit.new, "verify", lambda *args: (plan, data))
    with pytest.raises(ValueError, match="not represented"):
        audit.build()

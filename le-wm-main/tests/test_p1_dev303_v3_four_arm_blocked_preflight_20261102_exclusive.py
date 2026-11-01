import copy
import json

import pytest

from experiments.spectral_reachability import p1_dev303_v3_four_arm_blocked_preflight_20261102_exclusive as m


def inputs():
    return {"report": (m.RUN / "report.json").read_bytes(),
            "events": (m.RUN / "events.jsonl").read_bytes(),
            "model_prepare": m.MODEL_PREP.read_bytes(),
            "cuda_prepare": m.CUDA_PREP.read_bytes()}


def test_real_pinned_inputs_are_blocked_and_actions_fixed():
    result = m.audit_local()
    assert result["status"] == "blocked_uncertified_scorer_and_ledger"
    assert result["physical_execution_authorized"] is False
    assert result["report_score_provenance_verified"] is False
    assert result["report_ledger_verified"] is False
    assert [(r["role"], r["index"]) for r in result["arms_fixed_before_physics"]] == list(m.ROLES)
    assert all(len(r["actions"]) == 5 for r in result["arms_fixed_before_physics"])


@pytest.mark.parametrize("name", list(m.PINS))
def test_source_drift_fails_closed(name):
    data = inputs()
    data[name] += b" "
    with pytest.raises(ValueError, match="SHA mismatch"):
        m.audit(data)


def test_event_or_selection_tampering_fails_even_if_local_pins_replaced(monkeypatch):
    data = inputs()
    report = json.loads(data["report"])
    report["selection"]["selected_index"] = 2
    data["report"] = json.dumps(report).encode()
    pins = copy.copy(m.PINS)
    pins["report"] = m.hashlib.sha256(data["report"]).hexdigest()
    monkeypatch.setattr(m, "PINS", pins)
    with pytest.raises(ValueError, match="selection identity"):
        m.audit(data)


def test_missing_input_is_not_filled():
    data = inputs()
    del data["events"]
    with pytest.raises(ValueError, match="missing or extra"):
        m.audit(data)

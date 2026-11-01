"""CPU-only resource accounting fixtures, not online or model execution."""
from copy import deepcopy

import pytest

from experiments.spectral_reachability import p3_feedback_checkpoint_budget_20261030_v1 as m


def event(index, resource, detail, kind="record", origin="physical"):
    return {"index": index, "resource": resource, "detail": detail, "kind": kind, "origin": origin}


def fixture():
    events = [event(0, "matched_slot_attempt", {"slot": 0}),
              event(1, "native_matched_root_accepted", {}),
              event(3, "native_meter/public_step", {"units": 1}, "intent"),
              event(4, "encoder", {}, "intent", "encoding"),
              event(5, "projector", {}, "intent", "encoding"),
              event(6, "encoding_complete", {"boundary": 5}, origin="encoding"),
              event(10, "encoding_complete", {"boundary": 10}, origin="encoding"),
              event(14, "encoding_complete", {"boundary": 15}, origin="encoding"),
              event(18, "encoding_complete", {"boundary": 20}, origin="encoding"),
              event(20, "matched_slot_return", {"slot": 0})]
    delta = dict.fromkeys(m.budget.EXPECTED, 0)
    delta["predict_API"] = 1
    op = {"delta": delta, "resource": "predict", "shape": [1, 3, 192], "directions": None,
          "status": "returned"}
    wrapped = [{"payload": {"kind": "resource_attempt", "entry": op}},
               {"payload": {"kind": "resource_return", "entry": op}}]
    return events, wrapped, [(2, 2), (7, 7)]


def test_checkpoint_temporal_settlement_and_pending():
    events, wrapped, positions = fixture()
    c = m.checkpoint_costs(events, wrapped, positions, 0)
    assert c["0"]["attempted"]["predict_API"] == 0
    assert c["5"]["attempted"]["predict_API"] == 1
    assert c["5"]["completed"]["predict_API"] == 0
    assert c["10"]["completed"]["predict_API"] == 1
    assert c["5"]["physical_meter_attempted"]["public_step"] == 1
    assert c["5"]["observation_encoder_attempted"] == {"encoder": 1, "projector": 1}


def test_comparison_preserves_axes_shapes_and_unknown():
    events, wrapped, positions = fixture()
    a = m.checkpoint_costs(events, wrapped, positions, 0)
    b = deepcopy(a)
    b["5"]["attempted"]["predict_API"] += 2
    b["10"]["hook_shape_counts"] = {}
    pairs = m.compare(a, b)
    assert not pairs["5"]["numeric_attempted_match"]
    assert pairs["5"]["predict_attempted_signal_minus_fixed"] == 2
    assert not pairs["10"]["hook_shape_multiset_match"]
    assert m.compare(a, None)["20"]["match"] is None


def test_missing_boundary_or_cross_slot_positions_rejected():
    events, wrapped, positions = fixture()
    events[5]["detail"]["boundary"] = 6
    with pytest.raises(ValueError, match="checkpoint markers"):
        m.checkpoint_costs(events, wrapped, positions, 0)
    events, wrapped, positions = fixture()
    with pytest.raises(ValueError, match="outside slot"):
        m.checkpoint_costs(events, wrapped, [(2, 2), (21, 21)], 0)


def test_missing_resource_axis_rejected():
    events, wrapped, positions = fixture()
    wrapped[0]["payload"]["entry"]["delta"].pop("predict_API")
    with pytest.raises(ValueError, match="resource axes"):
        m.checkpoint_costs(events, wrapped, positions, 0)


def test_receipt_reconstruction_mismatch_fails_before_analysis(monkeypatch, tmp_path):
    receipt = {"report_sha256": "a", "completion_sha256": "b", "full_saved_strict": True}
    monkeypatch.setattr(m.budget, "read_pinned", lambda *args: receipt)
    monkeypatch.setattr(m.saved, "consume", lambda *args: dict(receipt, full_saved_strict=False))
    monkeypatch.setattr(m.budget, "inspect", lambda *args: pytest.fail("should fail first"))
    with pytest.raises(ValueError, match="strict receipt"):
        m.analyze(tmp_path, tmp_path / "receipt", "x", "prep", "contract", "sha")


def test_partial_remains_unknown(monkeypatch, tmp_path):
    receipt = {"report_sha256": "a", "completion_sha256": "b", "full_saved_strict": False}
    monkeypatch.setattr(m.budget, "read_pinned", lambda *args: receipt)
    monkeypatch.setattr(m.saved, "consume", lambda *args: receipt)
    monkeypatch.setattr(m.budget, "inspect", lambda *args: {"pairs": [None, None]})
    out = m.analyze(tmp_path, tmp_path / "receipt", "x", "prep", "contract", "sha")
    assert out["status"] == "partial_unmatched_unknown"
    assert out["end_resource_settlement"] is None
    assert out["pairs"][0]["20"]["match"] is None
    assert out["scientific_pass"] is False


def test_incomplete_plan_binding_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(m.budget, "read_pinned", lambda *args: {})
    with pytest.raises(ValueError, match="supplied together"):
        m.analyze(tmp_path, "receipt", "sha", "prep", "contract", "sha", plan_path="plan")


def test_plan_cannot_attach_another_run_or_receipt(monkeypatch, tmp_path):
    receipt = {"report_sha256": "a", "completion_sha256": "b"}
    monkeypatch.setattr(m.budget, "read_pinned", lambda *args: receipt)
    monkeypatch.setattr(m.binding, "verify_plan", lambda *args: {
        "run_path": str(tmp_path / "planned_run"), "receipt_path": str(tmp_path / "planned_receipt")})
    monkeypatch.setattr(m.saved, "consume", lambda *args: pytest.fail("wrong run must not reach strict consumer"))
    with pytest.raises(ValueError, match="plan path"):
        m.analyze(tmp_path, tmp_path / "receipt", "sha", "prep", "contract", "sha",
                  plan_path="plan", plan_sha="hash", reference="external-ref")

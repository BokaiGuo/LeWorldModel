import copy
import sys

import pytest

from experiments.spectral_reachability import codex_e1_hi_unallocated_event_bridge_20261030_v1 as m


@pytest.fixture(scope="module")
def inputs():
    return m.load_verified()


def test_real_rows_are_unallocated_and_ledger_unchanged(inputs):
    base, trace, targets = inputs
    before = copy.deepcopy(base)
    out = m.join(*inputs)
    assert base == before
    assert out["status"] == "blocked_incomplete"
    assert not out["ranking_authorized"] and not out["equal_budget_certified"]
    assert not out["scientific_admission"]
    for index, slot in enumerate(out["slots"]):
        assert slot["ledger_input"] == base["slots"][index]["ledger_input"]
        assert slot["ledger_preflight"]["structural_status"] == "incomplete"
        if index in (1, 4):
            ref = slot["unallocated_hi_source_row"]
            assert ref["row"] == (0 if index == 1 else 2)
            assert ref["allocation"] == "row_execution_only_not_target_causal_cost"
            assert ref["failed_attempts"] == ref["shared_costs"] == "unknown"
            assert ref["axes"]["high_predictor_calls"]["observed_returned_events"] == 14
            assert ref["axes"]["high_input_rows"]["observed_returned_input_rows"] == 98
            assert slot["ledger_input"]["phases"]["proposal"]["receipt_sha256"] is None
        else:
            assert "unallocated_hi_source_row" not in slot


@pytest.mark.parametrize("mutation", ["root", "task", "target", "row", "condition",
                                      "allocation", "axis", "count", "prior", "report"])
def test_invalid_inputs_fail_closed(inputs, mutation):
    base, trace, targets = copy.deepcopy(inputs)
    row = trace["row_execution_projections"][0]
    target = targets["rows"][1]
    if mutation == "root":
        row["root"] = 1
    elif mutation == "task":
        target["task_sha256"] = "0" * 64
    elif mutation == "target":
        target["target_data_sha256"] = "0" * 64
    elif mutation == "row":
        target["origin"]["report_row"] = 1
    elif mutation == "condition":
        row["condition"] = "on"
    elif mutation == "allocation":
        row["allocation"] = "per_target"
    elif mutation == "axis":
        del row["modules"]["high_cost"]
    elif mutation == "count":
        row["modules"]["high_cost"]["returned"] = 0
    elif mutation == "prior":
        base["slots"][1]["ledger_input"]["phases"]["proposal"]["scope"] = "per_arm"
    else:
        trace["report_sha256"] = "0" * 64
    with pytest.raises((ValueError, KeyError)):
        m.join(base, trace, targets)


def test_saved_trace_digest_fails(inputs, tmp_path, monkeypatch):
    fake = tmp_path / "trace.json"
    fake.write_text("{}")
    monkeypatch.setattr(m.hi, "OUTPUT", fake)
    with pytest.raises(ValueError, match="digest"):
        m.load_verified()


def test_recomputed_trace_mismatch(inputs, monkeypatch):
    trace = copy.deepcopy(inputs[1])
    trace["row_execution_projections"][0]["modules"]["high_cost"]["returned"] += 1
    monkeypatch.setattr(m.hi, "analyze", lambda *args: trace)
    with pytest.raises(ValueError, match="recomputation mismatch"):
        m.load_verified()


def test_exclusive_output(inputs, tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    monkeypatch.setattr(m, "OUTPUT", path)
    monkeypatch.setattr(m, "load_verified", lambda: inputs)
    monkeypatch.setattr(sys, "argv", ["audit"])
    assert m.main() == 0
    original = path.read_bytes()
    assert m.main() == 2
    assert path.read_bytes() == original
